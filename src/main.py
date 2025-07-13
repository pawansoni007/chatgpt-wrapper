import os
import json
import uuid
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add the current directory and parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from conversation_memory import EnhancedConversationManager
from thread_detection_system import ThreadAwareConversationManager
from intent_classification_system import EnhancedIntentClassifier
from multilayered_context_builder import MultiLayeredContextBuilder
from models.chat_models import ChatRequest, ChatResponse, ConversationSummary

import redis
import tiktoken
from cerebras.cloud.sdk import Cerebras
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Chat Wrapper API with Intent Classification", version="2.2.0 - Phase 2+3C")

#region Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#endregion CORS middleware

#region Initialize clients
cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True  # This makes Redis return strings instead of bytes
)
#endregion Initialize clients

#region Initialize tokenizer for GPT-4
try:
    encoding = tiktoken.encoding_for_model("gpt-4")
except Exception:
    encoding = tiktoken.get_encoding("cl100k_base")  # Fallback
#endregion Initialize tokenizer for GPT-4

class ConversationManager:
    def __init__(self):
        self.max_tokens = 60000  # Cerebras 65k context window
        self.reserved_tokens = 4000  # Reserve for response
        self.max_messages_before_compression = 50  # More messages with larger context
        self.semantic_similarity_threshold = 0.3  # Slightly lower threshold for larger context
        self.simple_conversation_threshold = 5  # Increased for 60k context (was 6 for 8k)
        
    def get_message_embedding(self, message_content: str) -> List[float]:
        """Get semantic embedding for a message using Cohere"""
        try:
            response = cohere_client.embed(
                texts=[message_content],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings[0]
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_embedding_key(self, conv_id: str, message_index: int) -> str:
        """Get Redis key for storing message embeddings"""
        return f"embedding:{conv_id}:{message_index}"
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough approximation
            return len(text) // 4
    
    def generate_conversation_id(self) -> str:
        """Generate unique conversation ID"""
        return str(uuid.uuid4())
    
    def get_conversation_key(self, conv_id: str) -> str:
        """Get Redis key for conversation"""
        return f"conversation:{conv_id}"
    
    def get_metadata_key(self, conv_id: str) -> str:
        """Get Redis key for conversation metadata"""
        return f"metadata:{conv_id}"
    
    def get_conversation(self, conv_id: str) -> List[Dict]:
        """Get conversation messages from Redis"""
        try:
            data = redis_client.get(self.get_conversation_key(conv_id))
            if data:
                return json.loads(data)
            return []
        except Exception as e:
            print(f"Error getting conversation: {e}")
            return []
    
    def save_conversation(self, conv_id: str, messages: List[Dict]):
        """Save conversation messages to Redis"""
        try:
            redis_client.set(
                self.get_conversation_key(conv_id), 
                json.dumps(messages)
            )
            
            # Update metadata
            metadata = {
                "message_count": len(messages),
                "last_updated": datetime.now().isoformat(),
                "total_tokens": sum(self.count_tokens(msg.get("content", "")) for msg in messages)
            }
            
            # Get existing metadata to preserve created_at
            existing_meta = redis_client.get(self.get_metadata_key(conv_id))
            if existing_meta:
                existing_meta = json.loads(existing_meta)
                metadata["created_at"] = existing_meta.get("created_at")
            else:
                metadata["created_at"] = datetime.now().isoformat()
            
            redis_client.set(
                self.get_metadata_key(conv_id),
                json.dumps(metadata)
            )
            
        except Exception as e:
            print(f"Error saving conversation: {e}")
            raise HTTPException(status_code=500, detail="Failed to save conversation")
    
    def store_message_embedding(self, conv_id: str, message_index: int, message_content: str):
        """Generate and store embedding for a message"""
        try:
            # Generate embedding
            embedding = self.get_message_embedding(message_content)
            if embedding:
                # Store in Redis
                embedding_key = self.get_embedding_key(conv_id, message_index)
                redis_client.set(embedding_key, json.dumps(embedding))
                print(f"Stored embedding for message {message_index} in conversation {conv_id}")
            else:
                print(f"Failed to generate embedding for message {message_index}")
        except Exception as e:
            print(f"Error storing embedding for message {message_index}: {e}")
    
    def get_relevant_messages(self, conv_id: str, user_message: str) -> List[Dict]:
        """Return messages ranked by semantic relevance to user_message"""
        try:
            # Get user message embedding
            user_embedding = self.get_message_embedding(user_message)
            if not user_embedding:
                print("Failed to generate embedding for user message, falling back to chronological")
                return []
            
            # Get all conversation messages
            conversation = self.get_conversation(conv_id)
            if not conversation:
                return []
            
            message_scores = []
            
            for idx, message in enumerate(conversation):
                # Get stored embedding for this message
                embedding_key = self.get_embedding_key(conv_id, idx)
                stored_embedding_data = redis_client.get(embedding_key)
                
                if stored_embedding_data:
                    try:
                        stored_embedding = json.loads(stored_embedding_data)
                        similarity = self.calculate_cosine_similarity(user_embedding, stored_embedding)
                        
                        message_scores.append({
                            'index': idx,
                            'message': message,
                            'similarity': similarity
                        })
                    except Exception as e:
                        print(f"Error processing embedding for message {idx}: {e}")
                        continue
                else:
                    # If no stored embedding, generate and store it
                    message_content = message.get('content', '')
                    if message_content:
                        self.store_message_embedding(conv_id, idx, message_content)
                        # Try to get the newly stored embedding
                        stored_embedding_data = redis_client.get(embedding_key)
                        if stored_embedding_data:
                            try:
                                stored_embedding = json.loads(stored_embedding_data)
                                similarity = self.calculate_cosine_similarity(user_embedding, stored_embedding)
                                message_scores.append({
                                    'index': idx,
                                    'message': message,
                                    'similarity': similarity
                                })
                            except Exception as e:
                                print(f"Error with newly generated embedding for message {idx}: {e}")
            
            # Sort by similarity (highest first)
            message_scores.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Filter by threshold and return
            relevant_messages = [
                msg_data for msg_data in message_scores 
                if msg_data['similarity'] >= self.semantic_similarity_threshold
            ]
            
            print(f"Found {len(relevant_messages)} relevant messages out of {len(conversation)} total")
            return relevant_messages
            
        except Exception as e:
            print(f"Error in get_relevant_messages: {e}")
            return []
    
    def prepare_context_simple(self, conv_id: str, user_message: str) -> tuple[List[Dict], int]:
        """Simple chronological context preparation for shorter conversations"""
        conversation = self.get_conversation(conv_id)
        
        # Calculate token budget
        user_tokens = self.count_tokens(user_message)
        available_tokens = self.max_tokens - user_tokens - self.reserved_tokens
        
        # System message
        system_content = (
            "You are a helpful assistant that maintains conversation context. "
            "Provide thoughtful, accurate responses based on the conversation history."
        )
        system_tokens = self.count_tokens(system_content)
        
        # Start building context
        context = [{"role": "system", "content": system_content}]
        used_tokens = system_tokens
        available_tokens -= system_tokens
        
        # Add messages from most recent to oldest
        selected_messages = []
        for message in reversed(conversation):
            msg_tokens = self.count_tokens(message.get("content", ""))
            if used_tokens + msg_tokens <= available_tokens:
                selected_messages.insert(0, message)  # Insert at beginning to maintain order
                used_tokens += msg_tokens
            else:
                break
        
        context.extend(selected_messages)
        context.append({"role": "user", "content": user_message})
        
        print(f"Simple context: {len(selected_messages)} messages, {used_tokens + user_tokens} tokens")
        return context, used_tokens + user_tokens
    
    def prepare_context_semantic(self, conv_id: str, user_message: str) -> tuple[List[Dict], int]:
        """Semantic context preparation using relevance-based selection"""
        # Calculate token budget
        user_tokens = self.count_tokens(user_message)
        available_tokens = self.max_tokens - user_tokens - self.reserved_tokens
        
        # System message
        system_content = (
            "You are a helpful assistant that maintains conversation context. "
            "Provide thoughtful, accurate responses based on the conversation history."
        )
        system_tokens = self.count_tokens(system_content)
        context = [{"role": "system", "content": system_content}]
        used_tokens = system_tokens
        available_tokens -= system_tokens
        
        # Get messages ranked by relevance
        relevant_messages = self.get_relevant_messages(conv_id, user_message)
        
        if not relevant_messages:
            # Fallback to simple method if semantic fails
            print("Semantic selection failed, falling back to simple method")
            return self.prepare_context_simple(conv_id, user_message)
        
        # Add most relevant messages that fit in token budget
        selected_messages = []
        for msg_data in relevant_messages:
            message = msg_data['message']
            msg_tokens = self.count_tokens(message.get("content", ""))
            
            if used_tokens + msg_tokens <= available_tokens:
                selected_messages.append({
                    'message': message,
                    'similarity': msg_data['similarity'],
                    'original_index': msg_data['index']
                })
                used_tokens += msg_tokens
            else:
                break
        
        # Sort selected messages chronologically to preserve conversation flow
        selected_messages.sort(key=lambda x: x['original_index'])
        
        # Extract just the message objects for context
        final_messages = [msg_data['message'] for msg_data in selected_messages]
        context.extend(final_messages)
        context.append({"role": "user", "content": user_message}) 
        
        # Print debug info
        similarities = [f"{msg_data['similarity']:.3f}" for msg_data in selected_messages]
        print(f"Semantic context: {len(final_messages)} messages, {used_tokens + user_tokens} tokens")
        print(f"Similarities: {similarities}")
        
        return context, used_tokens + user_tokens
    
    def prepare_context(self, conv_id: str, user_message: str) -> tuple[List[Dict], int]:
        """Smart context preparation with hybrid strategy optimized for 60k context"""
        conversation = self.get_conversation(conv_id)
        
        # For shorter conversations, use simple chronological selection
        # Increased threshold due to larger context window
        if len(conversation) <= self.simple_conversation_threshold:
            print(f"Using simple context (conversation length: {len(conversation)})")
            return self.prepare_context_simple(conv_id, user_message)
        
        # For longer conversations, use semantic selection
        print(f"Using semantic context (conversation length: {len(conversation)})")
        return self.prepare_context_semantic(conv_id, user_message)
    
    async def get_cerebras_response(self, messages: List[Dict]) -> tuple[str, int]:
        """Get response from Cerebras API"""
        try:
            response = cerebras_client.chat.completions.create(
                model="llama-3.3-70b",
                messages=messages,
                temperature=0.7,
                max_tokens=self.reserved_tokens - 100  # Leave some buffer
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return content, tokens_used
            
        except Exception as e:
            print(f"Cerebras API Error: {e}")
            raise HTTPException(status_code=500, detail=f"Cerebras API error: {str(e)}")
    
    def add_exchange(self, conv_id: str, user_message: str, assistant_message: str):
        """Add user-assistant exchange to conversation and store embeddings"""
        conversation = self.get_conversation(conv_id)
        
        # Calculate indices for new messages
        user_message_index = len(conversation)
        assistant_message_index = len(conversation) + 1
        
        # Add both messages
        conversation.append({
            "role": "user", 
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        conversation.append({
            "role": "assistant", 
            "content": assistant_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Store embeddings for both messages
        print(f"Storing embeddings for messages {user_message_index} and {assistant_message_index}")
        self.store_message_embedding(conv_id, user_message_index, user_message)
        self.store_message_embedding(conv_id, assistant_message_index, assistant_message)
        
        # Manage conversation length
        if len(conversation) > self.max_messages_before_compression:
            # Keep only the most recent messages
            # Note: This will break embedding indices, but it's a simple approach for now
            # In production, you'd want to implement proper summarization
            conversation = conversation[-self.max_messages_before_compression:]
        
        self.save_conversation(conv_id, conversation)
    
    def get_conversation_metadata(self, conv_id: str) -> Optional[Dict]:
        """Get conversation metadata"""
        try:
            data = redis_client.get(self.get_metadata_key(conv_id))
            if data:
                return json.loads(data)
            return None
        except Exception:
            return None

# Initialize conversation manager
conv_manager = ConversationManager()

# Initialize Phase 3B Thread-Aware Manager
thread_aware_manager = ThreadAwareConversationManager(
    conv_manager,
    cohere_client, 
    redis_client
)

# Initialize Intent Classification System
intent_classifier = EnhancedIntentClassifier(cerebras_client)

# Initialize Phase 2 MultiLayered Context Builder
multilayer_context_builder = MultiLayeredContextBuilder(
    base_conv_manager=conv_manager,
    thread_aware_manager=thread_aware_manager,
    memory_manager=None  # Will add memory integration later
)


#region Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Chat Wrapper API with Intent Classification is running!",
        "version": "2.2.0 - Phase 2+3C",
        "features": ["Intent Classification", "Multilayered Context Building", "Thread Detection", "Topic Lifecycle", "Context Selection"],
        "intent_categories": intent_classifier.get_intent_statistics()["categories"],
        "context_statistics": multilayer_context_builder.get_context_statistics(),
        "redis_connected": redis_client.ping()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with intent-aware conversation management"""
    
    conv_id = request.conversation_id or conv_manager.generate_conversation_id()
    
    try:
        # NEW: Classify user intent first
        recent_context = conv_manager.get_conversation(conv_id)[-4:]  # Last 4 messages for context
        intent_result = intent_classifier.classify_intent(request.message, recent_context)
        
        # Log intent classification results
        print(f"🎯 Intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})")
        print(f"📝 Reasoning: {intent_result.reasoning}")
        
        if intent_result.requires_clarification:
            print(f"⚠️  Clarification needed: {intent_result.clarification_reason}")
        
        # Phase 2: Multilayered Context Building with Intent Awareness
        context_result = multilayer_context_builder.build_context(intent_result, conv_id, request.message)
        context = context_result.context
        estimated_tokens = context_result.tokens_used
        
        print(f"🧵 Context built: Level {context_result.level_used}, Strategy: {context_result.strategy}")
        print(f"📊 Context: {len(context)} messages, {estimated_tokens} tokens")
        
        # Get response from Cerebras
        response_content, actual_tokens = await conv_manager.get_cerebras_response(context)
        
        # Save exchange WITH thread tracking (Phase 3B)
        thread_aware_manager.add_exchange_with_threads(conv_id, request.message, response_content)
        
        # Get updated conversation count
        conversation = conv_manager.get_conversation(conv_id)
        
        return ChatResponse(
            message=response_content,
            conversation_id=conv_id,
            tokens_used=actual_tokens,
            total_messages=len(conversation)
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get full conversation history with thread information"""
    conversation = conv_manager.get_conversation(conversation_id)
    metadata = conv_manager.get_conversation_metadata(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get thread information
    threads = thread_aware_manager.lifecycle_manager.load_threads(conversation_id)
    thread_info = []
    
    for thread in threads:
        thread_info.append({
            "thread_id": thread.thread_id,
            "topic": thread.topic,
            "status": thread.status.value,
            "message_indices": thread.messages,
            "confidence": thread.confidence_score
        })
    
    return {
        "conversation_id": conversation_id,
        "messages": conversation,
        "metadata": metadata,
        "threads": thread_info,
        "thread_summary": {
            "total_threads": len(threads),
            "active_threads": len([t for t in threads if t.status.value == "active"]),
            "topics": [t.topic for t in threads if t.status.value == "active"]
        }
    }

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all associated data"""
    try:
        # Delete conversation and metadata
        redis_client.delete(conv_manager.get_conversation_key(conversation_id))
        redis_client.delete(conv_manager.get_metadata_key(conversation_id))
        
        # Delete all embeddings for this conversation
        embedding_keys = redis_client.keys(f"embedding:{conversation_id}:*")
        if embedding_keys:
            redis_client.delete(*embedding_keys)
        
        # Delete threads (Phase 3B)
        redis_client.delete(thread_aware_manager.lifecycle_manager.get_threads_key(conversation_id))
        
        # Delete memory if exists
        memory_key = f"memory:{conversation_id}"
        redis_client.delete(memory_key)
            
        return {"message": "Conversation and all associated data deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations")
async def list_conversations():
    """List all conversations with thread information"""
    try:
        # Get all conversation keys
        keys = redis_client.keys("metadata:*")
        conversations = []
        
        for key in keys:
            conv_id = key.replace("metadata:", "")
            metadata = conv_manager.get_conversation_metadata(conv_id)
            if metadata:
                # Get thread summary
                threads = thread_aware_manager.lifecycle_manager.load_threads(conv_id)
                
                conv_data = {
                    "conversation_id": conv_id,
                    "message_count": metadata.get("message_count", 0),
                    "created_at": metadata.get("created_at"),
                    "last_updated": metadata.get("last_updated"),
                    "thread_count": len(threads),
                    "active_threads": len([t for t in threads if t.status.value == "active"]),
                    "topics": [t.topic for t in threads if t.status.value == "active"][:3]  # First 3 active topics
                }
                conversations.append(conv_data)
        
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/threads")
async def get_conversation_threads(conversation_id: str):
    """Get conversation threads analysis"""
    try:
        # Load threads
        threads = thread_aware_manager.lifecycle_manager.load_threads(conversation_id)
        
        if not threads:
            # Analyze conversation to create threads
            threads = thread_aware_manager.analyze_conversation_threads(conversation_id)
            thread_aware_manager.lifecycle_manager.save_threads(conversation_id, threads)
        
        # Convert to serializable format
        threads_data = []
        for thread in threads:
            thread_data = thread.to_dict()
            # Add some additional info
            thread_data["message_count"] = len(thread.messages)
            thread_data["is_active"] = thread.status.value == "active"
            threads_data.append(thread_data)
        
        return {
            "conversation_id": conversation_id,
            "threads": threads_data,
            "total_threads": len(threads_data),
            "active_threads": len([t for t in threads if t.status.value == "active"])
        }
        
    except Exception as e:
        print(f"Error getting threads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversations/{conversation_id}/reanalyze-threads")
async def reanalyze_conversation_threads(conversation_id: str):
    """Force re-analysis of conversation threads"""
    try:
        # Re-analyze the conversation
        threads = thread_aware_manager.analyze_conversation_threads(conversation_id)
        
        # Save the new threads
        thread_aware_manager.lifecycle_manager.save_threads(conversation_id, threads)
        
        return {
            "message": "Conversation threads re-analyzed successfully",
            "conversation_id": conversation_id,
            "threads_created": len(threads),
            "threads": [{"id": t.thread_id, "topic": t.topic, "messages": len(t.messages)} for t in threads]
        }
        
    except Exception as e:
        print(f"Error re-analyzing threads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/threads/{conversation_id}")
async def debug_thread_detection(conversation_id: str):
    """Debug endpoint to see how thread detection works"""
    try:
        conversation = conv_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get topic boundaries
        boundaries = thread_aware_manager.boundary_detector.detect_topic_boundaries(conversation)
        
        # Get conversation patterns
        patterns = thread_aware_manager.boundary_detector.detect_conversation_patterns(conversation)
        
        # Get threads
        threads = thread_aware_manager.lifecycle_manager.load_threads(conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "message_count": len(conversation),
            "detected_boundaries": boundaries,
            "conversation_patterns": patterns,
            "threads": [
                {
                    "id": t.thread_id,
                    "topic": t.topic,
                    "status": t.status.value,
                    "messages": t.messages,
                    "confidence": t.confidence_score
                } for t in threads
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/multilayer-context")
async def debug_multilayer_context(request: ChatRequest):
    """Debug endpoint to test multilayered context building"""
    try:
        conv_id = request.conversation_id or "debug_conversation"
        
        # Step 1: Classify intent
        recent_context = conv_manager.get_conversation(conv_id)[-4:] if request.conversation_id else []
        intent_result = intent_classifier.classify_intent(request.message, recent_context)
        
        # Step 2: Build multilayered context
        context_result = multilayer_context_builder.build_context(intent_result, conv_id, request.message)
        
        return {
            "message": request.message,
            "conversation_id": conv_id,
            "intent_classification": {
                "intent": intent_result.intent.value,
                "confidence": intent_result.confidence,
                "reasoning": intent_result.reasoning,
                "requires_clarification": intent_result.requires_clarification,
                "clarification_reason": intent_result.clarification_reason
            },
            "context_building": {
                "level_used": context_result.level_used,
                "strategy": context_result.strategy,
                "tokens_used": context_result.tokens_used,
                "context_messages": len(context_result.context),
                "debug_info": context_result.debug_info
            },
            "context_preview": [
                {
                    "role": msg.get("role", "unknown"),
                    "content_preview": msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", "")
                }
                for msg in context_result.context[:5]  # First 5 messages preview
            ]
        }
        
    except Exception as e:
        print(f"Multilayer context debug error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug/intent")
async def debug_intent_classification(request: ChatRequest):
    """Debug endpoint to test intent classification"""
    try:
        # Get recent context if conversation exists
        recent_context = []
        if request.conversation_id:
            conversation = conv_manager.get_conversation(request.conversation_id)
            recent_context = conversation[-4:] if conversation else []
        
        # Classify intent
        intent_result = intent_classifier.classify_intent(request.message, recent_context)
        
        return {
            "message": request.message,
            "conversation_id": request.conversation_id,
            "intent": intent_result.intent.value,
            "confidence": intent_result.confidence,
            "reasoning": intent_result.reasoning,
            "secondary_intent": intent_result.secondary_intent.value if intent_result.secondary_intent else None,
            "requires_clarification": intent_result.requires_clarification,
            "clarification_reason": intent_result.clarification_reason,
            "context_messages_used": len(recent_context)
        }
        
    except Exception as e:
        print(f"Intent classification debug error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#endregion Routes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["src"]
    )