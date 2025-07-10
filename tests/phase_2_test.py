import os
import json
import uuid
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import tiktoken
from groq import Groq
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Chat Wrapper API", version="1.0.0")

# Add CORS middleware for frontend integration later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True  # This makes Redis return strings instead of bytes
)

# Initialize tokenizer for GPT-4
try:
    encoding = tiktoken.encoding_for_model("gpt-4")
except Exception:
    encoding = tiktoken.get_encoding("cl100k_base")  # Fallback

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    tokens_used: int
    total_messages: int

class ConversationSummary(BaseModel):
    conversation_id: str
    message_count: int
    created_at: str
    last_updated: str

class ConversationManager:
    def __init__(self):
        self.max_tokens = 8000
        self.reserved_tokens = 1000  # Reserve for response
        self.max_messages_before_compression = 20
        self.semantic_similarity_threshold = 0.3  # Lower threshold for more inclusive selection
        self.simple_conversation_threshold = 6  # Use simple selection for conversations with <= 6 messages
        
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
        """Simple chronological context preparation for short conversations"""
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
        """Smart context preparation with hybrid strategy"""
        conversation = self.get_conversation(conv_id)
        
        # For short conversations, use simple chronological selection
        if len(conversation) <= self.simple_conversation_threshold:
            print(f"Using simple context (conversation length: {len(conversation)})")
            return self.prepare_context_simple(conv_id, user_message)
        
        # For longer conversations, use semantic selection
        print(f"Using semantic context (conversation length: {len(conversation)})")
        return self.prepare_context_semantic(conv_id, user_message)
    
    async def get_groq_response(self, messages: List[Dict]) -> tuple[str, int]:
        """Get response from Groq API"""
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=self.reserved_tokens - 100  # Leave some buffer
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return content, tokens_used
            
        except Exception as e:
            print(f"Groq API Error: {e}")
            raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")
    
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

#region Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Chat Wrapper API is running!",
        "version": "1.0.0",
        "redis_connected": redis_client.ping()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    
    # Generate conversation ID if not provided
    conv_id = request.conversation_id or conv_manager.generate_conversation_id()
    
    try:
        # Prepare context with intelligent token management
        context, estimated_tokens = conv_manager.prepare_context(conv_id, request.message)
        
        # Get response from Groq
        response_content, actual_tokens = await conv_manager.get_groq_response(context)
        
        # Save this exchange
        conv_manager.add_exchange(conv_id, request.message, response_content)
        
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
    """Get full conversation history"""
    conversation = conv_manager.get_conversation(conversation_id)
    metadata = conv_manager.get_conversation_metadata(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversation,
        "metadata": metadata
    }

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        # Delete conversation and metadata
        redis_client.delete(conv_manager.get_conversation_key(conversation_id))
        redis_client.delete(conv_manager.get_metadata_key(conversation_id))
        
        # Delete all embeddings for this conversation
        embedding_keys = redis_client.keys(f"embedding:{conversation_id}:*")
        if embedding_keys:
            redis_client.delete(*embedding_keys)
            
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations")
async def list_conversations():
    """List all conversations (basic implementation)"""
    try:
        # Get all conversation keys
        keys = redis_client.keys("metadata:*")
        conversations = []
        
        for key in keys:
            conv_id = key.replace("metadata:", "")
            metadata = conv_manager.get_conversation_metadata(conv_id)
            if metadata:
                conversations.append({
                    "conversation_id": conv_id,
                    "message_count": metadata.get("message_count", 0),
                    "created_at": metadata.get("created_at"),
                    "last_updated": metadata.get("last_updated")
                })
        
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#endregion Routes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)