import os
import json
import uuid
import numpy as np
import time
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

from thread_detection_system import ThreadAwareConversationManager
from intent_classification_system import EnhancedIntentClassifier
from smart_context_selector import SmartContextSelector
from models.chat_models import ChatRequest, ChatResponse, ConversationSummary
from conversation_logger import get_conversation_logger, log_conversation_turn, log_context_selection, LogLevel, get_strategy_rankings, get_performance_summary

# Azure OpenAI client
from llm_client import AzureOpenAIClient

import redis
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Chat Wrapper API with Azure OpenAI", version="3.0.0 - Azure OpenAI Only")

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
# Azure OpenAI client (replaces Cerebras and Cohere)
azure_client = AzureOpenAIClient()

# Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
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
        self.max_tokens = 120000  # Azure OpenAI supports large context
        self.reserved_tokens = 4000  # Reserve for response
        self.max_messages_before_compression = 100  # More messages with larger context
        self.semantic_similarity_threshold = 0.3  
        self.simple_conversation_threshold = 8
        
        # Use Azure OpenAI client
        self.azure_client = azure_client
        
    def get_message_embedding(self, message_content: str) -> List[float]:
        """Get semantic embedding using Azure OpenAI (replaces Cohere)"""
        try:
            return self.azure_client.get_embedding(message_content)
        except Exception as e:
            print(f"Error getting Azure OpenAI embedding: {e}")
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
        """Count tokens using Azure OpenAI tokenizer"""
        return self.azure_client.count_tokens(text)
    
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
        """Generate and store embedding for a message using Azure OpenAI"""
        try:
            # Generate embedding using Azure OpenAI
            embedding = self.get_message_embedding(message_content)
            if embedding:
                # Store in Redis
                embedding_key = self.get_embedding_key(conv_id, message_index)
                redis_client.set(embedding_key, json.dumps(embedding))
                print(f"Stored Azure OpenAI embedding for message {message_index} in conversation {conv_id}")
            else:
                print(f"Failed to generate Azure OpenAI embedding for message {message_index}")
        except Exception as e:
            print(f"Error storing embedding for message {message_index}: {e}")
    
    async def get_azure_response(self, messages: List[Dict], task_type: str = "chat") -> tuple[str, int]:
        """Get response from Azure OpenAI"""
        try:
            response = await self.azure_client.chat_completion(
                messages=messages,
                task_type=task_type,
                temperature=0.7,
                max_tokens=self.reserved_tokens - 100  # Leave some buffer
            )
            
            print(f"✅ Azure OpenAI Response using {response.model_used}")
            
            return response.content, response.tokens_used
            
        except Exception as e:
            print(f"Azure OpenAI API Error: {e}")
            raise HTTPException(status_code=500, detail=f"Azure OpenAI API error: {str(e)}")
    
    def add_exchange(self, conv_id: str, user_message: str, assistant_message: str):
        """Add user-assistant exchange to conversation and store Azure OpenAI embeddings"""
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
        
        # Store embeddings using Azure OpenAI
        print(f"Storing Azure OpenAI embeddings for messages {user_message_index} and {assistant_message_index}")
        self.store_message_embedding(conv_id, user_message_index, user_message)
        self.store_message_embedding(conv_id, assistant_message_index, assistant_message)
        
        # Manage conversation length
        if len(conversation) > self.max_messages_before_compression:
            # Keep only the most recent messages
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
# Initialize Intent Classification System with Azure OpenAI
intent_classifier = EnhancedIntentClassifier(azure_client)

# Initialize Thread-Aware Manager with Azure OpenAI and Intent Classifier
thread_aware_manager = ThreadAwareConversationManager(
    conv_manager,
    azure_client,  # Pass Azure client instead of Cohere
    redis_client,
    intent_classifier  # Pass intent classifier for smart context selection
)



# Initialize SmartContextSelector with Azure OpenAI
smart_context_selector = SmartContextSelector(
    cohere_client=azure_client,  # Use Azure client for embeddings
    conversation_manager=conv_manager,
    thread_manager=thread_aware_manager,
    memory_manager=None
)

#region Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    
    # Get Azure OpenAI client status
    azure_status = azure_client.get_status()
    
    return {
        "message": "Chat Wrapper API with Azure OpenAI (No Fallbacks) is running!",
        "version": "3.0.0 - Azure OpenAI Only",
        "features": [
            "Azure OpenAI GPT-4o for Chat", 
            "Azure OpenAI GPT-4o-mini for Intent Classification",
            "Azure OpenAI Embeddings (replaced Cohere)",
            "120k Token Context Window",
            "Intent Classification", 
            "Context Selection", 
            "Thread Detection", 
            "Semantic Search"
        ],
        "azure_openai": azure_status,
        "improvements": {
            "model_upgrade": "GPT-4o/GPT-4o-mini (no more Llama)",
            "embeddings_upgrade": "Azure OpenAI embeddings (replaced Cohere)",
            "context_window": "120k tokens",
            "response_quality": "Significantly improved",
            "no_fallbacks": "Clean, simple Azure OpenAI only"
        },
        "intent_categories": intent_classifier.get_intent_statistics()["categories"],
        "smart_context_stats": smart_context_selector.get_performance_stats(),
        "redis_connected": redis_client.ping()
    }

@app.post("/chat", response_model=ChatResponse) 
async def chat(request: ChatRequest): 
    """Enhanced chat endpoint with Azure OpenAI (no fallbacks)""" 
    
    conv_id = request.conversation_id or conv_manager.generate_conversation_id() 
    request_start_time = time.time() 
    
    try: 
        print(f"\n===== Azure OpenAI Chat Request for conv_id: {conv_id} =====") 
        
        # --- Stage 1: Classify Intent using Azure OpenAI GPT-4o-mini --- 
        intent_start_time = time.time() 
        print("1️⃣ Classifying intent with Azure OpenAI GPT-4o-mini...") 
        recent_context = conv_manager.get_conversation(conv_id)[-6:] 
        intent_result = await intent_classifier.classify_intent(request.message, recent_context, conv_id) 
        intent_duration = time.time() - intent_start_time 
        
        print(f"   Intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})") 
        
        if intent_result.requires_clarification: 
            print(f"   Clarification needed: {intent_result.clarification_reason}") 
        
        # --- Stage 2: Get Intent-Aware Thread Context ---
        context_start_time = time.time()
        print("2️⃣ Building intent-aware thread context...")
        
        # Use the new thread-aware context selection with intent
        context, tokens_used, strategy_used = await thread_aware_manager.prepare_context_with_threads(
            conv_id, 
            request.message,
            intent_result,  # Pass intent classification result
            token_limit=120000
        )
        
        context_duration = time.time() - context_start_time
        
        print(f"   Intent-Based Strategy: {strategy_used}")
        print(f"   Context size: {len(context)} messages, {tokens_used} tokens")
        print(f"   Intent: {intent_result.intent.value} → Strategy: {strategy_used}") 
        
        # --- Stage 3: Generate Response with Azure OpenAI GPT-4o --- 
        response_start_time = time.time() 
        print("3️⃣ Generating response with Azure OpenAI GPT-4o...") 
        response_content, actual_tokens = await conv_manager.get_azure_response(context, task_type="chat")
        response_duration = time.time() - response_start_time 
        
        # --- Stage 4: Save and Update --- 
        print("4️⃣ Saving exchange and updating threads...") 
        thread_aware_manager.add_exchange_with_threads(conv_id, request.message, response_content) 
        
        conversation = conv_manager.get_conversation(conv_id) 
        
        # Calculate total response time 
        total_response_time = time.time() - request_start_time 
        
        print(f"===== Azure OpenAI Request Complete. Total messages: {len(conversation)} =====\n") 
        
        return ChatResponse( 
            message=response_content, 
            conversation_id=conv_id, 
            tokens_used=actual_tokens, 
            total_messages=len(conversation) 
        ) 
    
    except Exception as e: 
        print(f"Chat error: {e}") 
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/azure-status")
async def get_azure_status():
    """Get Azure OpenAI client status"""
    return azure_client.get_status()

@app.get("/thread-metrics")
async def get_thread_metrics():
    """Get thread detection and context selection performance metrics"""
    try:
        metrics = thread_aware_manager.get_performance_metrics()
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")

@app.post("/tune-thread-parameters")
async def tune_thread_parameters(parameters: dict):
    """
    Runtime tuning of thread detection parameters
    
    Accepted parameters:
    - dbscan_eps: float (0.1-1.0) - DBSCAN clustering sensitivity
    - dbscan_min_samples: int (2-10) - Minimum samples per cluster
    - continuation_threshold: float (0.5-0.95) - Token threshold for continuation strategy
    - semantic_threshold: float (0.5-0.95) - Token threshold for semantic strategy  
    - max_threads_to_check: int (5-50) - Maximum threads to check for similarity
    - intent_confidence_threshold: float (0.5-0.9) - Minimum confidence for intent
    """
    try:
        # Validate parameters
        valid_params = {}
        
        if 'dbscan_eps' in parameters:
            eps = float(parameters['dbscan_eps'])
            if 0.1 <= eps <= 1.0:
                valid_params['dbscan_eps'] = eps
            else:
                raise ValueError("dbscan_eps must be between 0.1 and 1.0")
        
        if 'dbscan_min_samples' in parameters:
            min_samples = int(parameters['dbscan_min_samples'])
            if 2 <= min_samples <= 10:
                valid_params['dbscan_min_samples'] = min_samples
            else:
                raise ValueError("dbscan_min_samples must be between 2 and 10")
        
        if 'continuation_threshold' in parameters:
            threshold = float(parameters['continuation_threshold'])
            if 0.5 <= threshold <= 0.95:
                valid_params['continuation_threshold'] = threshold
            else:
                raise ValueError("continuation_threshold must be between 0.5 and 0.95")
                
        if 'semantic_threshold' in parameters:
            threshold = float(parameters['semantic_threshold'])
            if 0.5 <= threshold <= 0.95:
                valid_params['semantic_threshold'] = threshold
            else:
                raise ValueError("semantic_threshold must be between 0.5 and 0.95")
        
        if 'max_threads_to_check' in parameters:
            max_threads = int(parameters['max_threads_to_check'])
            if 5 <= max_threads <= 50:
                valid_params['max_threads_to_check'] = max_threads
            else:
                raise ValueError("max_threads_to_check must be between 5 and 50")
        
        if 'intent_confidence_threshold' in parameters:
            threshold = float(parameters['intent_confidence_threshold'])
            if 0.5 <= threshold <= 0.9:
                valid_params['intent_confidence_threshold'] = threshold
            else:
                raise ValueError("intent_confidence_threshold must be between 0.5 and 0.9")
        
        if not valid_params:
            raise ValueError("No valid parameters provided")
        
        # Apply tuning
        thread_aware_manager.tune_parameters(**valid_params)
        
        return {
            "status": "success",
            "updated_parameters": valid_params,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tuning parameters: {str(e)}")

# [Rest of the routes remain the same - get_conversation, delete_conversation, etc.]
# I'll add key ones but keeping response shorter

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
            "confidence": thread.confidence_score,
            "summary": thread.summary  # New
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
        
        # Delete threads
        redis_client.delete(thread_aware_manager.lifecycle_manager.get_threads_key(conversation_id))
        
        return {"message": "Conversation and all associated data deleted successfully"}
    except Exception as e:
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
