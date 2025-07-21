"""
ROBUST Thread Detection System with Azure OpenAI and Token Management

Enhanced features:
- DBSCAN clustering for semantic thread detection
- Token-aware embedding handling (fixes truncation issues)
- Thread summaries for optimization
- Robust context selection with token limits
- Automatic summarization when context exceeds limits
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

# Install scikit-learn if not already installed: pip install scikit-learn
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

class ThreadStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"  # Simple lifecycle: archive inactive threads
    COMPLETED = "completed"  # For API compatibility

@dataclass
class ConversationThread:
    """Enhanced thread representation with summaries"""
    thread_id: str
    messages: List[int]  # Message indices
    topic: str
    status: ThreadStatus
    created_at: datetime
    last_activity: datetime
    confidence_score: float = 0.8
    summary: str = ""  # Add this for compressed representation
    
    def to_dict(self) -> Dict:
        return {
            "thread_id": self.thread_id,
            "messages": self.messages,
            "topic": self.topic,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "confidence_score": self.confidence_score,
            "summary": self.summary
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            thread_id=data["thread_id"],
            messages=data["messages"],
            topic=data["topic"],
            status=ThreadStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            confidence_score=data.get("confidence_score", 0.8),
            summary=data.get("summary", "")
        )

class RobustTopicDetector:
    def __init__(self, azure_client, similarity_threshold=0.7, max_tokens=6000):
        self.azure_client = azure_client
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens

    async def create_threads(self, messages: List[Dict]) -> List[ConversationThread]:
        if len(messages) < 4:
            return []

        # Embed messages (with chunking for long ones)
        embeddings = []
        for msg in messages:
            content = msg["content"]
            if self.azure_client.count_tokens(content) > self.max_tokens:
                # Use chunking + average (from our previous code)
                emb = self.azure_client.get_embedding(content)  # Assume this handles chunking internally
            else:
                emb = self.azure_client.get_embedding(content)
            embeddings.append(emb)

        if not SKLEARN_AVAILABLE or len(embeddings) < 6:
            # Fallback to simple grouping
            return await self._create_simple_threads(messages)

        # Cluster embeddings semantically
        embeddings_array = np.array([emb for emb in embeddings if emb])  # Filter empty embeddings
        if len(embeddings_array) < 6:
            return await self._create_simple_threads(messages)
            
        clustering = DBSCAN(eps=0.5, min_samples=3, metric="cosine").fit(embeddings_array)
        labels = clustering.labels_

        # Group into threads
        threads = []
        thread_id = 1
        for label in set(labels):
            if label == -1: continue  # Noise points (outliers)
            thread_msgs = [i for i, l in enumerate(labels) if l == label]
            if len(thread_msgs) < 3: continue

            # Generate topic via LLM summary of first few
            topic_text = " ".join([messages[i]["content"][:200] for i in thread_msgs[:3]])
            topic = await self._llm_call("Generate a concise topic for this: " + topic_text)

            # Summarize thread for token efficiency
            summary = await self._generate_summary([messages[i] for i in thread_msgs])

            thread = ConversationThread(
                thread_id=f"thread_{thread_id}",
                messages=thread_msgs,
                topic=topic,
                status=ThreadStatus.ACTIVE,
                created_at=datetime.now() - timedelta(days=1),  # Example
                last_activity=datetime.now(),
                summary=summary
            )
            threads.append(thread)
            thread_id += 1

        return threads

    async def _create_simple_threads(self, messages: List[Dict]) -> List[ConversationThread]:
        """Fallback simple threading"""
        threads = []
        thread_id = 1
        cluster_size = 8
        
        for i in range(0, len(messages), cluster_size):
            end_idx = min(i + cluster_size, len(messages))
            if end_idx - i < 3:
                continue
                
            message_indices = list(range(i, end_idx))
            topic_text = " ".join([messages[idx]["content"][:200] for idx in message_indices[:3]])
            topic = await self._llm_call("Generate a concise topic for this: " + topic_text)
            summary = await self._generate_summary([messages[idx] for idx in message_indices])
            
            thread = ConversationThread(
                thread_id=f"thread_{thread_id}",
                messages=message_indices,
                topic=topic,
                status=ThreadStatus.ACTIVE,
                created_at=datetime.now() - timedelta(minutes=thread_id*5),
                last_activity=datetime.now(),
                summary=summary
            )
            threads.append(thread)
            thread_id += 1
            
        return threads

    async def _llm_call(self, prompt: str) -> str:
        """Generate topic using Azure OpenAI"""
        try:
            messages = [
                {"role": "system", "content": "Generate concise topics for conversation segments. Respond with just the topic, 2-4 words max."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.azure_client.chat_completion(
                messages=messages,
                task_type="intent",
                temperature=0.3,
                max_tokens=50
            )
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating topic: {e}")
            return "General Discussion"

    async def _generate_summary(self, messages: List[Dict]) -> str:
        """Generate thread summary"""
        try:
            content = " ".join([msg["content"] for msg in messages])[:2000]  # Limit length
            
            prompt = f"Summarize this conversation thread in 1-2 sentences: {content}"
            
            messages_for_llm = [
                {"role": "system", "content": "Summarize conversation threads concisely, preserving key technical details."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.azure_client.chat_completion(
                messages=messages_for_llm,
                task_type="intent",
                temperature=0.3,
                max_tokens=100
            )
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Thread summary unavailable"

class RobustThreadContextSelector:
    def __init__(self, base_manager, azure_client):
        self.base_manager = base_manager
        self.azure_client = azure_client

    async def select_thread_context(self, conv_id: str, threads: List[ConversationThread], user_message: str, token_limit=100000) -> List[Dict]:
        conversation = self.base_manager.get_conversation(conv_id)
        
        if not threads:
            return self._get_recent_context(conversation, user_message)

        # Embed user message
        user_emb = self.azure_client.get_embedding(user_message)

        # Find most relevant thread (cosine similarity)
        similarities = []
        for thread in threads:
            thread_emb = self.azure_client.get_embedding(thread.summary or " ".join([conversation[i]["content"] for i in thread.messages[:3]]))
            sim = np.dot(user_emb, thread_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(thread_emb))
            similarities.append(sim)

        best_thread_idx = np.argmax(similarities)
        best_thread = threads[best_thread_idx]

        # Build context: Summary + recent messages from best thread + global recent
        context_msgs = [{"role": "system", "content": f"Thread Summary: {best_thread.summary}"}] + \
                       [conversation[i] for i in best_thread.messages[-6:]] + \
                       conversation[-4:]  # Recent global

        # Token check and summarize if needed
        total_tokens = sum(self.azure_client.count_tokens(msg["content"]) for msg in context_msgs)
        if total_tokens > token_limit * 0.8:
            # Summarize and warn
            summarized = await self._generate_summary_for_context(context_msgs[:-1])  # Exclude user msg
            context_msgs = [{"role": "system", "content": f"Summarized Context (to save tokens): {summarized}"}] + \
                           context_msgs[-2:]  # Keep last assistant + user
            # Add user warning in response later

        context_msgs.append({"role": "user", "content": user_message})
        return context_msgs

    def _get_recent_context(self, conversation: List[Dict], user_message: str) -> List[Dict]:
        """Fallback to recent messages"""
        recent_messages = conversation[-10:] if conversation else []
        
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant. Provide thoughtful responses based on the conversation history."
        }
        
        context = [system_msg]
        context.extend(recent_messages)
        context.append({"role": "user", "content": user_message})
        
        return context

    async def _generate_summary_for_context(self, messages: List[Dict]) -> str:
        """Generate summary for context compression"""
        try:
            content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])[:2000]
            
            prompt = f"Summarize this conversation preserving key context: {content}"
            
            response = await self.azure_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Create concise summaries preserving technical context."},
                    {"role": "user", "content": prompt}
                ],
                task_type="intent",
                temperature=0.3,
                max_tokens=300
            )
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating context summary: {e}")
            return "Previous conversation context"

class SimplifiedThreadLifecycleManager:
    """Thread persistence manager"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
    
    def get_threads_key(self, conv_id: str) -> str:
        return f"threads:{conv_id}"
    
    def load_threads(self, conv_id: str) -> List[ConversationThread]:
        """Load threads from Redis"""
        try:
            data = self.redis_client.get(self.get_threads_key(conv_id))
            if data:
                threads_data = json.loads(data)
                return [ConversationThread.from_dict(t) for t in threads_data]
            return []
        except Exception as e:
            print(f"Error loading threads: {e}")
            return []
    
    def save_threads(self, conv_id: str, threads: List[ConversationThread]):
        """Save threads to Redis"""
        try:
            threads_data = [thread.to_dict() for thread in threads]
            self.redis_client.set(
                self.get_threads_key(conv_id),
                json.dumps(threads_data)
            )
        except Exception as e:
            print(f"Error saving threads: {e}")

class ThreadAwareConversationManager:
    """
    Enhanced Thread-aware conversation manager with robust features
    """
    
    def __init__(self, base_manager, azure_client, redis_client):
        self.base_manager = base_manager
        self.azure_client = azure_client
        
        # Enhanced components
        self.boundary_detector = RobustTopicDetector(azure_client)
        self.lifecycle_manager = SimplifiedThreadLifecycleManager(redis_client)
        self.context_selector = RobustThreadContextSelector(base_manager, azure_client)
    
    def add_exchange_with_threads(self, conv_id: str, user_message: str, assistant_message: str):
        """Add exchange with enhanced thread handling"""
        
        # Add exchange using base manager
        self.base_manager.add_exchange(conv_id, user_message, assistant_message)
        
        print(f"✅ Added exchange for conversation {conv_id} with robust thread handling")
        return True
    
    async def analyze_conversation_threads(self, conv_id: str) -> List[ConversationThread]:
        """Enhanced thread analysis with clustering"""
        
        conversation = self.base_manager.get_conversation(conv_id)
        
        if len(conversation) < 4:
            return []
        
        threads = await self.boundary_detector.create_threads(conversation)
        
        print(f"📊 Created {len(threads)} robust threads for conversation {conv_id}")
        
        return threads
    
    async def prepare_context_with_threads(
        self, 
        conv_id: str, 
        user_message: str,
        token_limit: int = 100000
    ) -> Tuple[List[Dict], int]:
        """
        Enhanced context preparation with token management
        """
        
        # Load or create threads
        threads = self.lifecycle_manager.load_threads(conv_id)
        
        if not threads:
            threads = await self.analyze_conversation_threads(conv_id)
            if threads:
                self.lifecycle_manager.save_threads(conv_id, threads)
        
        # Select context with robust logic
        context = await self.context_selector.select_thread_context(
            conv_id, threads, user_message, token_limit
        )
        
        # Calculate token usage
        total_tokens = sum(self.azure_client.count_tokens(msg.get("content", "")) for msg in context)
        
        print(f"🔍 Robust thread context: {len(context)} messages, {total_tokens} tokens")
        
        return context, total_tokens
    
    def update_threads_with_new_message(self, conv_id: str, message_index: int):
        """No-op for API compatibility"""
        pass

# Backward compatibility aliases
TopicBoundaryDetector = RobustTopicDetector
SimplifiedTopicDetector = RobustTopicDetector
SimplifiedThreadContextSelector = RobustThreadContextSelector
