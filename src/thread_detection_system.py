"""
SIMPLIFIED Thread Detection System

Removed complexity:
- No thread lifecycle management (active/dormant/completed)
- No thread reactivation logic  
- No complex boundary detection
- No semantic anchors and confidence scores

Kept essentials:
- Basic topic clustering for related messages
- Simple context preparation 
- Integration with existing API structure
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ThreadStatus(Enum):
    ACTIVE = "active"      # Only status we use now
    COMPLETED = "completed"  # For API compatibility

@dataclass
class ConversationThread:
    """SIMPLIFIED thread representation"""
    thread_id: str
    messages: List[int]  # Message indices
    topic: str
    status: ThreadStatus
    created_at: datetime
    last_activity: datetime
    confidence_score: float = 0.8  # Fixed score for simplicity
    
    def to_dict(self) -> Dict:
        return {
            "thread_id": self.thread_id,
            "messages": self.messages,
            "topic": self.topic,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "confidence_score": self.confidence_score
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
            confidence_score=data.get("confidence_score", 0.8)
        )

class SimplifiedTopicDetector:
    """SIMPLIFIED topic detection - just basic clustering"""
    
    def __init__(self, cohere_client):
        self.cohere_client = cohere_client
        self.cluster_size = 8  # Messages per cluster/thread
        
    def create_simple_threads(self, messages: List[Dict]) -> List[ConversationThread]:
        """Create simple threads by clustering messages into groups"""
        
        if len(messages) < 4:
            # Too short for threading
            return []
        
        threads = []
        thread_id = 1
        
        # Simple approach: group messages into fixed-size clusters
        for i in range(0, len(messages), self.cluster_size):
            end_idx = min(i + self.cluster_size, len(messages))
            
            if end_idx - i < 3:  # Skip very short groups
                continue
                
            message_indices = list(range(i, end_idx))
            
            # Generate simple topic from first few messages
            topic_messages = []
            for idx in message_indices[:3]:
                if idx < len(messages):
                    content = messages[idx].get("content", "")
                    if content:
                        topic_messages.append(content)
            
            topic = self._generate_simple_topic(topic_messages)
            
            # All threads are active (no complex lifecycle)
            thread = ConversationThread(
                thread_id=f"thread_{thread_id}",
                messages=message_indices,
                topic=topic,
                status=ThreadStatus.ACTIVE,
                created_at=datetime.now() - timedelta(minutes=thread_id*5),
                last_activity=datetime.now(),
                confidence_score=0.8
            )
            
            threads.append(thread)
            thread_id += 1
        
        return threads
    
    def _generate_simple_topic(self, messages: List[str]) -> str:
        """Generate a simple topic name"""
        
        if not messages:
            return "General Discussion"
        
        # Combine messages for topic analysis
        combined_text = " ".join(messages)[:300]  # Limit length
        
        # Simple keyword extraction
        words = combined_text.lower().split()
        
        # Look for common technical terms
        tech_keywords = {
            'react', 'python', 'javascript', 'api', 'database', 'frontend', 
            'backend', 'authentication', 'deployment', 'error', 'bug', 'fix',
            'create', 'build', 'implement', 'feature', 'component', 'function'
        }
        
        found_keywords = [word for word in words if word in tech_keywords]
        
        if found_keywords:
            # Use most common technical term
            most_common = max(set(found_keywords), key=found_keywords.count)
            return f"{most_common.title()} Discussion"
        
        # Fallback to generic topics based on message patterns
        if any(word in combined_text.lower() for word in ['error', 'bug', 'fix', 'problem']):
            return "Debugging Session"
        elif any(word in combined_text.lower() for word in ['create', 'build', 'new', 'start']):
            return "Development Task"
        elif any(word in combined_text.lower() for word in ['how', 'what', 'why', 'explain']):
            return "Learning & Questions"
        else:
            return "General Discussion"

class SimplifiedThreadLifecycleManager:
    """SIMPLIFIED lifecycle management - just basic persistence"""
    
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

class SimplifiedThreadContextSelector:
    """SIMPLIFIED context selection using threads"""
    
    def __init__(self, base_manager):
        self.base_manager = base_manager
    
    def select_thread_context(self, conv_id: str, threads: List[ConversationThread], user_message: str) -> List[Dict]:
        """SIMPLIFIED context selection"""
        
        conversation = self.base_manager.get_conversation(conv_id)
        
        if not conversation or not threads:
            # Fallback to recent messages
            return self._get_recent_context(conversation, user_message)
        
        # Simple approach: use messages from the last thread + some recent context
        last_thread = threads[-1] if threads else None
        
        if last_thread:
            # Get messages from last thread
            thread_messages = []
            for idx in last_thread.messages[-6:]:  # Last 6 from thread
                if idx < len(conversation):
                    thread_messages.append(conversation[idx])
            
            # Add some recent messages not in thread
            recent_start = max(0, len(conversation) - 4)
            recent_messages = conversation[recent_start:]
            
            # Combine without duplicates
            all_messages = []
            used_indices = set(last_thread.messages[-6:])
            
            # Add thread messages
            all_messages.extend(thread_messages)
            
            # Add recent messages not already included
            for i, msg in enumerate(recent_messages, start=recent_start):
                if i not in used_indices:
                    all_messages.append(msg)
            
            # Sort by original order and take last 12
            all_messages = all_messages[-12:]
            
        else:
            all_messages = conversation[-10:]  # Fallback
        
        return self._build_context_with_system(all_messages, user_message)
    
    def _get_recent_context(self, conversation: List[Dict], user_message: str) -> List[Dict]:
        """Fallback to recent messages"""
        recent_messages = conversation[-10:] if conversation else []
        return self._build_context_with_system(recent_messages, user_message)
    
    def _build_context_with_system(self, messages: List[Dict], user_message: str) -> List[Dict]:
        """Build context with system message"""
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant. Provide thoughtful responses based on the conversation history."
        }
        
        context = [system_msg]
        context.extend(messages)
        context.append({"role": "user", "content": user_message})
        
        return context

class ThreadAwareConversationManager:
    """
    SIMPLIFIED Thread-aware conversation manager
    
    Removed complexity:
    - No thread reactivation
    - No complex boundary detection  
    - No thread lifecycle transitions
    - No semantic anchors
    
    Kept essentials:
    - Basic thread grouping
    - Simple context preparation
    - API compatibility
    """
    
    def __init__(self, base_manager, cohere_client, redis_client):
        self.base_manager = base_manager
        self.cohere_client = cohere_client
        
        # SIMPLIFIED components
        self.boundary_detector = SimplifiedTopicDetector(cohere_client)
        self.lifecycle_manager = SimplifiedThreadLifecycleManager(redis_client)
        self.context_selector = SimplifiedThreadContextSelector(base_manager)
    
    def add_exchange_with_threads(self, conv_id: str, user_message: str, assistant_message: str):
        """SIMPLIFIED: Add exchange without complex thread updates"""
        
        # Add exchange using base manager
        self.base_manager.add_exchange(conv_id, user_message, assistant_message)
        
        # Simple thread update: just mark that we have new messages
        # No complex analysis or reactivation needed
        print(f"✅ Added exchange for conversation {conv_id}")
        
        return True
    
    def analyze_conversation_threads(self, conv_id: str) -> List[ConversationThread]:
        """SIMPLIFIED: Create basic threads"""
        
        conversation = self.base_manager.get_conversation(conv_id)
        
        if len(conversation) < 4:
            return []  # Too short for threading
        
        threads = self.boundary_detector.create_simple_threads(conversation)
        
        print(f"📊 Created {len(threads)} simple threads for conversation {conv_id}")
        
        return threads
    
    def prepare_context_with_threads(self, conv_id: str, user_message: str) -> Tuple[List[Dict], int]:
        """SIMPLIFIED: Prepare context using simple thread logic"""
        
        # Load or create threads
        threads = self.lifecycle_manager.load_threads(conv_id)
        
        if not threads:
            threads = self.analyze_conversation_threads(conv_id)
            if threads:  # Only save if we created any
                self.lifecycle_manager.save_threads(conv_id, threads)
        
        # Select context using simplified logic
        context = self.context_selector.select_thread_context(conv_id, threads, user_message)
        
        # Calculate token usage
        total_tokens = sum(self.base_manager.count_tokens(msg.get("content", "")) for msg in context)
        
        print(f"🔍 Thread context: {len(context)} messages, {total_tokens} tokens")
        
        return context, total_tokens
    
    def update_threads_with_new_message(self, conv_id: str, message_index: int):
        """SIMPLIFIED: No-op for API compatibility"""
        # In the simplified version, we don't do complex per-message thread updates
        pass

# Backward compatibility aliases
TopicBoundaryDetector = SimplifiedTopicDetector  # For existing imports
