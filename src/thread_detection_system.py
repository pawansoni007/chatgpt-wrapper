import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ThreadStatus(Enum):
    ACTIVE = "active"
    DORMANT = "dormant" 
    COMPLETED = "completed"
    ARCHIVED = "archived"

@dataclass
class ConversationThread:
    thread_id: str
    messages: List[int]  # Message indices in the conversation
    topic: str
    status: ThreadStatus
    created_at: datetime
    last_activity: datetime
    confidence_score: float
    semantic_anchor: List[float]  # Embedding representing thread topic
    subtopics: List[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "thread_id": self.thread_id,
            "messages": self.messages,
            "topic": self.topic,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "confidence_score": self.confidence_score,
            "semantic_anchor": self.semantic_anchor,
            "subtopics": self.subtopics or []
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
            confidence_score=data["confidence_score"],
            semantic_anchor=data["semantic_anchor"],
            subtopics=data.get("subtopics", [])
        )

class TopicBoundaryDetector:
    """Detects topic boundaries and conversation flow changes"""
    
    def __init__(self, cohere_client):
        self.cohere_client = cohere_client
        
        # Configuration
        self.similarity_threshold = 0.6  # Lower = more sensitive to topic changes
        self.window_size = 3  # Messages to consider for boundary detection
        self.min_thread_length = 2  # Minimum messages to form a thread
        self.topic_shift_threshold = 0.4  # Threshold for detecting major topic shifts
        
    def get_message_embedding(self, message_content: str) -> List[float]:
        """Get semantic embedding for a message using Cohere"""
        try:
            response = self.cohere_client.embed(
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
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception:
            return 0.0
    
    def detect_topic_boundaries(self, messages: List[Dict]) -> List[int]:
        """Detect points where topics change significantly"""
        if len(messages) < 2:
            return []
        
        boundaries = []
        embeddings = []
        
        # First, get conversation patterns
        patterns = self.detect_conversation_patterns(messages)
        
        # Generate embeddings for all messages
        for msg in messages:
            content = msg.get("content", "")
            if content:
                embedding = self.get_message_embedding(content)
                embeddings.append(embedding)
            else:
                embeddings.append([])
        
        # Sliding window analysis with pattern override
        for i in range(self.window_size, len(messages)):
            
            # Check for explicit continuation signals first
            if i in patterns["continuation_signals"]:
                print(f"Continuation signal detected at message {i} - skipping boundary")
                continue  # Skip boundary detection for continuations
                
            # Check for explicit topic shift signals
            if i in patterns["topic_shifts"]:
                boundaries.append(i)
                print(f"Explicit topic shift detected at message {i}")
                continue
            
            # Compare current window with previous window
            current_similarities = []
            
            for j in range(max(0, i - self.window_size), i):
                if embeddings[j] and embeddings[i]:
                    similarity = self.calculate_cosine_similarity(embeddings[j], embeddings[i])
                    current_similarities.append(similarity)
            
            # If average similarity drops below threshold, mark as boundary
            if current_similarities:
                avg_similarity = sum(current_similarities) / len(current_similarities)
                if avg_similarity < self.similarity_threshold:
                    boundaries.append(i)
                    print(f"Topic boundary detected at message {i}, similarity: {avg_similarity:.3f}")
        
        return boundaries
    
    def detect_conversation_patterns(self, messages: List[Dict]) -> Dict:
        """Analyze conversation patterns for better boundary detection"""
        patterns = {
            "qa_pairs": [],
            "topic_shifts": [],
            "continuation_signals": [],
            "closure_signals": []
        }
        
        # Look for Q&A patterns
        for i in range(len(messages) - 1):
            current = messages[i]
            next_msg = messages[i + 1]
            
            if (current.get("role") == "user" and 
                next_msg.get("role") == "assistant"):
                patterns["qa_pairs"].append(i)
        
        # Look for explicit topic shift signals
        shift_keywords = [
            "let's talk about", "moving on", "by the way", "speaking of",
            "on another note", "changing topics", "different question",
            "new topic", "switching gears"
        ]
        
        for i, msg in enumerate(messages):
            content = msg.get("content", "").lower()
            if any(keyword in content for keyword in shift_keywords):
                patterns["topic_shifts"].append(i)
        
        # Look for continuation signals
        continuation_keywords = [
            "also", "additionally", "furthermore", "moreover",
            "building on that", "related to that", "similarly",
            "back to", "returning to", "about that", "continuing with",
            "going back to", "as for", "regarding"
        ]
        
        for i, msg in enumerate(messages):
            content = msg.get("content", "").lower()
            if any(keyword in content for keyword in continuation_keywords):
                patterns["continuation_signals"].append(i)
        
        # Look for closure signals
        closure_keywords = [
            "thanks", "that helps", "got it", "perfect", "understood",
            "makes sense", "that's all", "no more questions"
        ]
        
        for i, msg in enumerate(messages):
            content = msg.get("content", "").lower()
            if any(keyword in content for keyword in closure_keywords):
                patterns["closure_signals"].append(i)
        
        return patterns
    
    def create_threads_from_boundaries(self, messages: List[Dict], boundaries: List[int]) -> List[ConversationThread]:
        """Create conversation threads based on detected boundaries"""
        threads = []
        thread_id = 1
        
        # Add start and end boundaries
        all_boundaries = [0] + boundaries + [len(messages)]
        
        for i in range(len(all_boundaries) - 1):
            start_idx = all_boundaries[i]
            end_idx = all_boundaries[i + 1]
            
            # Skip very short threads
            if end_idx - start_idx < self.min_thread_length:
                continue
            
            thread_messages = list(range(start_idx, end_idx))
            
            # Generate thread topic
            thread_content = []
            for msg_idx in thread_messages[:5]:  # Use first 5 messages for topic
                if msg_idx < len(messages):
                    thread_content.append(messages[msg_idx].get("content", ""))
            
            topic = self.generate_thread_topic(thread_content)
            
            # Create semantic anchor (average embedding of thread messages)
            semantic_anchor = self.create_semantic_anchor(thread_content)
            
            # Determine thread status
            is_recent = end_idx == len(messages)  # Last thread is active
            status = ThreadStatus.ACTIVE if is_recent else ThreadStatus.COMPLETED
            
            thread = ConversationThread(
                thread_id=f"thread_{thread_id}",
                messages=thread_messages,
                topic=topic,
                status=status,
                created_at=datetime.now() - timedelta(hours=thread_id),  # Approximate
                last_activity=datetime.now() if is_recent else datetime.now() - timedelta(hours=thread_id-1),
                confidence_score=self.calculate_thread_confidence(thread_content),
                semantic_anchor=semantic_anchor
            )
            
            threads.append(thread)
            thread_id += 1
        
        return threads
    
    def generate_thread_topic(self, messages: List[str]) -> str:
        """Generate a descriptive topic for a thread"""
        if not messages:
            return "General Discussion"
        
        # Simple topic extraction based on keywords
        combined_text = " ".join(messages[:3]).lower()  # First 3 messages
        
        # Technical keywords
        tech_keywords = {
            "react": "React Development",
            "python": "Python Programming", 
            "javascript": "JavaScript Development",
            "api": "API Development",
            "database": "Database Discussion",
            "server": "Server Configuration",
            "docker": "Docker/Containerization",
            "aws": "AWS/Cloud Services",
            "git": "Git/Version Control"
        }
        
        # Check for technical topics
        for keyword, topic in tech_keywords.items():
            if keyword in combined_text:
                return topic
        
        # General topics
        if any(word in combined_text for word in ["help", "how", "what", "question"]):
            return "Technical Support"
        elif any(word in combined_text for word in ["error", "bug", "problem", "issue"]):
            return "Troubleshooting"
        elif any(word in combined_text for word in ["learn", "understand", "explain"]):
            return "Learning/Education"
        else:
            return "General Discussion"
    
    def create_semantic_anchor(self, messages: List[str]) -> List[float]:
        """Create a semantic anchor for the thread by averaging embeddings"""
        if not messages:
            return []
        
        embeddings = []
        for msg in messages[:3]:  # Use first 3 messages
            if msg.strip():
                embedding = self.get_message_embedding(msg)
                if embedding:
                    embeddings.append(np.array(embedding))
        
        if not embeddings:
            return []
        
        # Average the embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding.tolist()
    
    def calculate_thread_confidence(self, messages: List[str]) -> float:
        """Calculate confidence score for thread coherence"""
        if len(messages) < 2:
            return 0.5
        
        # Simple heuristic based on message count and content length
        message_count_score = min(len(messages) / 10, 1.0)  # More messages = higher confidence
        
        # Content coherence (simplified)
        total_length = sum(len(msg) for msg in messages)
        avg_length = total_length / len(messages) if messages else 0
        length_score = min(avg_length / 100, 1.0)  # Longer messages = more substantial
        
        return (message_count_score + length_score) / 2

class ThreadLifecycleManager:
    """Manages the lifecycle of conversation threads"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        
        # Configuration
        self.dormant_timeout = timedelta(hours=2)  # Time before thread becomes dormant
        self.archive_timeout = timedelta(days=7)   # Time before thread is archived
        self.max_active_threads = 3               # Maximum concurrent active threads
        
    def get_threads_key(self, conv_id: str) -> str:
        """Get Redis key for conversation threads"""
        return f"threads:{conv_id}"
    
    def load_threads(self, conv_id: str) -> List[ConversationThread]:
        """Load threads from Redis"""
        try:
            data = self.redis_client.get(self.get_threads_key(conv_id))
            if data:
                threads_data = json.loads(data)
                return [ConversationThread.from_dict(thread_data) for thread_data in threads_data]
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
            print(f"✅ Saved {len(threads)} threads for conversation {conv_id}")
        except Exception as e:
            print(f"❌ Error saving threads: {e}")
    
    def update_thread_status(self, threads: List[ConversationThread], new_message_index: int) -> List[ConversationThread]:
        """Update thread statuses based on activity"""
        now = datetime.now()
        updated_threads = []
        
        for thread in threads:
            # Check if this thread is affected by new message
            if new_message_index in thread.messages or new_message_index == max(thread.messages) + 1:
                # Thread is being continued
                thread.status = ThreadStatus.ACTIVE
                thread.last_activity = now
                if new_message_index not in thread.messages:
                    thread.messages.append(new_message_index)
            else:
                # Check for timeout-based status changes
                time_since_activity = now - thread.last_activity
                
                if thread.status == ThreadStatus.ACTIVE:
                    if time_since_activity > self.dormant_timeout:
                        thread.status = ThreadStatus.DORMANT
                        print(f"Thread {thread.thread_id} ({thread.topic}) marked as dormant")
                elif thread.status == ThreadStatus.DORMANT:
                    if time_since_activity > self.archive_timeout:
                        thread.status = ThreadStatus.ARCHIVED
                        print(f"Thread {thread.thread_id} ({thread.topic}) archived")
            
            updated_threads.append(thread)
        
        return updated_threads
    
    def manage_thread_limits(self, threads: List[ConversationThread]) -> List[ConversationThread]:
        """Ensure we don't exceed maximum active threads"""
        active_threads = [t for t in threads if t.status == ThreadStatus.ACTIVE]
        
        if len(active_threads) > self.max_active_threads:
            # Sort by last activity (oldest first)
            active_threads.sort(key=lambda t: t.last_activity)
            
            # Mark oldest threads as dormant
            excess_count = len(active_threads) - self.max_active_threads
            for i in range(excess_count):
                active_threads[i].status = ThreadStatus.DORMANT
                print(f"Thread {active_threads[i].thread_id} marked dormant due to limit")
        
        return threads
    
    def find_reactivation_candidates(self, threads: List[ConversationThread], user_message: str, cohere_client) -> List[Tuple[ConversationThread, float]]:
        """Find dormant threads that might be relevant to reactivate"""
        if not user_message.strip():
            return []
        
        # Get embedding for user message
        try:
            response = cohere_client.embed(
                texts=[user_message],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            user_embedding = response.embeddings[0]
        except Exception:
            return []
        
        candidates = []
        dormant_threads = [t for t in threads if t.status == ThreadStatus.DORMANT]
        
        for thread in dormant_threads:
            if thread.semantic_anchor:
                # Calculate similarity to thread's semantic anchor
                similarity = self.calculate_cosine_similarity(user_embedding, thread.semantic_anchor)
                
                # High similarity suggests thread reactivation
                if similarity > 0.7:  # Threshold for reactivation
                    candidates.append((thread, similarity))
        
        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception:
            return 0.0

class ThreadContextSelector:
    """Selects appropriate thread context for responses"""
    
    def __init__(self, base_conv_manager):
        self.base_manager = base_conv_manager
        
        # Configuration
        self.max_context_threads = 2      # Maximum threads to include in context
        self.recent_messages_priority = 5  # Always include recent messages
        
    def select_thread_context(self, conv_id: str, threads: List[ConversationThread], user_message: str) -> List[Dict]:
        """Select relevant thread context for the current message"""
        
        if not threads:
            # Fallback to base manager
            context, _ = self.base_manager.prepare_context(conv_id, user_message)
            return context
        
        # Get all conversation messages
        all_messages = self.base_manager.get_conversation(conv_id)
        if not all_messages:
            return []
        
        # Build context with thread awareness
        context = []
        
        # System message
        system_content = self.build_system_message_with_threads(threads)
        context.append({"role": "system", "content": system_content})
        
        # Select relevant threads
        relevant_threads = self.select_relevant_threads(threads, user_message)
        
        # Add messages from relevant threads
        included_message_indices = set()
        
        for thread in relevant_threads:
            # Include messages from this thread
            thread_messages = []
            for msg_idx in thread.messages:
                if msg_idx < len(all_messages) and msg_idx not in included_message_indices:
                    thread_messages.append(all_messages[msg_idx])
                    included_message_indices.add(msg_idx)
            
            if thread_messages:
                # Add thread header for context
                thread_header = f"--- {thread.topic} Thread ---"
                context.append({"role": "system", "content": thread_header})
                context.extend(thread_messages)
        
        # Always include recent messages not in threads
        recent_start = max(0, len(all_messages) - self.recent_messages_priority)
        for i in range(recent_start, len(all_messages)):
            if i not in included_message_indices:
                context.append(all_messages[i])
        
        # Add current user message
        context.append({"role": "user", "content": user_message})
        
        return context
    
    def build_system_message_with_threads(self, threads: List[ConversationThread]) -> str:
        """Build enhanced system message with thread awareness"""
        base_message = (
            "You are a helpful assistant that maintains conversation context across multiple topics. "
            "The conversation has been organized into thematic threads. "
            "Provide responses that are contextually appropriate for the current thread."
        )
        
        if threads:
            active_threads = [t for t in threads if t.status == ThreadStatus.ACTIVE]
            if active_threads:
                thread_topics = [t.topic for t in active_threads]
                base_message += f"\n\nActive discussion topics: {', '.join(thread_topics)}"
        
        return base_message
    
    def select_relevant_threads(self, threads: List[ConversationThread], user_message: str) -> List[ConversationThread]:
        """Select the most relevant threads for context"""
        
        # Always include active threads
        active_threads = [t for t in threads if t.status == ThreadStatus.ACTIVE]
        
        # If we have room for more threads, add dormant but relevant ones
        remaining_slots = self.max_context_threads - len(active_threads)
        
        if remaining_slots > 0:
            dormant_threads = [t for t in threads if t.status == ThreadStatus.DORMANT]
            
            # Sort dormant threads by last activity (most recent first)
            dormant_threads.sort(key=lambda t: t.last_activity, reverse=True)
            
            # Add most recent dormant threads
            active_threads.extend(dormant_threads[:remaining_slots])
        
        return active_threads[:self.max_context_threads]

# Integration class that brings everything together
class ThreadAwareConversationManager:
    """Phase 3B implementation: Thread Detection & Topic Lifecycle"""
    
    def __init__(self, base_conv_manager, cohere_client, redis_client):
        self.base_manager = base_conv_manager
        
        # Initialize Phase 3B components
        self.boundary_detector = TopicBoundaryDetector(cohere_client)
        self.lifecycle_manager = ThreadLifecycleManager(redis_client)
        self.context_selector = ThreadContextSelector(base_conv_manager)
        
        # Copy base manager attributes
        for attr in ['max_tokens', 'reserved_tokens', 'count_tokens']:
            if hasattr(base_conv_manager, attr):
                setattr(self, attr, getattr(base_conv_manager, attr))
    
    def __getattr__(self, name):
        """Delegate unknown attributes to base manager"""
        return getattr(self.base_manager, name)
    
    def analyze_conversation_threads(self, conv_id: str) -> List[ConversationThread]:
        """Analyze conversation and create/update threads"""
        
        # Get all messages
        messages = self.base_manager.get_conversation(conv_id)
        if not messages:
            return []
        
        print(f"🧵 Analyzing {len(messages)} messages for thread detection")
        
        # Detect topic boundaries
        boundaries = self.boundary_detector.detect_topic_boundaries(messages)
        print(f"🎯 Detected {len(boundaries)} topic boundaries: {boundaries}")
        
        # Create threads from boundaries
        new_threads = self.boundary_detector.create_threads_from_boundaries(messages, boundaries)
        print(f"✅ Created {len(new_threads)} conversation threads")
        
        for thread in new_threads:
            print(f"  - {thread.thread_id}: {thread.topic} ({len(thread.messages)} messages)")
        
        return new_threads
    
    def update_threads_with_new_message(self, conv_id: str, message_index: int) -> List[ConversationThread]:
        """Update existing threads when a new message is added"""
        
        # Load existing threads
        existing_threads = self.lifecycle_manager.load_threads(conv_id)
        
        if not existing_threads:
            # No existing threads, analyze from scratch
            return self.analyze_conversation_threads(conv_id)
        
        # Update thread statuses with new message
        updated_threads = self.lifecycle_manager.update_thread_status(existing_threads, message_index)
        
        # Manage thread limits
        updated_threads = self.lifecycle_manager.manage_thread_limits(updated_threads)
        
        # Save updated threads
        self.lifecycle_manager.save_threads(conv_id, updated_threads)
        
        return updated_threads
    
    def prepare_context_with_threads(self, conv_id: str, user_message: str) -> Tuple[List[Dict], int]:
        """Prepare context using thread-aware selection"""
        
        # Load or analyze threads
        threads = self.lifecycle_manager.load_threads(conv_id)
        
        if not threads:
            # First time analysis
            threads = self.analyze_conversation_threads(conv_id)
            self.lifecycle_manager.save_threads(conv_id, threads)
        
        # Check for thread reactivation opportunities
        reactivation_candidates = self.lifecycle_manager.find_reactivation_candidates(
            threads, user_message, self.boundary_detector.cohere_client
        )
        
        # Reactivate highly relevant dormant threads
        for thread, similarity in reactivation_candidates[:1]:  # Reactivate top 1 candidate
            thread.status = ThreadStatus.ACTIVE
            thread.last_activity = datetime.now()
            print(f"🔄 Reactivated thread: {thread.topic} (similarity: {similarity:.3f})")
        
        # Select context using thread awareness
        context = self.context_selector.select_thread_context(conv_id, threads, user_message)
        
        # Calculate token usage
        total_tokens = sum(self.base_manager.count_tokens(msg.get("content", "")) for msg in context)
        
        print(f"🧵 Thread-aware context: {len(context)} messages from {len(threads)} threads, {total_tokens} tokens")
        
        return context, total_tokens
    
    def add_exchange_with_threads(self, conv_id: str, user_message: str, assistant_message: str):
        """Add exchange and update thread tracking"""
        
        # Get current message count to determine new indices
        current_messages = self.base_manager.get_conversation(conv_id)
        user_message_index = len(current_messages)
        assistant_message_index = len(current_messages) + 1
        
        # Add exchange using base manager
        self.base_manager.add_exchange(conv_id, user_message, assistant_message)
        
        # Update threads with new messages
        self.update_threads_with_new_message(conv_id, user_message_index)
        self.update_threads_with_new_message(conv_id, assistant_message_index)
        
        print(f"✅ Added exchange and updated threads for conversation {conv_id}")
        
        return True
