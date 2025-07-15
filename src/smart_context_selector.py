"""
SmartContextSelector - Single-Pass Comprehensive Context Search

This is the core optimization that reduces API calls by 50% while improving context quality.
Instead of separate context searches for intent classification and response generation,
this does ONE comprehensive search and reuses the context for both purposes.

Key Innovation:
- Memory-guided topic filtering  
- Thread-aware context selection
- Semantic search within filtered results
- Single context reused for intent classification AND response generation
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from intent_classification_system import IntentClassification, IntentCategory

@dataclass
class ComprehensiveContextResult:
    """Result of comprehensive context search"""
    context: List[Dict]           # The actual context messages
    tokens_used: int             # Estimated token count
    relevant_threads: List[str]   # Thread IDs that influenced selection
    memory_factors: List[str]     # Memory factors that influenced selection
    semantic_scores: List[float]  # Semantic similarity scores for selected messages
    selection_strategy: str       # Strategy used for selection
    debug_info: Dict             # Detailed debug information

class ContextSelectionStrategy(Enum):
    """Different strategies for context selection"""
    MEMORY_GUIDED = "memory_guided"           # Use memory to guide topic filtering
    THREAD_FOCUSED = "thread_focused"         # Focus on active thread context
    SEMANTIC_HYBRID = "semantic_hybrid"       # Hybrid semantic + thread + memory
    RECENT_FALLBACK = "recent_fallback"       # Simple recent messages fallback

class SmartContextSelector:
    """
    Single-pass comprehensive context selector that integrates:
    1. Memory management (long-term patterns)
    2. Thread detection (topic clustering) 
    3. Semantic search (relevance-based selection)
    
    This context is then reused for both intent classification AND response generation,
    eliminating the need for separate context building operations.
    """
    
    def __init__(self, cohere_client, conversation_manager, thread_manager=None, memory_manager=None):
        """
        Initialize the SmartContextSelector
        
        Args:
            cohere_client: Cohere client for semantic embeddings
            conversation_manager: Base conversation manager
            thread_manager: Optional thread detection system
            memory_manager: Optional conversation memory system
        """
        self.cohere_client = cohere_client
        self.conv_manager = conversation_manager
        self.thread_manager = thread_manager
        self.memory_manager = memory_manager
        
        # Context selection configuration
        self.max_tokens = getattr(conversation_manager, 'max_tokens', 60000)
        self.reserved_tokens = getattr(conversation_manager, 'reserved_tokens', 4000)
        self.intent_reserved_tokens = 1000  # Reserve tokens for intent classification
        
        # Selection thresholds
        self.semantic_threshold = 0.3        # Minimum semantic similarity
        self.thread_relevance_threshold = 0.5  # Minimum thread relevance
        self.memory_importance_threshold = 0.4  # Minimum memory importance
        
        # Context limits
        self.max_context_messages = 20       # Maximum messages to include
        self.min_context_messages = 3        # Minimum messages to include
        self.fallback_recent_count = 8       # Recent messages for fallback
        
        # Strategy selection thresholds
        self.conversation_length_for_semantic = 10   # Use semantic above this length
        self.conversation_length_for_memory = 20     # Use memory above this length
    
    def get_comprehensive_context(self, user_message: str, conv_id: str, intent_info: IntentClassification) -> ComprehensiveContextResult:
   
        """
        Main method: Get comprehensive context for both intent classification AND response generation
        
        This is the core optimization - ONE context search serving both purposes.
        
        Args:
            user_message: Current user message
            conv_id: Conversation ID
            intent_info: The pre-classified intent of the user message.  # <--- ADD this comment
            
        Returns:
            ComprehensiveContextResult with context suitable for both intent and response
        """
        
        print(f"🔍 SmartContextSelector: Building comprehensive context for conv {conv_id}")
        
        # Get conversation length to determine strategy
        conversation = self.conv_manager.get_conversation(conv_id)
        conv_length = len(conversation)
        
        print(f"📊 Conversation length: {conv_length} messages")
        
        # Select optimal strategy based on conversation characteristics AND pre-classified intent
        strategy = self._select_optimal_strategy(conv_length, user_message, conv_id, intent_info)
        
        print(f"🎯 Selected strategy: {strategy.value}")
        
        # Execute the selected strategy
        if strategy == ContextSelectionStrategy.MEMORY_GUIDED:
            return self._memory_guided_selection(user_message, conv_id, conversation)
        elif strategy == ContextSelectionStrategy.SEMANTIC_HYBRID:
            return self._semantic_hybrid_selection(user_message, conv_id, conversation)
        else:
            return self._recent_fallback_selection(user_message, conv_id, conversation)

    def _select_optimal_strategy(self, conv_length: int, user_message: str, conv_id: str, intent_info: IntentClassification) -> ContextSelectionStrategy:
        """
        Intelligently select the best context selection strategy based on
        the pre-classified intent of the user's message.
        """
        print(f"🧐 Intent-driven strategy. Primary Intent: {intent_info.intent.value}, Confidence: {intent_info.confidence:.2f}")

        primary_intent = intent_info.intent

        # Rule 1: High-confidence, simple continuations are always the fast path.
        if primary_intent in [IntentCategory.CONTINUATION, IntentCategory.REFINEMENT, IntentCategory.CORRECTION, IntentCategory.CONVERSATIONAL_FILLER] and intent_info.confidence > 0.7:
            print("🔗 Intent is a direct follow-up. Using efficient RECENT_FALLBACK.")
            return ContextSelectionStrategy.RECENT_FALLBACK

        # Rule 2: Intents that require broad knowledge MUST use a deep search.
        # If a thread manager is available, it's ALWAYS the best tool for this.
        if primary_intent in [IntentCategory.NEW_REQUEST, IntentCategory.DEBUGGING, IntentCategory.EXPLANATION, IntentCategory.COMPARISON, IntentCategory.STATUS_CHECK, IntentCategory.META_INSTRUCTION]:
            if self.thread_manager:
                print("🌀 Intent requires broad context. Delegating to SEMANTIC_HYBRID (Thread-Aware).")
                return ContextSelectionStrategy.SEMANTIC_HYBRID
            # If no thread manager, we can't do the hybrid, so fall back to simple recent context.
            # In a future step, you could add a pure "semantic search" strategy here.
            else:
                print("⚠️ No thread manager for deep search. Using RECENT_FALLBACK.")
                return ContextSelectionStrategy.RECENT_FALLBACK

        # Rule 3: If confidence is low, play it safe with the best available tool.
        if intent_info.confidence < 0.6:
            print(f"⚠️ Low confidence intent ({intent_info.confidence:.2f}). Performing secondary relevance check...")
            # Use our fallback function as a tie-breaker.
            context_relevance = self._check_recent_context_relevance(user_message, conv_id)

            if context_relevance > 0.65:
                # The intent is unclear, treat it as a poorly-phrased follow-up.
                print("   -> High context relevance. Treating as a follow-up. Using RECENT_FALLBACK.")
                return ContextSelectionStrategy.RECENT_FALLBACK
            else:
                # The intent is unclear AND it's a topic shift. This is high risk.
                # Use the most powerful search to find any possible context.
                print("   -> Low context relevance. This is an ambiguous topic shift. Using SEMANTIC_HYBRID to be safe.")
                return ContextSelectionStrategy.SEMANTIC_HYBRID

        # A final default case for anything that slips through.
        print("⚡ No strong signal. Defaulting to RECENT_FALLBACK.")
        return ContextSelectionStrategy.RECENT_FALLBACK
        
    def _check_recent_context_relevance(self, user_message: str, conv_id: str) -> float:
        """
        Check if recent context (last X messages) is semantically relevant to user's question
        
        Args:
            user_message: User's current question
            conv_id: Conversation ID
            
        Returns:
            Float 0.0-1.0: How relevant recent context is to the question
            - High score (>0.6): Recent context makes sense → use recent context (fast)
            - Low score (<0.6): Recent context doesn't help → use semantic search
        """
        
        try:
            # Get recent messages (last 4-6 messages)
            conversation = self.conv_manager.get_conversation(conv_id)
            if not conversation or len(conversation) < 2:
                return 0.5  # Neutral - not enough context to judge
            
            # Get last 4 messages as recent context
            recent_messages = conversation[-4:]
            recent_text = " ".join([msg.get("content", "") for msg in recent_messages])
            
            # Get embeddings for user question and recent context
            user_embedding = self._get_message_embedding(user_message)
            recent_embedding = self._get_message_embedding(recent_text)
            
            if not user_embedding or not recent_embedding:
                return 0.5  # Can't compute similarity, assume neutral
            
            # Calculate semantic similarity
            relevance_score = self._calculate_cosine_similarity(user_embedding, recent_embedding)
            
            print(f"🔍 Context relevance check: '{user_message[:30]}...' vs recent context = {relevance_score:.3f}")
            
            return relevance_score
            
        except Exception as e:
            print(f"⚠️ Error checking context relevance: {e}")
            return 0.5  # Neutral on error
          
    def _semantic_hybrid_selection(self, user_message: str, conv_id: str, conversation: List[Dict]) -> ComprehensiveContextResult:
        """
        Builds the most sophisticated context by delegating to the ThreadAwareConversationManager.
        This method handles thread analysis, reactivation, and context assembly.
        """
        print("🎨 Delegating context creation to Thread-Aware Manager for hybrid selection.")
        
        if not self.thread_manager:
            print("⚠️ Thread manager not available. Falling back to recent context.")
            return self._recent_fallback_selection(user_message, conv_id, conversation)

        try:
            # DELEGATE the entire context preparation to the specialized manager.
            # This single call will handle:
            #  - Loading/analyzing threads
            #  - Reactivating dormant threads if relevant
            #  - Building a context with thread headers
            context, tokens = self.thread_manager.prepare_context_with_threads(conv_id, user_message)
            
            # We can extract details for our result object after the fact
            threads = self.thread_manager.lifecycle_manager.load_threads(conv_id)
            active_threads = [t for t in threads if t.status.value == "active"]
            relevant_thread_ids = [t.thread_id for t in active_threads[:2]]
            
            return ComprehensiveContextResult(
                context=context,
                tokens_used=tokens,
                relevant_threads=relevant_thread_ids,
                memory_factors=[], # Memory can be integrated here later
                semantic_scores=[], # This method doesn't produce scores, which is fine
                selection_strategy=ContextSelectionStrategy.SEMANTIC_HYBRID.value,
                debug_info={
                    "delegation_target": "ThreadAwareConversationManager.prepare_context_with_threads",
                    "total_threads_found": len(threads),
                    "active_threads": len(active_threads)
                }
            )

        except Exception as e:
            print(f"❌ Thread-aware context selection failed: {e}. Falling back.")
            return self._recent_fallback_selection(user_message, conv_id, conversation)
    
    def _recent_fallback_selection(self, user_message: str, conv_id: str, conversation: List[Dict]) -> ComprehensiveContextResult:
        """
        Simple fallback selection using recent messages
        Used for short conversations or when other systems fail
        """
        
        print("⚡ Recent fallback context selection")
        
        # Simple recent message selection
        recent_messages = conversation[-self.fallback_recent_count:] if conversation else []
        
        # Build basic context with system message
        system_content = (
            "You are a helpful assistant that maintains conversation context. "
            "Provide thoughtful, accurate responses based on the conversation history."
        )
        
        context = [{"role": "system", "content": system_content}]
        
        if recent_messages:
            context.extend(recent_messages)
        
        # Add current user message
        context.append({"role": "user", "content": user_message})
        
        # Calculate tokens
        tokens = sum(self._count_tokens(msg.get("content", "")) for msg in context)
        
        return ComprehensiveContextResult(
            context=context,
            tokens_used=tokens,
            relevant_threads=[],
            memory_factors=["Fallback: No memory/thread filtering"],
            semantic_scores=[],
            selection_strategy=ContextSelectionStrategy.RECENT_FALLBACK.value,
            debug_info={
                "recent_messages_used": len(recent_messages),
                "original_conversation_length": len(conversation),
                "fallback_reason": "Short conversation or system failure"
            }
        )
     
    def _get_message_embedding(self, message_content: str) -> List[float]:
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
    
    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
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
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using the conversation manager's method"""
        if hasattr(self.conv_manager, 'count_tokens'):
            return self.conv_manager.count_tokens(text)
        else:
            # Fallback approximation
            return len(text) // 4
   
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for monitoring and optimization"""
        return {
            "config": {
                "max_tokens": self.max_tokens,
                "semantic_threshold": self.semantic_threshold,
                "max_context_messages": self.max_context_messages,
                "conversation_length_for_semantic": self.conversation_length_for_semantic,
                "conversation_length_for_memory": self.conversation_length_for_memory
            },
            "strategy_thresholds": {
                "recent_fallback": f"< {self.conversation_length_for_semantic} messages",
                "thread_focused": f"{self.conversation_length_for_semantic}-{self.conversation_length_for_memory} messages",
                "memory_guided": f"> {self.conversation_length_for_memory} messages",
                "semantic_hybrid": "when both thread and memory available"
            },
            "components_available": {
                "cohere_client": self.cohere_client is not None,
                "conversation_manager": self.conv_manager is not None,
                "thread_manager": self.thread_manager is not None,
                "memory_manager": self.memory_manager is not None
            }
        }

    #region Memory - NOT used - TODO - To evaluate if needed.
     
    def _memory_guided_selection(self, user_message: str, conv_id: str, conversation: List[Dict]) -> ComprehensiveContextResult:
        """
        Memory-guided context selection for long conversations
        Uses conversation memory to guide topic filtering, then applies semantic search
        """
        
        print("🧠 Memory-guided context selection")
        
        # Get conversation memory to guide selection
        memory_factors = []
        relevant_topics = []
        
        if self.memory_manager:
            try:
                memory = self.memory_manager.get_conversation_memory(conv_id)
                relevant_topics = list(memory.get('topics_discussed', {}).keys())
                memory_factors = [f"Topics: {', '.join(relevant_topics[:3])}"]
                print(f"💭 Memory topics: {relevant_topics}")
            except Exception as e:
                print(f"⚠️ Memory access failed: {e}")
                memory_factors = ["Memory unavailable"]
        
        # Filter conversation based on memory-guided topic relevance
        if relevant_topics:
            filtered_messages = self._filter_by_memory_topics(conversation, relevant_topics, user_message)
        else:
            # Fallback to recent messages if memory unavailable
            filtered_messages = conversation[-self.fallback_recent_count:]
        
        # Apply semantic search within memory-filtered results
        context_result = self._apply_semantic_search(
            filtered_messages, user_message, max_messages=self.max_context_messages
        )
        
        return ComprehensiveContextResult(
            context=context_result['context'],
            tokens_used=context_result['tokens'],
            relevant_threads=[],
            memory_factors=memory_factors,
            semantic_scores=context_result['scores'],
            selection_strategy=ContextSelectionStrategy.MEMORY_GUIDED.value,
            debug_info={
                "filtered_messages": len(filtered_messages),
                "original_conversation_length": len(conversation),
                "memory_topics": relevant_topics,
                "semantic_threshold": self.semantic_threshold
            }
        )
    
    def _apply_semantic_search(self, messages: List[Dict], user_message: str, max_messages: int) -> Dict:
        """
        Apply semantic search to filter and rank messages by relevance
        
        Args:
            messages: List of messages to search through
            user_message: Current user message for similarity comparison
            max_messages: Maximum messages to return
            
        Returns:
            Dict with 'context', 'tokens', and 'scores'
        """
        
        print(f"🔎 Applying semantic search to {len(messages)} messages")
        
        if not messages:
            return self._build_empty_context(user_message)
        
        try:
            # Get user message embedding
            user_embedding = self._get_message_embedding(user_message)
            if not user_embedding:
                print("⚠️ Failed to get user message embedding, using chronological order")
                return self._build_chronological_context(messages, user_message, max_messages)
            
            # Calculate similarities for all messages
            message_scores = []
            
            for i, message in enumerate(messages):
                content = message.get('content', '')
                if not content:
                    continue
                    
                # Get or calculate message embedding
                msg_embedding = self._get_message_embedding(content)
                if msg_embedding:
                    similarity = self._calculate_cosine_similarity(user_embedding, msg_embedding)
                    message_scores.append({
                        'message': message,
                        'similarity': similarity,
                        'index': i
                    })
            
            # Sort by similarity (highest first)
            message_scores.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Filter by threshold and select top messages
            relevant_messages = [
                msg_data for msg_data in message_scores 
                if msg_data['similarity'] >= self.semantic_threshold
            ][:max_messages]
            
            if not relevant_messages:
                print("⚠️ No messages meet semantic threshold, using recent messages")
                return self._build_chronological_context(messages, user_message, max_messages)
            
            # Sort selected messages chronologically to preserve conversation flow
            relevant_messages.sort(key=lambda x: x['index'])
            
            # Build context
            selected_messages = [msg_data['message'] for msg_data in relevant_messages]
            scores = [msg_data['similarity'] for msg_data in relevant_messages]
            
            context = self._build_context_with_system(selected_messages, user_message)
            tokens = sum(self._count_tokens(msg.get("content", "")) for msg in context)
            
            print(f"✅ Semantic search: {len(selected_messages)} messages selected, avg similarity: {sum(scores)/len(scores):.3f}")
            
            return {
                'context': context,
                'tokens': tokens,
                'scores': scores
            }
            
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            return self._build_chronological_context(messages, user_message, max_messages)
   
    def _filter_by_memory_topics(self, messages: List[Dict], topics: List[str], user_message: str) -> List[Dict]:
        """
        Filter messages based on memory topics relevance
        """
        
        print(f"🎯 Filtering by memory topics: {topics[:3]}")
        
        # Simple keyword-based filtering for now
        # In production, this could use more sophisticated topic modeling
        filtered_messages = []
        topic_keywords = set()
        
        # Extract keywords from topics
        for topic in topics:
            topic_keywords.update(topic.lower().split())
        
        for message in messages:
            content = message.get('content', '').lower()
            
            # Check if message content relates to any memory topics
            if any(keyword in content for keyword in topic_keywords):
                filtered_messages.append(message)
            # Always include recent messages regardless of topic match
            elif messages.index(message) >= len(messages) - 5:
                filtered_messages.append(message)
        
        # Ensure we have minimum messages
        if len(filtered_messages) < self.min_context_messages:
            filtered_messages = messages[-self.fallback_recent_count:]
        
        print(f"📝 Memory filtering result: {len(messages)} → {len(filtered_messages)} messages")
        return filtered_messages
    
    def _build_context_with_system(self, messages: List[Dict], user_message: str) -> List[Dict]:
        """Build complete context with system message"""
        system_content = (
            "You are a helpful assistant that maintains conversation context. "
            "Provide thoughtful, accurate responses based on the conversation history."
        )
        
        context = [{"role": "system", "content": system_content}]
        context.extend(messages)
        context.append({"role": "user", "content": user_message})
        
        return context
    
    def _build_chronological_context(self, messages: List[Dict], user_message: str, max_messages: int) -> Dict:
        """Build context using chronological order as fallback"""
        
        recent_messages = messages[-max_messages:] if messages else []
        context = self._build_context_with_system(recent_messages, user_message)
        tokens = sum(self._count_tokens(msg.get("content", "")) for msg in context)
        
        return {
            'context': context,
            'tokens': tokens,
            'scores': []
        }
    
    def _build_empty_context(self, user_message: str) -> Dict:
        """Build minimal context when no messages available"""
        
        context = self._build_context_with_system([], user_message)
        tokens = sum(self._count_tokens(msg.get("content", "")) for msg in context)
        
        return {
            'context': context,
            'tokens': tokens,
            'scores': []
        }
      
    
    #endregion Memory guided context selection - TODO - To evaulate if needed.
   