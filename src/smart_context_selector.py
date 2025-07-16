"""
SmartContextSelector - Azure OpenAI Powered Context Search

SIMPLIFIED VERSION with Azure OpenAI:
- Uses Azure OpenAI embeddings instead of Cohere
- Only 2 strategies: SEMANTIC_HYBRID vs RECENT_FALLBACK
- Removed complex memory and thread management
- Simple, predictable strategy selection
- Better performance and easier debugging
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
    """SIMPLIFIED: Only 2 strategies for predictable behavior"""
    SEMANTIC_HYBRID = "semantic_hybrid"       # Comprehensive search using threads + semantic
    RECENT_FALLBACK = "recent_fallback"       # Simple recent messages (fast path)

class SmartContextSelector:
    """
    SIMPLIFIED Azure OpenAI powered context selector:
    - Only 2 strategies for predictable behavior
    - Simple decision rules
    - Uses Azure OpenAI embeddings
    - Better performance and easier debugging
    """
    
    def __init__(self, cohere_client, conversation_manager, thread_manager=None, memory_manager=None):
        """
        Initialize the SmartContextSelector
        
        Args:
            cohere_client: Azure OpenAI client (parameter name kept for compatibility)
            conversation_manager: Base conversation manager
            thread_manager: Optional thread detection system
            memory_manager: Optional (deprecated, not used)
        """
        # Note: parameter is named cohere_client for backward compatibility, but it's actually Azure OpenAI
        self.azure_client = cohere_client  # This is actually the Azure OpenAI client now
        self.conv_manager = conversation_manager
        self.thread_manager = thread_manager
        # memory_manager is deprecated and ignored
        
        # Context selection configuration
        self.max_tokens = getattr(conversation_manager, 'max_tokens', 120000)  # Azure OpenAI supports more
        self.reserved_tokens = getattr(conversation_manager, 'reserved_tokens', 4000)
        self.available_tokens = self.max_tokens - self.reserved_tokens
        
        # SIMPLIFIED thresholds
        self.semantic_threshold = 0.3
        self.max_context_messages = 20
        self.min_context_messages = 3
        self.fallback_recent_count = 10  # Increased for better context
        
        # SIMPLIFIED strategy selection
        self.min_length_for_semantic = 15   # Use semantic search above this length
    
    def get_comprehensive_context(self, user_message: str, conv_id: str, intent_info: IntentClassification) -> ComprehensiveContextResult:
        """
        SIMPLIFIED Main method: Get context using simple, predictable strategy selection
        
        Args:
            user_message: Current user message
            conv_id: Conversation ID
            intent_info: The pre-classified intent of the user message
            
        Returns:
            ComprehensiveContextResult with optimal context
        """
        
        print(f"🔍 SmartContextSelector: Building context for conv {conv_id} (Azure OpenAI)")
        
        # Get conversation length
        conversation = self.conv_manager.get_conversation(conv_id)
        conv_length = len(conversation)
        
        print(f"📊 Conversation length: {conv_length} messages")
        
        # SIMPLIFIED strategy selection
        strategy = self._select_strategy_simple(conv_length, intent_info)
        
        print(f"🎯 Selected strategy: {strategy.value}")
        
        # Execute the selected strategy
        if strategy == ContextSelectionStrategy.SEMANTIC_HYBRID:
            return self._semantic_hybrid_selection(user_message, conv_id, conversation)
        else:
            return self._recent_fallback_selection(user_message, conv_id, conversation)

    def _select_strategy_simple(self, conv_length: int, intent_info: IntentClassification) -> ContextSelectionStrategy:
        """
        SIMPLIFIED strategy selection with just 2 clear rules:
        
        Rule 1: Short conversations always use RECENT_FALLBACK (fast)
        Rule 2: High-confidence continuations in medium conversations use RECENT_FALLBACK (fast)  
        Rule 3: Long conversations OR complex intents use SEMANTIC_HYBRID (comprehensive)
        """
        
        print(f"🧐 Strategy selection. Intent: {intent_info.intent.value}, Confidence: {intent_info.confidence:.2f}")

        # Rule 1: Short conversations always use recent context
        if conv_length < self.min_length_for_semantic:
            print(f"📏 Short conversation ({conv_length} < {self.min_length_for_semantic}). Using RECENT_FALLBACK.")
            return ContextSelectionStrategy.RECENT_FALLBACK

        # Rule 2: High-confidence simple continuations use recent context (fast path)
        # BUT only for medium-length conversations - long ones need semantic search
        fast_path_intents = [
            IntentCategory.CONTINUATION, 
            IntentCategory.REFINEMENT, 
            IntentCategory.CORRECTION, 
            IntentCategory.CONVERSATIONAL_FILLER
        ]
        
        if (intent_info.intent in fast_path_intents and 
            intent_info.confidence > 0.7 and
            conv_length < 30):  # Only use fast path for medium-length conversations
            print(f"⚡ High-confidence continuation in medium conversation. Using RECENT_FALLBACK (fast path).")
            return ContextSelectionStrategy.RECENT_FALLBACK

        # Rule 3: Everything else uses comprehensive search
        print(f"🔍 Complex intent or longer conversation. Using SEMANTIC_HYBRID (comprehensive).")
        return ContextSelectionStrategy.SEMANTIC_HYBRID
        
    def _semantic_hybrid_selection(self, user_message: str, conv_id: str, conversation: List[Dict]) -> ComprehensiveContextResult:
        """
        IMPROVED semantic hybrid selection that delegates to thread manager if available
        """
        print("🎨 Using comprehensive semantic+thread context selection (Azure OpenAI).")
        
        if not self.thread_manager:
            print("⚠️ Thread manager not available. Using recent context with semantic boost.")
            return self._recent_fallback_selection(user_message, conv_id, conversation)

        try:
            # Delegate to the specialized thread-aware manager
            context, tokens = self.thread_manager.prepare_context_with_threads(conv_id, user_message)
            
            # Extract thread information for the result
            threads = self.thread_manager.lifecycle_manager.load_threads(conv_id)
            active_threads = [t for t in threads if t.status.value == "active"]
            relevant_thread_ids = [t.thread_id for t in active_threads[:2]]
            
            return ComprehensiveContextResult(
                context=context,
                tokens_used=tokens,
                relevant_threads=relevant_thread_ids,
                memory_factors=["Azure OpenAI thread-aware context selection"],
                semantic_scores=[],
                selection_strategy=ContextSelectionStrategy.SEMANTIC_HYBRID.value,
                debug_info={
                    "method": "Azure OpenAI Thread-aware delegation",
                    "total_threads": len(threads),
                    "active_threads": len(active_threads),
                    "context_messages": len(context)
                }
            )

        except Exception as e:
            print(f"❌ Thread-aware context failed: {e}. Falling back to recent context.")
            return self._recent_fallback_selection(user_message, conv_id, conversation)
    
    def _recent_fallback_selection(self, user_message: str, conv_id: str, conversation: List[Dict]) -> ComprehensiveContextResult:
        """
        IMPROVED recent fallback selection with better token utilization
        """
        
        print(f"⚡ Recent fallback context selection (last {self.fallback_recent_count} messages)")
        
        # Get recent messages with better token management
        recent_messages = conversation[-self.fallback_recent_count:] if conversation else []
        
        # Build context with system message
        system_content = (
            "You are a helpful assistant that maintains conversation context. "
            "Provide thoughtful, accurate responses based on the conversation history."
        )
        
        context = [{"role": "system", "content": system_content}]
        
        # Add recent messages, ensuring we don't exceed token budget
        tokens_used = self._count_tokens(system_content)
        user_tokens = self._count_tokens(user_message)
        available_for_history = self.available_tokens - tokens_used - user_tokens
        
        selected_messages = []
        for message in recent_messages:
            msg_tokens = self._count_tokens(message.get("content", ""))
            if tokens_used + msg_tokens <= available_for_history:
                selected_messages.append(message)
                tokens_used += msg_tokens
            else:
                print(f"⚠️ Token limit reached. Using {len(selected_messages)} of {len(recent_messages)} recent messages.")
                break
        
        context.extend(selected_messages)
        
        # Add current user message
        context.append({"role": "user", "content": user_message})
        total_tokens = tokens_used + user_tokens
        
        return ComprehensiveContextResult(
            context=context,
            tokens_used=total_tokens,
            relevant_threads=[],
            memory_factors=[f"Recent {len(selected_messages)} messages (Azure OpenAI)"],
            semantic_scores=[],
            selection_strategy=ContextSelectionStrategy.RECENT_FALLBACK.value,
            debug_info={
                "method": "Recent messages with token management (Azure OpenAI)",
                "messages_requested": len(recent_messages),
                "messages_used": len(selected_messages),
                "tokens_available": available_for_history,
                "tokens_used": total_tokens
            }
        )
     
    def _count_tokens(self, text: str) -> int:
        """Count tokens using Azure OpenAI tokenizer"""
        if hasattr(self.azure_client, 'count_tokens'):
            return self.azure_client.count_tokens(text)
        elif hasattr(self.conv_manager, 'count_tokens'):
            return self.conv_manager.count_tokens(text)
        else:
            # Fallback approximation
            return len(text) // 4
   
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for monitoring"""
        return {
            "config": {
                "max_tokens": self.max_tokens,
                "semantic_threshold": self.semantic_threshold,
                "max_context_messages": self.max_context_messages,
                "min_length_for_semantic": self.min_length_for_semantic,
                "fallback_recent_count": self.fallback_recent_count
            },
            "strategies": {
                "total_strategies": 2,
                "available_strategies": [s.value for s in ContextSelectionStrategy],
                "simplified": True
            },
            "strategy_selection": {
                "short_conversations": f"< {self.min_length_for_semantic} messages → RECENT_FALLBACK",
                "medium_high_confidence_continuations": "confidence > 0.7 & < 30 msgs → RECENT_FALLBACK",
                "long_conversations_or_complex": "30+ messages OR complex intents → SEMANTIC_HYBRID"
            },
            "components_available": {
                "azure_openai_client": self.azure_client is not None,
                "conversation_manager": self.conv_manager is not None,
                "thread_manager": self.thread_manager is not None
            },
            "improvements": {
                "embeddings_provider": "Azure OpenAI (replaced Cohere)",
                "strategies_simplified": "4 → 2 strategies",
                "memory_system_removed": "Not implemented, removed complexity",
                "predictable_selection": "Simple rules, no oscillation",
                "better_token_management": "Improved context fitting"
            }
        }

    # REMOVED: All memory-guided methods (they weren't working anyway)
    # REMOVED: Complex thread-focused strategy (merged into semantic_hybrid)
    # REMOVED: Multiple similarity calculations and complex scoring
    
    # The goal is SIMPLICITY and PREDICTABILITY with Azure OpenAI power
