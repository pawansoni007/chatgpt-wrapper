"""
Phase 2: MultiLayeredContextBuilder

Intent-aware context building system with 3-level depth strategy.
Bridges intent classification with thread-aware context selection.
"""

import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from intent_classification_system import IntentCategory, IntentClassification


@dataclass
class ContextBuildResult:
    """Result of context building process"""
    context: List[Dict]  # The actual context messages
    tokens_used: int     # Estimated token count
    level_used: int      # Which context level was used (1, 2, or 3)
    strategy: str        # Strategy name that was applied
    debug_info: Dict     # Additional debug information


class MultiLayeredContextBuilder:
    """

    Level 1 (Fast): High confidence intents (≥0.9) - minimal processing
    Level 2 (Semantic): Medium confidence intents (0.7-0.9) - semantic search + threads
    Level 3 (Clarification): Low confidence intents (<0.7) - ask for clarification
    """
    
    def __init__(self, base_conv_manager, thread_aware_manager, memory_manager=None):
        """
        Initialize the multilayered context builder
        
        Args:
            base_conv_manager: Original ConversationManager for basic operations
            thread_aware_manager: ThreadAwareConversationManager for thread operations
            memory_manager: Optional memory manager for enhanced context
        """
        self.base_manager = base_conv_manager
        self.thread_manager = thread_aware_manager
        self.memory_manager = memory_manager
        
        # Context level configuration
        self.level_1_confidence = 0.9   # High confidence - fast context
        self.level_2_confidence = 0.7   # Medium confidence - semantic context
        self.level_3_confidence = 0.5   # Low confidence - clarification needed
        
        # Token management
        self.max_tokens = getattr(base_conv_manager, 'max_tokens', 60000)
        self.reserved_tokens = getattr(base_conv_manager, 'reserved_tokens', 4000)
        self.count_tokens = getattr(base_conv_manager, 'count_tokens', self._fallback_token_count)
        
        # Context depth settings
        self.level_1_depth = 5      # Recent messages for fast context - TODO - Probably I might decrease this.
        self.level_2_max_context = 15  # Max context messages for semantic
        self.level_3_depth = 3      # Minimal context for clarification
        
        # Intent strategy mapping (will be expanded step by step)
        self.intent_strategies = self._initialize_intent_strategies()
    
    def _fallback_token_count(self, text: str) -> int:
        """Fallback token counting if base manager doesn't have it"""
        return len(text) // 4  # Rough approximation
    
    def _initialize_intent_strategies(self) -> Dict[IntentCategory, str]:
        """Initialize intent-to-strategy mapping (starting simple)"""
        return {
            # Phase 2.1: Core strategies (implemented first)
            IntentCategory.CONTINUATION: "continuation",
            IntentCategory.DEBUGGING: "debugging", 
            IntentCategory.NEW_REQUEST: "new_request",
            
            # Phase 2.2: Knowledge strategies (next batch)
            IntentCategory.EXPLANATION: "explanation",
            IntentCategory.STATUS_CHECK: "status_check",
            
            # TODO - Phase 2.3: Remaining strategies (fallback to generic for now)
            IntentCategory.REFINEMENT: "generic",
            IntentCategory.ARTIFACT_GENERATION: "generic",
            IntentCategory.COMPARISON: "generic",
            IntentCategory.CORRECTION: "generic",
            IntentCategory.META_INSTRUCTION: "generic",
            IntentCategory.CONVERSATIONAL_FILLER: "minimal",
        }
    
    def build_context(self, intent_result: IntentClassification, conv_id: str, user_message: str) -> ContextBuildResult:
        """
        Main context building orchestrator
        
        Args:
            intent_result: Result from intent classification
            conv_id: Conversation ID
            user_message: Current user message
            
        Returns:
            ContextBuildResult with context, tokens, and metadata
        """
        
        print(f"🔧 Building context for intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})")
        
        # Determine context level based on confidence
        if intent_result.confidence >= self.level_1_confidence:
            level = 1
            context, tokens = self._build_level_1_context(intent_result, conv_id, user_message)
        elif intent_result.confidence >= self.level_2_confidence:
            level = 2
            context, tokens = self._build_level_2_context(intent_result, conv_id, user_message)
        else:
            level = 3
            context, tokens = self._build_level_3_context(intent_result, conv_id, user_message)
        
        # Get strategy name
        strategy = self.intent_strategies.get(intent_result.intent, "generic")
        
        # Build debug info
        debug_info = {
            "intent": intent_result.intent.value,
            "confidence": intent_result.confidence,
            "requires_clarification": intent_result.requires_clarification,
            "context_messages": len(context),
            "strategy_applied": strategy,
            "level_triggered": f"Level {level}"
        }
        
        print(f"✅ Context built: Level {level}, Strategy: {strategy}, Messages: {len(context)}, Tokens: {tokens}")
        
        return ContextBuildResult(
            context=context,
            tokens_used=tokens,
            level_used=level,
            strategy=strategy,
            debug_info=debug_info
        )
    
    def _build_level_1_context(self, intent_result: IntentClassification, conv_id: str, user_message: str) -> Tuple[List[Dict], int]:
        """
        Level 1: Fast context for high-confidence intents (≥0.9)
        Strategy: Minimal processing, quick response
        """
        
        print(f"🚀 Level 1 (Fast): High confidence intent processing")
        
        strategy = self.intent_strategies.get(intent_result.intent, "generic")
        
        if strategy == "continuation":
            # For continuations, focus on recent work and active threads
            return self._build_continuation_fast_context(conv_id, user_message)
        elif strategy == "minimal":
            # For conversational filler, minimal context
            return self._build_minimal_context(conv_id, user_message)
        elif strategy == "new_request":
            # For new requests, fresh start with project overview
            return self._build_new_request_fast_context(conv_id, user_message)
        else:
            # Generic fast context - recent messages
            return self._build_recent_context(conv_id, user_message, self.level_1_depth)
    
    def _build_level_2_context(self, intent_result: IntentClassification, conv_id: str, user_message: str) -> Tuple[List[Dict], int]:
        """
        Level 2: Semantic context for medium-confidence intents (0.7-0.9)
        Strategy: Semantic search + thread awareness + memory integration
        """
        
        print(f"🧠 Level 2 (Semantic): Medium confidence semantic processing")
        
        strategy = self.intent_strategies.get(intent_result.intent, "generic")
        
        # Start with thread-aware context (existing functionality)
        thread_context, thread_tokens = self.thread_manager.prepare_context_with_threads(conv_id, user_message)
        
        # Apply intent-specific enhancements
        if strategy == "debugging":
            return self._enhance_debugging_context(thread_context, conv_id, user_message)
        elif strategy == "explanation":
            return self._enhance_explanation_context(thread_context, conv_id, user_message)
        elif strategy == "continuation":
            return self._enhance_continuation_context(thread_context, conv_id, user_message)
        else:
            # For unimplemented strategies, return thread-aware context
            return thread_context, thread_tokens
    
    def _build_level_3_context(self, intent_result: IntentClassification, conv_id: str, user_message: str) -> Tuple[List[Dict], int]:
        """
        Level 3: Clarification context for low-confidence intents (<0.7)
        Strategy: Ask for clarification before proceeding
        """
        
        print(f"❓ Level 3 (Clarification): Low confidence, requesting clarification")
        
        # Build clarification prompt
        clarification_prompt = self._build_clarification_prompt(intent_result, user_message)
        
        # Get minimal recent context for clarification
        recent_context, recent_tokens = self._build_recent_context(conv_id, clarification_prompt, self.level_3_depth)
        
        # Enhance system message with clarification instructions
        if recent_context and recent_context[0].get('role') == 'system':
            enhanced_system = self._build_clarification_system_prompt(intent_result, recent_context[0]['content'])
            recent_context[0]['content'] = enhanced_system
        
        # Calculate tokens (recalculate since we modified the system message)
        total_tokens = sum(self.count_tokens(msg.get("content", "")) for msg in recent_context)
        
        return recent_context, total_tokens
    
    # Helper methods for context building
    
    def _build_recent_context(self, conv_id: str, user_message: str, depth: int) -> Tuple[List[Dict], int]:
        """
        Build context with recent messages - FAST DIRECT APPROACH
        No API calls, no complex logic, just grab last N messages
        """
        try:
            print(f"🚀 Fast direct context: fetching last {depth} messages")
            
            # Get conversation directly (no expensive processing)
            conversation = self.base_manager.get_conversation(conv_id)
            
            # Take exactly the last N messages (chronological)
            recent_messages = conversation[-depth:] if conversation else []
            
            # Build simple, fast context
            system_content = (
                "You are a helpful assistant that maintains conversation context. "
                "Provide thoughtful, accurate responses based on the conversation history."
            )
            
            context = [{"role": "system", "content": system_content}]
            
            # Add recent conversation messages
            if recent_messages:
                context.extend(recent_messages)
                print(f"   Added {len(recent_messages)} recent messages")
            else:
                print(f"   No previous messages in conversation")
            
            # Add current user message
            context.append({"role": "user", "content": user_message})
            
            # Calculate tokens directly (no double processing)
            total_tokens = sum(self.count_tokens(msg.get("content", "")) for msg in context)
            
            print(f"   Fast context built: {len(context)} messages, {total_tokens} tokens, 0 API calls")
            return context, total_tokens
            
        except Exception as e:
            print(f"❌ Error in fast recent context building: {e}")
            # Fallback to minimal context
            return self._build_minimal_context(conv_id, user_message)
    
    #region Context builders for level 3 type context

    def _build_clarification_prompt(self, intent_result: IntentClassification, user_message: str) -> str:
        """Build clarification prompt for ambiguous intents"""
        
        clarification_base = f"I want to make sure I understand your request correctly. "
        
        if intent_result.clarification_reason:
            clarification_base += f"{intent_result.clarification_reason} "
        
        # Add intent-specific clarification questions
        if intent_result.intent == IntentCategory.NEW_REQUEST:
            clarification_base += "Are you looking to start a new project or feature?"
        elif intent_result.intent == IntentCategory.DEBUGGING:
            clarification_base += "Are you experiencing a specific error or issue that needs fixing?"
        elif intent_result.intent == IntentCategory.EXPLANATION:
            clarification_base += "What specific concept or topic would you like me to explain?"
        else:
            clarification_base += "Could you provide more details about what you're trying to accomplish?"
        
        clarification_base += f"\n\nYour original message: \"{user_message}\""
        
        return clarification_base
    
    def _build_clarification_system_prompt(self, intent_result: IntentClassification, base_system: str) -> str:
        """Enhance system prompt for clarification scenarios"""
        
        clarification_enhancement = (
            "\n\nIMPORTANT: The user's intent is unclear. Ask for clarification before proceeding. "
            "Be specific about what information you need to provide the best help."
        )
        
        return base_system + clarification_enhancement
    
    #endregion

    #region Intent-specific context builders (Phase 2.1 - Core strategies - Fast contexts)
    
    def _build_continuation_fast_context(self, conv_id: str, user_message: str) -> Tuple[List[Dict], int]:
        """Fast context for continuation intents - focus on recent work"""
        
        print("🔄 Fast continuation context: Recent work + active threads")
        
        # Get recent context but prioritize continuity
        context, tokens = self._build_recent_context(conv_id, user_message, self.level_1_depth)
        
        # Enhance system message for continuation
        if context and context[0].get('role') == 'system':
            continuation_enhancement = (
                " Focus on maintaining continuity with the current work and "
                "building upon what has already been established."
            )
            context[0]['content'] += continuation_enhancement
            tokens += self.count_tokens(continuation_enhancement)
        
        return context, tokens
    
    def _build_new_request_fast_context(self, conv_id: str, user_message: str) -> Tuple[List[Dict], int]:
        """Fast context for new requests - fresh start approach"""
        
        print("✨ Fast new request context: Fresh start with minimal history")
        
        # Get minimal conversation overview
        conversation = self.base_manager.get_conversation(conv_id)
        
        # Build fresh context
        system_content = (
            "You are a helpful assistant starting a new task or project. "
            "Focus on understanding requirements and providing a clear starting point."
        )
        
        context = [{"role": "system", "content": system_content}]
        
        # Add brief project context if conversation exists
        if conversation:
            overview = f"Note: This is part of an ongoing conversation with {len(conversation)} previous messages."
            context.append({"role": "system", "content": overview})
        
        # Add user message
        context.append({"role": "user", "content": user_message})
        
        # Calculate tokens
        tokens = sum(self.count_tokens(msg.get("content", "")) for msg in context)
        
        return context, tokens
    
    def _build_minimal_context(self, conv_id: str, user_message: str) -> Tuple[List[Dict], int]:
        """Build minimal context for simple interactions"""
        system_content = (
            "You are a helpful assistant. Provide clear, concise responses."
        )
        
        context = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message}
        ]
        
        tokens = self.count_tokens(system_content) + self.count_tokens(user_message)
        return context, tokens
        
    #endregion

    #region Intent-specific enhancements for Level 2 (Phase 2.1 - placeholders for now)
    
    def _enhance_debugging_context(self, base_context: List[Dict], conv_id: str, user_message: str) -> Tuple[List[Dict], int]:
        """Enhance context for debugging intents (placeholder for Phase 2.2)"""
        print("🐛 Debugging context enhancement (placeholder)")
        
        # For now, just enhance system message
        if base_context and base_context[0].get('role') == 'system':
            debug_enhancement = (
                " Pay special attention to error messages, stack traces, and "
                "recent code changes when helping with debugging."
            )
            base_context[0]['content'] += debug_enhancement
        
        tokens = sum(self.count_tokens(msg.get("content", "")) for msg in base_context)
        return base_context, tokens
    
    def _enhance_explanation_context(self, base_context: List[Dict], conv_id: str, user_message: str) -> Tuple[List[Dict], int]:
        """Enhance context for explanation intents (placeholder for Phase 2.2)"""
        print("📚 Explanation context enhancement (placeholder)")
        
        # For now, just enhance system message
        if base_context and base_context[0].get('role') == 'system':
            explanation_enhancement = (
                " Focus on providing clear, educational explanations with examples. "
                "Adjust complexity based on the user's apparent expertise level."
            )
            base_context[0]['content'] += explanation_enhancement
        
        tokens = sum(self.count_tokens(msg.get("content", "")) for msg in base_context)
        return base_context, tokens
    
    def _enhance_continuation_context(self, base_context: List[Dict], conv_id: str, user_message: str) -> Tuple[List[Dict], int]:
        """Enhance context for continuation intents (placeholder for Phase 2.2)"""
        print("🔄 Continuation context enhancement (placeholder)")
        
        # For now, just enhance system message
        if base_context and base_context[0].get('role') == 'system':
            continuation_enhancement = (
                " Maintain consistency with previous work and technology choices. "
                "Build upon existing progress rather than starting over."
            )
            base_context[0]['content'] += continuation_enhancement
        
        tokens = sum(self.count_tokens(msg.get("content", "")) for msg in base_context)
        return base_context, tokens
    #endregion 

    #region Debug and utility methods
    
    def get_context_statistics(self) -> Dict:
        """Get statistics about context building for debugging"""
        return {
            "level_1_confidence_threshold": self.level_1_confidence,
            "level_2_confidence_threshold": self.level_2_confidence,
            "level_3_confidence_threshold": self.level_3_confidence,
            "level_1_depth": self.level_1_depth,
            "level_2_max_context": self.level_2_max_context,
            "level_3_depth": self.level_3_depth,
            "implemented_strategies": [k.value for k, v in self.intent_strategies.items() if v != "generic"],
            "placeholder_strategies": [k.value for k, v in self.intent_strategies.items() if v == "generic"],
            "max_tokens": self.max_tokens,
            "reserved_tokens": self.reserved_tokens
        }
    
    #endregion 

#region Test utilities for development

def test_multilayered_context_builder():
    """Test function for development and debugging"""
    print("🧪 Testing MultiLayeredContextBuilder (placeholder)")
    print("This will be expanded when we integrate with main.py")


if __name__ == "__main__":
    print("MultiLayeredContextBuilder - Phase 2 Implementation")
    print("=" * 50)
    test_multilayered_context_builder()

#endregion