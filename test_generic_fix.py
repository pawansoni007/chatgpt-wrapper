#!/usr/bin/env python3
"""
Test Level 1 Fix - Direct Recent Context Path
Test the exact _build_recent_context method we fixed
"""

import sys
sys.path.append('src')

from intent_classification_system import IntentCategory, IntentClassification
from multilayered_context_builder import MultiLayeredContextBuilder

class MockLongConversationManager:
    def __init__(self):
        self.max_tokens = 60000
        self.reserved_tokens = 4000
        self.simple_conversation_threshold = 5
        
    def count_tokens(self, text):
        return len(text) // 4
        
    def get_conversation(self, conv_id):
        # 10 messages - this would trigger expensive semantic search in OLD approach
        return [
            {"role": "user", "content": "Start building React auth system"},
            {"role": "assistant", "content": "I'll help you build authentication..."},
            {"role": "user", "content": "Add JWT token handling"},
            {"role": "assistant", "content": "Here's JWT implementation..."},
            {"role": "user", "content": "Set up login form"},
            {"role": "assistant", "content": "Login form component created..."},
            {"role": "user", "content": "Add form validation"},
            {"role": "assistant", "content": "Validation rules added..."},
            {"role": "user", "content": "Style the components"},
            {"role": "assistant", "content": "CSS styling applied..."}
        ]
        
    def prepare_context(self, conv_id, user_message):
        conversation = self.get_conversation(conv_id)
        print(f"💸 OLD METHOD CALLED: prepare_context with {len(conversation)} messages")
        print(f"💸 Would trigger EXPENSIVE semantic search (len > {self.simple_conversation_threshold})")
        return [{"role": "system", "content": "Expensive semantic result"}], 200

def test_recent_context_fix():
    print("🧪 Testing _build_recent_context Fix")
    print("=" * 50)
    
    mock_conv_manager = MockLongConversationManager()
    
    context_builder = MultiLayeredContextBuilder(
        base_conv_manager=mock_conv_manager,
        thread_aware_manager=None,
        memory_manager=None
    )
    
    conversation = mock_conv_manager.get_conversation("test")
    print(f"📊 Setup: {len(conversation)} messages (>{mock_conv_manager.simple_conversation_threshold} threshold)")
    
    # Test with GENERIC strategy (not NEW_REQUEST) to hit _build_recent_context path
    print(f"\n🎯 Testing GENERIC Level 1 strategy:")
    
    # Use ARTIFACT_GENERATION intent which maps to "generic" strategy
    intent_result = IntentClassification(
        intent=IntentCategory.ARTIFACT_GENERATION,  # This maps to "generic" strategy
        confidence=0.95,  # High confidence = Level 1
        reasoning="High confidence generic request"
    )
    
    user_message = "Add unit tests for the auth system"
    print(f"   Intent: {intent_result.intent.value} (generic strategy)")
    print(f"   Confidence: {intent_result.confidence} (Level 1)")
    print(f"   Message: \"{user_message}\"")
    
    # This should use the generic Level 1 path which calls _build_recent_context
    context_result = context_builder.build_context(intent_result, "test_conv", user_message)
    
    print(f"\n✅ Results:")
    print(f"   Level: {context_result.level_used}")
    print(f"   Strategy: {context_result.strategy}")
    print(f"   Messages: {len(context_result.context)}")
    print(f"   Tokens: {context_result.tokens_used}")
    
    # Check if we avoided calling the expensive base manager method
    print(f"\n🚀 Performance Check:")
    print(f"   Used fast path: {'✅' if context_result.level_used == 1 else '❌'}")
    print(f"   Direct array access: {'✅' if 'Fast direct context' in str(context_result.debug_info) else '❌'}")

if __name__ == "__main__":
    test_recent_context_fix()
