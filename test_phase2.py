#!/usr/bin/env python3
"""
Phase 2 Test Script
Test the MultiLayeredContextBuilder with different intent scenarios
"""

import sys
import os
sys.path.append('src')

from intent_classification_system import IntentCategory, IntentClassification
from multilayered_context_builder import MultiLayeredContextBuilder

# Mock managers for testing
class MockConversationManager:
    def __init__(self):
        self.max_tokens = 60000
        self.reserved_tokens = 4000
        
    def count_tokens(self, text):
        return len(text) // 4
        
    def get_conversation(self, conv_id):
        # Mock conversation history
        return [
            {"role": "user", "content": "I want to build a React app"},
            {"role": "assistant", "content": "I'll help you build a React app. Let's start with create-react-app..."},
            {"role": "user", "content": "Great! Now I need to add authentication"},
            {"role": "assistant", "content": "For authentication, I recommend using JWT tokens with a backend API..."}
        ]
        
    def prepare_context(self, conv_id, user_message):
        conversation = self.get_conversation(conv_id)
        context = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        context.extend(conversation)
        context.append({"role": "user", "content": user_message})
        
        tokens = sum(self.count_tokens(msg.get("content", "")) for msg in context)
        return context, tokens

class MockThreadManager:
    def prepare_context_with_threads(self, conv_id, user_message):
        # Mock thread-aware context
        return [
            {"role": "system", "content": "You are a helpful assistant with thread awareness."},
            {"role": "user", "content": "Previous work on React app"},
            {"role": "assistant", "content": "Working on authentication system"},
            {"role": "user", "content": user_message}
        ], 200

def test_multilayered_context():
    """Test the multilayered context builder with different scenarios"""
    
    print("🧪 Testing Phase 2: MultiLayeredContextBuilder")
    print("=" * 60)
    
    # Initialize mock managers
    mock_conv_manager = MockConversationManager()
    mock_thread_manager = MockThreadManager()
    
    # Initialize context builder
    context_builder = MultiLayeredContextBuilder(
        base_conv_manager=mock_conv_manager,
        thread_aware_manager=mock_thread_manager,
        memory_manager=None
    )
    
    # Test cases for different intent scenarios
    test_cases = [
        {
            "name": "High Confidence NEW_REQUEST",
            "intent": IntentCategory.NEW_REQUEST,
            "confidence": 0.95,
            "message": "I want to build a new Express.js API",
            "expected_level": 1
        },
        {
            "name": "Medium Confidence CONTINUATION", 
            "intent": IntentCategory.CONTINUATION,
            "confidence": 0.85,
            "message": "Continue working on the authentication system",
            "expected_level": 2
        },
        {
            "name": "Low Confidence DEBUGGING",
            "intent": IntentCategory.DEBUGGING,
            "confidence": 0.65,
            "message": "Something is not working properly",
            "expected_level": 3
        },
        {
            "name": "High Confidence DEBUGGING",
            "intent": IntentCategory.DEBUGGING, 
            "confidence": 0.92,
            "message": "Fix this TypeError: Cannot read property 'name' of undefined",
            "expected_level": 1
        },
        {
            "name": "Medium Confidence EXPLANATION",
            "intent": IntentCategory.EXPLANATION,
            "confidence": 0.78,
            "message": "How does JWT authentication work?",
            "expected_level": 2
        }
    ]
    
    print("📊 Test Results:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Intent: {test_case['intent'].value}")
        print(f"   Message: \"{test_case['message']}\"")
        print(f"   Confidence: {test_case['confidence']}")
        
        # Create intent classification
        intent_result = IntentClassification(
            intent=test_case['intent'],
            confidence=test_case['confidence'],
            reasoning=f"Test case for {test_case['name']}"
        )
        
        # Build context
        try:
            context_result = context_builder.build_context(
                intent_result, 
                "test_conversation", 
                test_case['message']
            )
            
            # Verify results
            level_match = context_result.level_used == test_case['expected_level']
            status = "✅ PASS" if level_match else "❌ FAIL"
            
            print(f"   Expected Level: {test_case['expected_level']}")
            print(f"   Actual Level: {context_result.level_used} {status}")
            print(f"   Strategy: {context_result.strategy}")
            print(f"   Context Messages: {len(context_result.context)}")
            print(f"   Tokens Used: {context_result.tokens_used}")
            
            if not level_match:
                print(f"   ⚠️  Level mismatch! Expected {test_case['expected_level']}, got {context_result.level_used}")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Context Level Logic Test:")
    print(f"   Level 1 (Fast): confidence ≥ {context_builder.level_1_confidence}")
    print(f"   Level 2 (Semantic): confidence ≥ {context_builder.level_2_confidence}")  
    print(f"   Level 3 (Clarification): confidence < {context_builder.level_3_confidence}")
    
    print("\n🚀 Strategy Implementation Status:")
    stats = context_builder.get_context_statistics()
    print(f"   ✅ Implemented: {len(stats['implemented_strategies'])} strategies")
    print(f"   📝 Placeholders: {len(stats['placeholder_strategies'])} strategies")
    
    print("\n✅ Phase 2 Core Testing Complete!")
    print("🎯 Next: Test with real server using /debug/multilayer-context endpoint")

if __name__ == "__main__":
    test_multilayered_context()
