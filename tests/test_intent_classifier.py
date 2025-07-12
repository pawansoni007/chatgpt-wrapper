#!/usr/bin/env python3
"""
Test script for EnhancedIntentClassifier
Tests the intent classification system with your existing Cerebras setup
"""

import sys
import os
from dotenv import load_dotenv

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

# Load environment variables
load_dotenv()

from cerebras.cloud.sdk import Cerebras
from intent_classification_system import EnhancedIntentClassifier, IntentCategory

def test_basic_classification():
    """Test basic intent classification functionality"""
    
    print("🧪 Testing EnhancedIntentClassifier")
    print("=" * 60)
    
    # Initialize Cerebras client (same as in main.py)
    try:
        cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        classifier = EnhancedIntentClassifier(cerebras_client)
        
        print("✅ Cerebras client initialized successfully")
        print("✅ Intent classifier created successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False
    
    # Test cases covering all 11 intent categories
    test_cases = [
        # Task-Oriented Intents
        ("Can you help me build a new authentication system for my app?", "NEW_REQUEST"),
        ("Continue with the user registration feature", "CONTINUATION"), 
        ("Make this code more efficient and clean", "REFINEMENT"),
        ("I'm getting a TypeError in my Python script", "DEBUGGING"),
        ("Write unit tests for the login function", "ARTIFACT_GENERATION"),
        
        # Knowledge-Seeking Intents
        ("What's the status of our project? Are we done with auth?", "STATUS_CHECK"),
        ("How does JWT authentication work?", "EXPLANATION"),
        ("Compare MongoDB vs PostgreSQL for this use case", "COMPARISON"),
        
        # Context-Management Intents
        ("Actually, I meant Express.js not React", "CORRECTION"),
        ("Please always format code blocks in markdown", "META_INSTRUCTION"),
        ("Thanks! That's exactly what I needed", "CONVERSATIONAL_FILLER")
    ]
    
    print(f"\n🎯 Testing {len(test_cases)} sample messages:")
    print("-" * 60)
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, (message, expected_intent) in enumerate(test_cases, 1):
        try:
            # Classify intent
            result = classifier.classify_intent(message)
            
            # Check if prediction matches expected
            is_correct = result.intent.value == expected_intent
            if is_correct:
                correct_predictions += 1
            
            # Display result
            status = "✅" if is_correct else "❌"
            print(f"\n{i:2d}. {status} Message: \"{message[:50]}{'...' if len(message) > 50 else ''}\"")
            print(f"     Expected: {expected_intent}")
            print(f"     Got:      {result.intent.value} (confidence: {result.confidence:.2f})")
            print(f"     Reasoning: {result.reasoning}")
            
            if result.requires_clarification:
                print(f"     ⚠️  Clarification needed: {result.clarification_reason}")
                
        except Exception as e:
            print(f"\n{i:2d}. ❌ Error testing message: {e}")
    
    # Summary
    accuracy = (correct_predictions / total_tests) * 100
    print("\n" + "=" * 60)
    print(f"📊 Test Results:")
    print(f"   Correct predictions: {correct_predictions}/{total_tests}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 70:
        print("✅ Intent classifier is working well!")
    elif accuracy >= 50:
        print("⚠️  Intent classifier needs tuning")
    else:
        print("❌ Intent classifier needs significant improvement")
    
    return accuracy >= 50

def test_context_awareness():
    """Test intent classification with conversation context"""
    
    print("\n🔄 Testing Context-Aware Classification")
    print("-" * 60)
    
    cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
    classifier = EnhancedIntentClassifier(cerebras_client)
    
    # Simulate a conversation context
    conversation_context = [
        {"role": "user", "content": "Help me build a React authentication system"},
        {"role": "assistant", "content": "I'll help you create a React auth system with JWT tokens..."},
        {"role": "user", "content": "Great! Can you also add user registration?"},
        {"role": "assistant", "content": "Sure! Here's the registration component..."}
    ]
    
    # Test messages that depend on context
    context_tests = [
        ("Continue with that", "CONTINUATION"),  # Should understand "that" refers to auth system
        ("Add password reset too", "CONTINUATION"),  # Adding to existing work
        ("Actually, use OAuth instead", "CORRECTION"),  # Correcting previous approach
    ]
    
    for message, expected in context_tests:
        result = classifier.classify_intent(message, conversation_context)
        
        status = "✅" if result.intent.value == expected else "❌"
        print(f"{status} \"{message}\" -> {result.intent.value} (confidence: {result.confidence:.2f})")
        print(f"   Context reasoning: {result.reasoning}")

def test_edge_cases():
    """Test edge cases and error handling"""
    
    print("\n🧪 Testing Edge Cases")
    print("-" * 60)
    
    cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
    classifier = EnhancedIntentClassifier(cerebras_client)
    
    edge_cases = [
        "",  # Empty message
        "   ",  # Whitespace only
        "a",  # Single character
        "🚀🎯💻",  # Emojis only
        "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco",  # Very long message
    ]
    
    for i, message in enumerate(edge_cases, 1):
        try:
            result = classifier.classify_intent(message)
            print(f"{i}. \"{repr(message)}\" -> {result.intent.value} (confidence: {result.confidence:.2f})")
        except Exception as e:
            print(f"{i}. ❌ Error with \"{repr(message)}\": {e}")

def test_synchronous_wrapper():
    """Test the synchronous wrapper function"""
    
    print("\n⚡ Testing Synchronous Wrapper")
    print("-" * 60)
    
    cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
    classifier = EnhancedIntentClassifier(cerebras_client)
    
    try:
        # Test direct call (no longer need sync wrapper)
        result = classifier.classify_intent("Help me debug this error")
        print(f"✅ Direct call works: {result.intent.value} (confidence: {result.confidence:.2f})")
        
        # Test backward compatibility wrapper
        result2 = classifier.classify_intent_sync("Help me debug this error")
        print(f"✅ Sync wrapper (backward compat): {result2.intent.value} (confidence: {result2.confidence:.2f})")
        
        # Test statistics
        stats = classifier.get_intent_statistics()
        print(f"✅ Statistics: {stats['total_categories']} categories loaded")
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    
    print("🎯 EnhancedIntentClassifier Test Suite")
    print("=" * 60)
    print("Testing integration with your existing Cerebras setup...")
    
    try:
        # Check environment
        if not os.getenv("CEREBRAS_API_KEY"):
            print("❌ CEREBRAS_API_KEY not found in environment")
            print("Make sure your .env file is properly configured")
            return
        
        print("✅ Environment configured")
        
        # Run tests
        success = test_basic_classification()
        
        if success:
            test_context_awareness()
            test_edge_cases()
            test_synchronous_wrapper()
            
            print("\n🎉 All tests completed!")
            print("\n📝 Next steps:")
            print("1. Review test results above")
            print("2. Adjust confidence thresholds if needed")
            print("3. Ready to integrate with your main.py!")
        else:
            print("\n⚠️  Basic classification failed - check your Cerebras API setup")
            
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
