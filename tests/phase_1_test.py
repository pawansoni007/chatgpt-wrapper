#!/usr/bin/env python3
"""
Test script for Phase 1: Cohere Integration
Tests embedding generation, similarity calculation, and Redis key generation
"""

import os
import json
from dotenv import load_dotenv
from src.main import ConversationManager

# Load environment variables
load_dotenv()

def test_cohere_connection():
    """Test 1: Verify Cohere API connection"""
    print("🧪 Test 1: Cohere API Connection")
    
    conv_manager = ConversationManager()
    
    try:
        # Test with a simple message
        test_message = "Hello, this is a test message"
        embedding = conv_manager.get_message_embedding(test_message)
        
        if embedding and len(embedding) > 0:
            print(f"✅ SUCCESS: Got embedding with {len(embedding)} dimensions")
            print(f"   First 5 values: {embedding[:5]}")
            return True
        else:
            print("❌ FAILED: Empty or None embedding returned")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Cohere API error: {e}")
        return False

def test_embedding_generation():
    """Test 2: Test embedding generation for different message types"""
    print("\n🧪 Test 2: Embedding Generation")
    
    conv_manager = ConversationManager()
    
    test_messages = [
        "I need help with React state management",
        "What's the weather like today?", 
        "How do I center a div with CSS?",
        "Can you explain machine learning basics?"
    ]
    
    embeddings = {}
    
    try:
        for i, message in enumerate(test_messages):
            print(f"   Generating embedding {i+1}/4: '{message[:30]}...'")
            embedding = conv_manager.get_message_embedding(message)
            
            if embedding and len(embedding) > 0:
                embeddings[i] = embedding
                print(f"   ✅ Generated: {len(embedding)} dimensions")
            else:
                print(f"   ❌ Failed to generate embedding for message {i}")
                return False, {}
        
        print(f"✅ SUCCESS: Generated {len(embeddings)} embeddings")
        return True, embeddings
        
    except Exception as e:
        print(f"❌ FAILED: Error generating embeddings: {e}")
        return False, {}

def test_similarity_calculation(embeddings):
    """Test 3: Test cosine similarity calculations"""
    print("\n🧪 Test 3: Similarity Calculations")
    
    conv_manager = ConversationManager()
    
    try:
        # Test similarity between related messages (React and CSS - both web dev)
        react_embedding = embeddings[0]  # React state management
        css_embedding = embeddings[2]    # CSS centering
        weather_embedding = embeddings[1] # Weather (unrelated)
        
        # Calculate similarities
        similar_similarity = conv_manager.calculate_cosine_similarity(react_embedding, css_embedding)
        different_similarity = conv_manager.calculate_cosine_similarity(react_embedding, weather_embedding)
        identical_similarity = conv_manager.calculate_cosine_similarity(react_embedding, react_embedding)
        
        print(f"   React ↔ CSS similarity: {similar_similarity:.3f}")
        print(f"   React ↔ Weather similarity: {different_similarity:.3f}")
        print(f"   React ↔ React similarity: {identical_similarity:.3f}")
        
        # Validate results
        if identical_similarity > 0.99:  # Should be ~1.0 for identical
            print("   ✅ Identical similarity check passed")
        else:
            print(f"   ❌ Identical similarity should be ~1.0, got {identical_similarity}")
            return False
            
        if similar_similarity > different_similarity:
            print("   ✅ Related topics are more similar than unrelated")
        else:
            print("   ⚠️  WARNING: Related topics not more similar than unrelated")
            
        print("✅ SUCCESS: Similarity calculations working")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Error calculating similarities: {e}")
        return False

def test_redis_key_generation():
    """Test 4: Test Redis key generation"""
    print("\n🧪 Test 4: Redis Key Generation")
    
    conv_manager = ConversationManager()
    
    try:
        # Test different scenarios
        test_cases = [
            ("conv_123", 0, "embedding:conv_123:0"),
            ("abc-def-456", 5, "embedding:abc-def-456:5"),
            ("test", 999, "embedding:test:999")
        ]
        
        for conv_id, msg_index, expected in test_cases:
            result = conv_manager.get_embedding_key(conv_id, msg_index)
            if result == expected:
                print(f"   ✅ {conv_id}:{msg_index} → {result}")
            else:
                print(f"   ❌ Expected {expected}, got {result}")
                return False
                
        print("✅ SUCCESS: Redis key generation working")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Error generating Redis keys: {e}")
        return False

def test_full_workflow():
    """Test 5: Test the complete workflow"""
    print("\n🧪 Test 5: Complete Workflow Test")
    
    conv_manager = ConversationManager()
    
    try:
        # Simulate a conversation
        conv_id = "test_conversation_123"
        messages = [
            "I'm learning Python programming",
            "What's the best way to handle exceptions?",
            "Can you recommend some good books?"
        ]
        
        print("   Simulating conversation workflow...")
        
        # Generate embeddings and keys for each message
        for i, message in enumerate(messages):
            print(f"   Processing message {i}: '{message[:25]}...'")
            
            # Generate embedding
            embedding = conv_manager.get_message_embedding(message)
            if not embedding:
                print(f"   ❌ Failed to generate embedding for message {i}")
                return False
                
            # Generate Redis key
            redis_key = conv_manager.get_embedding_key(conv_id, i)
            expected_key = f"embedding:{conv_id}:{i}"
            
            if redis_key != expected_key:
                print(f"   ❌ Wrong Redis key: expected {expected_key}, got {redis_key}")
                return False
                
            print(f"   ✅ Message {i}: embedding ({len(embedding)} dims) → key: {redis_key}")
        
        print("✅ SUCCESS: Complete workflow test passed")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: Workflow error: {e}")
        return False

def main():
    """Run all Phase 1 tests"""
    print("🚀 Starting Phase 1 Cohere Integration Tests\n")
    
    tests = [
        ("Cohere Connection", test_cohere_connection),
        ("Embedding Generation", test_embedding_generation),
        ("Redis Key Generation", test_redis_key_generation),
        ("Full Workflow", test_full_workflow)
    ]
    
    results = {}
    embeddings = {}
    
    for test_name, test_func in tests:
        if test_name == "Embedding Generation":
            success, embeddings = test_func()
            results[test_name] = success
        elif test_name == "Similarity Calculation" and embeddings:
            results[test_name] = test_similarity_calculation(embeddings)
        else:
            results[test_name] = test_func()
    
    # Run similarity test if we have embeddings
    if embeddings:
        print("\n🧪 Test 3: Similarity Calculations")
        results["Similarity Calculation"] = test_similarity_calculation(embeddings)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 1 is ready for Phase 2")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Fix issues before proceeding to Phase 2")

if __name__ == "__main__":
    main()