#!/usr/bin/env python3
"""
Basic API Test Script for Phase 3B
Quick verification that the thread detection system is working
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_basic_functionality():
    """Test basic API functionality"""
    print("🔍 Testing basic API functionality...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"✅ API Health: {response.json()['message']}")
        print(f"✅ Version: {response.json()['version']}")
    except Exception as e:
        print(f"❌ API Health failed: {e}")
        return False
    
    # Test chat endpoint
    try:
        chat_response = requests.post(f"{BASE_URL}/chat", json={
            "message": "Hello, can you help me test the thread detection system?"
        })
        
        if chat_response.status_code == 200:
            data = chat_response.json()
            print(f"✅ Chat works: {data['conversation_id']}")
            print(f"✅ Response length: {len(data['message'])} characters")
            return data['conversation_id']
        else:
            print(f"❌ Chat failed: {chat_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False

def test_thread_endpoints(conv_id):
    """Test thread-specific endpoints"""
    print(f"\n🧵 Testing thread endpoints for conversation {conv_id}...")
    
    # Test threads endpoint
    try:
        threads_response = requests.get(f"{BASE_URL}/conversations/{conv_id}/threads")
        if threads_response.status_code == 200:
            threads_data = threads_response.json()
            print(f"✅ Threads endpoint works")
            print(f"✅ Thread count: {threads_data.get('total_threads', 0)}")
        else:
            print(f"❌ Threads endpoint failed: {threads_response.status_code}")
    except Exception as e:
        print(f"❌ Threads test failed: {e}")
    
    # Test debug endpoint
    try:
        debug_response = requests.get(f"{BASE_URL}/debug/threads/{conv_id}")
        if debug_response.status_code == 200:
            debug_data = debug_response.json()
            print(f"✅ Debug endpoint works")
            print(f"✅ Message count: {debug_data.get('message_count', 0)}")
        else:
            print(f"❌ Debug endpoint failed: {debug_response.status_code}")
    except Exception as e:
        print(f"❌ Debug test failed: {e}")

def test_over_contextualization_scenario():
    """Test the specific over-contextualization scenario"""
    print("\n🍎 Testing over-contextualization fix...")
    
    conv_id = f"over-context-test-{int(requests.time.time()) if hasattr(requests, 'time') else '123'}"
    
    # Create a conversation with topic changes
    messages = [
        "Help me create a React app",
        "Now tell me about human psychology", 
        "What are apples?"
    ]
    
    for i, msg in enumerate(messages):
        try:
            response = requests.post(f"{BASE_URL}/chat", json={
                "message": msg,
                "conversation_id": conv_id
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"  Message {i+1}: ✅")
                
                # Check the apple response specifically
                if "apple" in msg.lower():
                    apple_response = data['message'].lower()
                    bad_words = ['react', 'psychology', 'human']
                    found_bad = [w for w in bad_words if w in apple_response]
                    
                    if found_bad:
                        print(f"    ⚠️  Found context leakage: {found_bad}")
                    else:
                        print(f"    ✅ Clean apple response (no context leakage)")
            else:
                print(f"  Message {i+1}: ❌ Failed ({response.status_code})")
                
        except Exception as e:
            print(f"  Message {i+1}: ❌ Error ({e})")

def main():
    """Run basic tests"""
    print("🧪 BASIC PHASE 3B TESTS")
    print("="*50)
    
    # Test 1: Basic functionality
    conv_id = test_basic_functionality()
    
    if conv_id:
        # Test 2: Thread endpoints
        test_thread_endpoints(conv_id)
        
        # Test 3: Over-contextualization
        test_over_contextualization_scenario()
        
        print("\n🎉 Basic tests completed!")
        print("Run the full test suite with: python tests/test_thread_detection.py")
    else:
        print("\n❌ Basic tests failed - check your API server")

if __name__ == "__main__":
    main()
