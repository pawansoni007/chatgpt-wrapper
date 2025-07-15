#!/usr/bin/env python3
"""
Quick test to verify intent classification integration is working
"""

import sys
import os
import requests
import json
import time

# Test the integrated intent classification
def test_integration():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Intent Classification Integration")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Server running")
            print(f"   Version: {data['version']}")
            print(f"   Features: {', '.join(data['features'])}")
            print(f"   Intent categories: {len(data['intent_categories'])}")
        else:
            print("❌ Server not responding")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return
    
    # Test intent classification debug endpoint
    test_messages = [
        ("Help me fix this bug", "DEBUGGING"),
        ("Continue with the auth system", "CONTINUATION"),
        ("How does JWT work?", "EXPLANATION"),
        ("Thanks, that's perfect!", "CONVERSATIONAL_FILLER"),
        ("Build a new React app", "NEW_REQUEST")
    ]
    
    print("\n🎯 Testing Intent Classification:")
    print("-" * 50)
    
    for message, expected in test_messages:
        try:
            response = requests.post(
                f"{base_url}/debug/intent",
                json={"message": message},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                actual = data["intent"]
                confidence = data["confidence"]
                reasoning = data["reasoning"]
                
                status = "✅" if actual == expected else "❌"
                print(f"{status} \"{message}\"")
                print(f"   Expected: {expected} | Got: {actual} ({confidence:.2f})")
                print(f"   Reasoning: {reasoning[:60]}...")
                
            else:
                print(f"❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
    
    # Test full chat with intent classification
    print("\n💬 Testing Full Chat Flow:")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"message": "I'm getting a TypeError in my Python code"},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat response received")
            print(f"   Conversation ID: {data['conversation_id']}")
            print(f"   Response length: {len(data['message'])} chars")
            print(f"   Tokens used: {data['tokens_used']}")
            print("   NOTE: Check server logs for intent classification details")
        else:
            print(f"❌ Chat failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Chat request failed: {e}")
    
    print("\n🎉 Integration testing complete!")
    print("\n📝 Next steps:")
    print("1. Check server logs to see intent classification in action")
    print("2. Ready to build MultiLayeredContextBuilder (Phase 2)")

if __name__ == "__main__":
    test_integration()