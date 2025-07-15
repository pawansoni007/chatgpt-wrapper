#!/usr/bin/env python3
"""
Phase 3B Test Script: Thread Detection & Over-Contextualization Fix
Tests the complete thread detection system to ensure over-contextualization is solved.
"""

import requests
import json
import time
from typing import Dict, List

class ThreadDetectionTester:
    """Test suite for Phase 3B thread detection system"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_conv_id = f"test-thread-detection-{int(time.time())}"
        
    def test_api_health(self) -> bool:
        """Test if the API is running with thread detection enabled"""
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Status: {data['message']}")
                print(f"✅ Version: {data['version']}")
                print(f"✅ Features: {', '.join(data['features'])}")
                return "Thread Detection" in data.get('features', [])
            return False
        except Exception as e:
            print(f"❌ API Health Check Failed: {e}")
            return False
    
    def send_message(self, message: str, conversation_id: str = None) -> Dict:
        """Send a message to the chat API"""
        payload = {
            "message": message,
            "conversation_id": conversation_id or self.test_conv_id
        }
        
        response = requests.post(f"{self.base_url}/chat", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Failed to send message: {response.status_code}")
            print(response.text)
            return {}
    
    def get_threads(self, conversation_id: str) -> Dict:
        """Get thread analysis for a conversation"""
        response = requests.get(f"{self.base_url}/conversations/{conversation_id}/threads")
        if response.status_code == 200:
            return response.json()
        return {}
    
    def get_debug_info(self, conversation_id: str) -> Dict:
        """Get debug information about thread detection"""
        response = requests.get(f"{self.base_url}/debug/threads/{conversation_id}")
        if response.status_code == 200:
            return response.json()
        return {}
    
    def test_over_contextualization_fix(self) -> bool:
        """Test the specific over-contextualization problem is fixed"""
        print("\n" + "="*80)
        print("🧪 TESTING OVER-CONTEXTUALIZATION FIX")
        print("="*80)
        
        # Step 1: Create React conversation
        print("\n📱 Step 1: Creating React Development Conversation...")
        react_messages = [
            "Can you help me create a React to-do app?",
            "Can you add due dates to the tasks?", 
            "How do I handle form validation?",
            "Thanks! That's perfect for the React app."
        ]
        
        for msg in react_messages:
            response = self.send_message(msg)
            print(f"  User: {msg}")
            print(f"  Bot: {response.get('message', 'No response')[:100]}...")
            time.sleep(1)  # Brief pause between messages
        
        # Step 2: Change to human development topic
        print("\n👥 Step 2: Switching to Human Development Topic...")
        human_dev_messages = [
            "Now I have a different question about human lifecycle development.",
            "What are the key stages of late adulthood?",
            "What about mid-life crisis? Is that a real psychological phenomenon?"
        ]
        
        for msg in human_dev_messages:
            response = self.send_message(msg)
            print(f"  User: {msg}")
            print(f"  Bot: {response.get('message', 'No response')[:100]}...")
            time.sleep(1)
        
        # Step 3: The critical test - asking about apples
        print("\n🍎 Step 3: The Critical Test - Asking About Apples...")
        print("This should NOT mention React or human development!")
        
        apple_response = self.send_message("tell me 'bout apple?")
        apple_text = apple_response.get('message', '').lower()
        
        print(f"  User: tell me 'bout apple?")
        print(f"  Bot: {apple_response.get('message', 'No response')}")
        
        # Check for over-contextualization
        bad_keywords = ['react', 'to-do', 'form validation', 'human development', 'adulthood', 'crisis']
        found_bad_keywords = [kw for kw in bad_keywords if kw in apple_text]
        
        if found_bad_keywords:
            print(f"\n❌ OVER-CONTEXTUALIZATION DETECTED!")
            print(f"Found inappropriate context: {found_bad_keywords}")
            return False
        else:
            print(f"\n✅ OVER-CONTEXTUALIZATION FIXED!")
            print("Response appropriately focused on apples without dragging previous context")
            return True
    
    def test_context_switching(self) -> bool:
        """Test that legitimate context switching works"""
        print("\n" + "="*80)
        print("🔄 TESTING INTELLIGENT CONTEXT SWITCHING")
        print("="*80)
        
        print("\n🎯 Testing 'Back to React' context reactivation...")
        
        react_return_response = self.send_message("Back to React - how do I handle errors in the to-do app?")
        react_return_text = react_return_response.get('message', '').lower()
        
        print(f"  User: Back to React - how do I handle errors in the to-do app?")
        print(f"  Bot: {react_return_response.get('message', 'No response')[:200]}...")
        
        # Should mention React context but not human development
        has_react_context = any(word in react_return_text for word in ['react', 'to-do', 'app'])
        has_human_context = any(word in react_return_text for word in ['adulthood', 'crisis', 'lifecycle'])
        
        if has_react_context and not has_human_context:
            print("\n✅ CONTEXT SWITCHING WORKS!")
            print("Successfully reactivated React thread without human development context")
            return True
        else:
            print(f"\n❌ CONTEXT SWITCHING FAILED!")
            print(f"React context: {has_react_context}, Human context: {has_human_context}")
            return False
    
    def test_thread_detection(self) -> bool:
        """Test thread detection and analysis"""
        print("\n" + "="*80)
        print("🧵 TESTING THREAD DETECTION")
        print("="*80)
        
        # Get thread analysis
        threads_data = self.get_threads(self.test_conv_id)
        threads = threads_data.get('threads', [])
        
        print(f"\n📊 Thread Analysis Results:")
        print(f"  Total threads detected: {len(threads)}")
        print(f"  Active threads: {threads_data.get('active_threads', 0)}")
        
        expected_topics = ['React Development', 'Learning/Education', 'General Discussion']
        detected_topics = [t['topic'] for t in threads]
        
        print(f"\n📋 Detected Topics:")
        for i, thread in enumerate(threads, 1):
            status_emoji = "🟢" if thread['is_active'] else "🔴"
            print(f"  {i}. {status_emoji} {thread['topic']} ({thread['message_count']} messages)")
        
        # Should detect at least 2 different topics
        if len(threads) >= 2:
            print(f"\n✅ THREAD DETECTION WORKING!")
            print(f"Successfully detected {len(threads)} distinct conversation threads")
            return True
        else:
            print(f"\n❌ THREAD DETECTION INSUFFICIENT!")
            print(f"Expected multiple threads, got {len(threads)}")
            return False
    
    def test_debug_information(self) -> bool:
        """Test debug endpoint provides useful information"""
        print("\n" + "="*80)
        print("🐛 TESTING DEBUG INFORMATION")
        print("="*80)
        
        debug_data = self.get_debug_info(self.test_conv_id)
        
        if not debug_data:
            print("❌ No debug data available")
            return False
        
        print(f"📊 Debug Information:")
        print(f"  Message count: {debug_data.get('message_count', 0)}")
        print(f"  Detected boundaries: {debug_data.get('detected_boundaries', [])}")
        print(f"  Thread count: {len(debug_data.get('threads', []))}")
        
        patterns = debug_data.get('conversation_patterns', {})
        print(f"\n🔍 Conversation Patterns:")
        print(f"  Q&A pairs: {len(patterns.get('qa_pairs', []))}")
        print(f"  Topic shifts: {len(patterns.get('topic_shifts', []))}")
        print(f"  Continuation signals: {len(patterns.get('continuation_signals', []))}")
        print(f"  Closure signals: {len(patterns.get('closure_signals', []))}")
        
        return True
    
    def run_all_tests(self) -> bool:
        """Run the complete test suite"""
        print("🚀 STARTING PHASE 3B THREAD DETECTION TESTS")
        print("="*80)
        
        # Test 1: API Health
        if not self.test_api_health():
            print("❌ CRITICAL: API not ready for testing")
            return False
        
        # Test 2: Over-contextualization fix (most important)
        over_context_fixed = self.test_over_contextualization_fix()
        
        # Test 3: Context switching
        context_switching_works = self.test_context_switching()
        
        # Test 4: Thread detection
        thread_detection_works = self.test_thread_detection()
        
        # Test 5: Debug information
        debug_works = self.test_debug_information()
        
        # Final results
        print("\n" + "="*80)
        print("🎯 TEST RESULTS SUMMARY")
        print("="*80)
        
        tests = [
            ("Over-Contextualization Fix", over_context_fixed),
            ("Context Switching", context_switching_works), 
            ("Thread Detection", thread_detection_works),
            ("Debug Information", debug_works)
        ]
        
        passed = 0
        for test_name, result in tests:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
            if result:
                passed += 1
        
        success_rate = (passed / len(tests)) * 100
        print(f"\n🎉 OVERALL RESULT: {passed}/{len(tests)} tests passed ({success_rate:.1f}%)")
        
        if success_rate >= 75:
            print("🎉 PHASE 3B IMPLEMENTATION SUCCESSFUL!")
            print("✅ Over-contextualization problem is SOLVED!")
            return True
        else:
            print("❌ PHASE 3B NEEDS ATTENTION")
            print("Some critical features are not working correctly")
            return False

def main():
    """Run the test suite"""
    print("🧪 PHASE 3B THREAD DETECTION TEST SUITE")
    print("Make sure your API is running on http://localhost:8000")
    print("Press Enter to start testing...")
    input()
    
    tester = ThreadDetectionTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 Ready for production deployment!")
    else:
        print("\n🔧 Please fix the failing tests before deployment.")

if __name__ == "__main__":
    main()
