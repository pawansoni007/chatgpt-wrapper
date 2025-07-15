#!/usr/bin/env python3
"""
Enhanced Conversation Test Runner with Intent Accuracy Calculation
================================================================

This enhanced test runner properly calculates intent accuracy and context quality
by comparing expected intents with actual intent classifications from the system.
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

class EnhancedConversationTester:
    """Enhanced test orchestrator with proper intent accuracy calculation"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {}
        self.current_conversation_id = None
        
        # Test configuration
        self.delay_between_messages = 0.5  # Seconds between messages
        self.max_retries = 3
        self.timeout = 30
        
    def generate_test_conversation_id(self, test_name: str) -> str:
        """Generate unique conversation ID for each test"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"test_{test_name}_{timestamp}_{str(uuid.uuid4())[:8]}"
    
    def send_message(self, message: str, conversation_id: str = None) -> Dict:
        """Send a message to the chat endpoint"""
        
        if conversation_id is None:
            conversation_id = self.current_conversation_id
            
        payload = {
            "message": message,
            "conversation_id": conversation_id
        }
        
        for i in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Error sending message: {e}, retrying...")
                time.sleep(2)
        
        return {"error": f"Failed to send message after {self.max_retries} retries"}
    
    def debug_intent_classification(self, message: str, conversation_id: str = None) -> Dict:
        """Debug intent classification for a message"""
        
        payload = {
            "message": message,
            "conversation_id": conversation_id
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/debug/intent",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error debugging intent: {e}")
            return {"error": str(e)}
    
    def get_conversation_details(self, conversation_id: str) -> Dict:
        """Get full conversation details including threads"""
        try:
            response = self.session.get(
                f"{self.base_url}/conversations/{conversation_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting conversation details: {e}")
            return {"error": str(e)}
    
    def calculate_intent_accuracy(self, message_responses: List[Dict]) -> float:
        """Calculate intent classification accuracy based on expected vs actual intents"""
        
        if not message_responses:
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        for response in message_responses:
            expected_intent = response.get('expected_intent')
            conversation_id = response.get('conversation_id')
            user_message = response.get('user_message')
            
            if expected_intent and conversation_id and user_message:
                # Get actual intent classification
                intent_debug = self.debug_intent_classification(user_message, conversation_id)
                
                if 'error' not in intent_debug:
                    actual_intent = intent_debug.get('intent')
                    
                    if actual_intent == expected_intent:
                        correct_predictions += 1
                    
                    total_predictions += 1
                    
                    print(f"  Intent check: Expected={expected_intent}, Actual={actual_intent}, Match={actual_intent == expected_intent}")
        
        accuracy = correct_predictions / max(total_predictions, 1)
        print(f"📊 Intent Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
        
        return accuracy
    
    def calculate_context_quality(self, conversation_data: Dict, message_responses: List[Dict]) -> float:
        """Calculate context quality based on conversation coherence and thread management"""
        
        if not conversation_data or not message_responses:
            return 0.0
        
        quality_score = 0.0
        factors = []
        
        # Factor 1: Thread detection quality (30% weight)
        threads = conversation_data.get("threads", [])
        if threads:
            avg_thread_confidence = sum(t.get("confidence", 0) for t in threads) / len(threads)
            thread_score = min(avg_thread_confidence, 1.0)
            factors.append(("Thread Detection", thread_score, 0.3))
        else:
            factors.append(("Thread Detection", 0.0, 0.3))
        
        # Factor 2: Response consistency (30% weight)
        response_times = [r.get('response_time', 0) for r in message_responses]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            # Good response time is under 3 seconds
            response_score = max(0, min(1.0, 1.0 - (avg_response_time - 1.0) / 3.0))
            factors.append(("Response Time", response_score, 0.3))
        else:
            factors.append(("Response Time", 0.0, 0.3))
        
        # Factor 3: Conversation length management (20% weight)
        messages = conversation_data.get("messages", [])
        if messages:
            # Longer conversations with maintained quality are better
            length_score = min(1.0, len(messages) / 20.0)  # Optimal around 20 messages
            factors.append(("Length Management", length_score, 0.2))
        else:
            factors.append(("Length Management", 0.0, 0.2))
        
        # Factor 4: Error rate (20% weight)
        total_turns = len(message_responses)
        successful_turns = sum(1 for r in message_responses if r.get('assistant_response'))
        error_rate = 1.0 - (successful_turns / max(total_turns, 1))
        error_score = 1.0 - error_rate
        factors.append(("Error Rate", error_score, 0.2))
        
        # Calculate weighted score
        for factor_name, score, weight in factors:
            quality_score += score * weight
            print(f"  {factor_name}: {score:.2%} (weight: {weight:.0%})")
        
        print(f"📊 Context Quality: {quality_score:.2%}")
        return quality_score
    
    def calculate_memory_effectiveness(self, conversation_data: Dict, test_scenario: Dict) -> float:
        """Calculate memory effectiveness based on context retention"""
        
        # Simple heuristic based on conversation length and thread management
        messages = conversation_data.get("messages", [])
        threads = conversation_data.get("threads", [])
        
        if not messages:
            return 0.0
        
        # Base score from conversation length
        length_factor = min(1.0, len(messages) / 15.0)  # Optimal around 15 messages
        
        # Thread management factor
        thread_factor = len(threads) / max(len(messages) // 5, 1)  # Expect 1 thread per 5 messages
        thread_factor = min(1.0, thread_factor)
        
        # Expected behaviors factor
        expected_behaviors = test_scenario.get("expected_behaviors", [])
        behavior_score = 1.0 if len(expected_behaviors) > 0 else 0.5
        
        memory_score = (length_factor * 0.4 + thread_factor * 0.4 + behavior_score * 0.2)
        
        print(f"📊 Memory Effectiveness: {memory_score:.2%}")
        return memory_score
    
    def analyze_conversation_quality(self, conversation_data: Dict, message_responses: List[Dict], test_context: Dict) -> Dict:
        """Enhanced analysis of conversation quality with proper metric calculation"""
        
        print(f"\n🔍 Analyzing conversation quality for {len(message_responses)} turns...")
        
        # Calculate intent accuracy
        intent_accuracy = self.calculate_intent_accuracy(message_responses)
        
        # Calculate context quality
        context_quality = self.calculate_context_quality(conversation_data, message_responses)
        
        # Calculate memory effectiveness
        memory_effectiveness = self.calculate_memory_effectiveness(conversation_data, test_context)
        
        # Calculate overall response relevance
        response_relevance = (intent_accuracy + context_quality) / 2.0
        
        # Calculate overall score
        overall_score = (intent_accuracy * 0.3 + context_quality * 0.3 + memory_effectiveness * 0.2 + response_relevance * 0.2)
        
        analysis = {
            "conversation_id": conversation_data.get("conversation_id"),
            "total_messages": len(conversation_data.get("messages", [])),
            "threads_detected": len(conversation_data.get("threads", [])),
            "active_threads": conversation_data.get("thread_summary", {}).get("active_threads", 0),
            "context_quality_score": context_quality,
            "intent_accuracy_score": intent_accuracy,
            "memory_effectiveness": memory_effectiveness,
            "response_relevance": response_relevance,
            "overall_score": overall_score,
            "detailed_analysis": {
                "intent_accuracy_details": f"Calculated from {len(message_responses)} expected vs actual intent comparisons",
                "context_quality_details": "Based on thread detection, response time, length management, and error rate",
                "memory_effectiveness_details": "Based on conversation length, thread management, and expected behaviors"
            }
        }
        
        # Thread-specific analysis
        threads = conversation_data.get("threads", [])
        if threads:
            avg_confidence = sum(t.get("confidence", 0) for t in threads) / len(threads)
            analysis["thread_confidence_avg"] = avg_confidence
            analysis["topics_identified"] = [t.get("topic", "") for t in threads]
        
        # Message ratio analysis
        messages = conversation_data.get("messages", [])
        if messages:
            user_messages = [m for m in messages if m.get("role") == "user"]
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            
            analysis["user_messages"] = len(user_messages)
            analysis["assistant_messages"] = len(assistant_messages)
            analysis["message_ratio"] = len(assistant_messages) / max(len(user_messages), 1)
        
        # Test-specific context
        analysis["test_type"] = test_context.get("test_type", "unknown")
        analysis["expected_behaviors"] = test_context.get("expected_behaviors", [])
        analysis["complexity_level"] = test_context.get("complexity_level", "unknown")
        
        return analysis
    
    def run_conversation_test(self, test_scenario: Dict) -> Dict:
        """Run a complete conversation test scenario with enhanced analysis"""
        
        print(f"\n🚀 Starting Test: {test_scenario['name']}")
        print(f"📝 Description: {test_scenario['description']}")
        print(f"⚡ Complexity: {test_scenario['complexity_level']}")
        print("=" * 60)
        
        # Generate conversation ID for this test
        conversation_id = self.generate_test_conversation_id(test_scenario['name'])
        self.current_conversation_id = conversation_id
        
        # Test execution data
        test_start_time = time.time()
        message_responses = []
        errors = []
        
        # Execute conversation turns
        for i, turn in enumerate(test_scenario['conversation_turns'], 1):
            print(f"\n💬 Turn {i}: {turn['user_message'][:60]}...")
            
            # Send message
            start_time = time.time()
            response = self.send_message(turn['user_message'], conversation_id)
            response_time = time.time() - start_time
            
            if "error" in response:
                errors.append({
                    "turn": i,
                    "error": response["error"],
                    "user_message": turn['user_message']
                })
                print(f"❌ Error in turn {i}: {response['error']}")
                continue
            
            # Store response data
            message_responses.append({
                "turn": i,
                "user_message": turn['user_message'],
                "assistant_response": response.get('message', ''),
                "tokens_used": response.get('tokens_used', 0),
                "response_time": response_time,
                "conversation_id": response.get('conversation_id'),
                "total_messages": response.get('total_messages', 0),
                "expected_intent": turn.get('expected_intent'),
                "expected_topics": turn.get('expected_topics', [])
            })
            
            print(f"✅ Response ({response_time:.2f}s): {response.get('message', '')[:80]}...")
            
            # Brief delay between messages
            if i < len(test_scenario['conversation_turns']):
                time.sleep(self.delay_between_messages)
        
        test_duration = time.time() - test_start_time
        
        # Get final conversation analysis
        print(f"\n🔍 Analyzing conversation quality...")
        conversation_details = self.get_conversation_details(conversation_id)
        
        # Enhanced conversation quality analysis
        quality_analysis = self.analyze_conversation_quality(
            conversation_details, 
            message_responses,
            test_scenario
        )
        
        # Compile test results
        test_results = {
            "test_name": test_scenario['name'],
            "test_type": test_scenario['test_type'],
            "complexity_level": test_scenario['complexity_level'],
            "conversation_id": conversation_id,
            "execution_summary": {
                "total_turns": len(test_scenario['conversation_turns']),
                "successful_turns": len(message_responses),
                "errors": len(errors),
                "total_duration": test_duration,
                "avg_response_time": sum(r['response_time'] for r in message_responses) / max(len(message_responses), 1)
            },
            "message_responses": message_responses,
            "errors": errors,
            "conversation_analysis": quality_analysis,
            "conversation_details": conversation_details,
            "test_timestamp": datetime.now().isoformat(),
            "success": len(errors) == 0
        }
        
        # Store results
        self.test_results[test_scenario['name']] = test_results
        
        print(f"\n✅ Test Complete: {test_scenario['name']}")
        print(f"📊 Results: {len(message_responses)}/{len(test_scenario['conversation_turns'])} successful turns")
        print(f"⏱️  Duration: {test_duration:.2f}s")
        print(f"🧵 Threads: {quality_analysis.get('threads_detected', 0)} detected")
        print(f"🎯 Intent Accuracy: {quality_analysis.get('intent_accuracy_score', 0):.1%}")
        print(f"📋 Context Quality: {quality_analysis.get('context_quality_score', 0):.1%}")
        print(f"🧠 Memory Effectiveness: {quality_analysis.get('memory_effectiveness', 0):.1%}")
        print(f"🎯 Overall Score: {quality_analysis.get('overall_score', 0):.1%}")
        
        return test_results
    
    def save_test_results(self, output_file: str = None):
        """Save test results to JSON file"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"enhanced_conversation_test_results_{timestamp}.json"
        
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            output_file
        )
        
        # Prepare summary
        summary = {
            "test_suite_summary": {
                "total_tests": len(self.test_results),
                "successful_tests": sum(1 for r in self.test_results.values() if r['success']),
                "total_conversation_turns": sum(r['execution_summary']['total_turns'] for r in self.test_results.values()),
                "total_duration": sum(r['execution_summary']['total_duration'] for r in self.test_results.values()),
                "avg_intent_accuracy": sum(r['conversation_analysis'].get('intent_accuracy_score', 0) for r in self.test_results.values()) / max(len(self.test_results), 1),
                "avg_context_quality": sum(r['conversation_analysis'].get('context_quality_score', 0) for r in self.test_results.values()) / max(len(self.test_results), 1),
                "avg_memory_effectiveness": sum(r['conversation_analysis'].get('memory_effectiveness', 0) for r in self.test_results.values()) / max(len(self.test_results), 1),
                "avg_overall_score": sum(r['conversation_analysis'].get('overall_score', 0) for r in self.test_results.values()) / max(len(self.test_results), 1),
                "test_timestamp": datetime.now().isoformat()
            },
            "individual_test_results": self.test_results
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Enhanced test results saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None
    
    def run_all_tests(self, test_scenarios: List[Dict]) -> Dict:
        """Run all test scenarios and compile comprehensive results"""
        
        print("\n" + "=" * 80)
        print("🧪 STARTING ENHANCED CONVERSATION ROBUSTNESS TESTING")
        print("=" * 80)
        
        suite_start_time = time.time()
        
        for test_scenario in test_scenarios:
            try:
                self.run_conversation_test(test_scenario)
            except Exception as e:
                print(f"❌ Test '{test_scenario['name']}' failed with error: {e}")
                # Continue with other tests
                continue
        
        suite_duration = time.time() - suite_start_time
        
        # Generate comprehensive summary
        print(f"\n" + "=" * 80)
        print("📊 ENHANCED TEST SUITE SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Successful Tests: {sum(1 for r in self.test_results.values() if r['success'])}")
        print(f"Total Duration: {suite_duration:.2f} seconds")
        print(f"Total Conversation Turns: {sum(r['execution_summary']['total_turns'] for r in self.test_results.values())}")
        
        if self.test_results:
            avg_intent_accuracy = sum(r['conversation_analysis'].get('intent_accuracy_score', 0) for r in self.test_results.values()) / len(self.test_results)
            avg_context_quality = sum(r['conversation_analysis'].get('context_quality_score', 0) for r in self.test_results.values()) / len(self.test_results)
            avg_memory_effectiveness = sum(r['conversation_analysis'].get('memory_effectiveness', 0) for r in self.test_results.values()) / len(self.test_results)
            avg_overall_score = sum(r['conversation_analysis'].get('overall_score', 0) for r in self.test_results.values()) / len(self.test_results)
            
            print(f"Average Intent Accuracy: {avg_intent_accuracy:.1%}")
            print(f"Average Context Quality: {avg_context_quality:.1%}")
            print(f"Average Memory Effectiveness: {avg_memory_effectiveness:.1%}")
            print(f"Average Overall Score: {avg_overall_score:.1%}")
        
        # Save results
        results_file = self.save_test_results()
        
        return {
            "suite_summary": {
                "total_tests": len(self.test_results),
                "successful_tests": sum(1 for r in self.test_results.values() if r['success']),
                "suite_duration": suite_duration,
                "results_file": results_file
            },
            "test_results": self.test_results
        }

if __name__ == "__main__":
    print("Enhanced Conversation Robustness Test Runner")
    print("This runner provides proper intent accuracy and context quality calculation")
    print("Import this module and use EnhancedConversationTester class")