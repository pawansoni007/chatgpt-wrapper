#!/usr/bin/env python3
"""
Master Conversation Test Runner
==============================

Comprehensive test orchestrator for all conversation robustness tests.
This runner executes all test suites and provides detailed analysis
of the chat endpoint's conversation management capabilities.

Test Suites:
1. Basic Level (5-15 turns) - Fundamental conversation abilities
2. Medium Level (20-40 turns) - Multi-topic conversations
3. Tough Level (50+ turns) - Complex interleaved topics
4. Specialized Sessions - Specific use case adaptations

Usage:
    python run_all_conversation_tests.py [--suite SUITE_NAME] [--output OUTPUT_FILE]
    
Examples:
    python run_all_conversation_tests.py                    # Run all test suites
    python run_all_conversation_tests.py --suite basic     # Run only basic tests
    python run_all_conversation_tests.py --suite medium    # Run only medium tests
    python run_all_conversation_tests.py --suite tough     # Run only tough tests  
    python run_all_conversation_tests.py --suite specialized # Run only specialized tests
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from typing import Dict, List, Optional

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import all test modules
from basic_conversation_tests import run_basic_level_tests
from medium_conversation_tests import run_medium_level_tests  
from tough_conversation_tests import run_tough_level_tests
from specialized_conversation_tests import run_specialized_tests
from main_conversation_test_runner import ConversationTester

class MasterTestRunner:
    """Master orchestrator for all conversation robustness tests"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or current_dir
        self.test_results = {}
        self.start_time = None
        self.suite_execution_times = {}
        
        # Test suite configuration
        self.available_suites = {
            'basic': {
                'name': 'Basic Level Tests',
                'description': 'Fundamental conversation abilities (5-15 turns)',
                'function': run_basic_level_tests,
                'expected_duration': '2-5 minutes'
            },
            'medium': {
                'name': 'Medium Level Tests', 
                'description': 'Multi-topic conversations (20-40 turns)',
                'function': run_medium_level_tests,
                'expected_duration': '5-10 minutes'
            },
            'tough': {
                'name': 'Tough Level Tests',
                'description': 'Complex interleaved topics (50+ turns)',
                'function': run_tough_level_tests,
                'expected_duration': '10-20 minutes'
            },
            'specialized': {
                'name': 'Specialized Tests',
                'description': 'Specific use case adaptations',
                'function': run_specialized_tests,
                'expected_duration': '8-15 minutes'
            }
        }
    
    def check_server_status(self) -> bool:
        """Check if the chat server is running and responsive"""
        
        print("🔍 Checking chat server status...")
        
        try:
            tester = ConversationTester()
            response = tester.session.get(f"{tester.base_url}/", timeout=10)
            server_info = response.json()
            
            print(f"✅ Server is running: {server_info.get('message', 'OK')}")
            print(f"📋 Version: {server_info.get('version', 'Unknown')}")
            print(f"🔧 Features: {', '.join(server_info.get('features', []))}")
            
            # Test a simple chat to ensure full functionality
            test_response = tester.send_message("Hello, this is a connection test.")
            if "error" in test_response:
                print(f"⚠️ Server responding but chat endpoint has issues: {test_response['error']}")
                return False
                
            print("✅ Chat endpoint is functional")
            return True
            
        except Exception as e:
            print(f"❌ Cannot connect to chat server: {e}")
            print("Please ensure the server is running with: python src/main.py")
            return False
    
    def run_test_suite(self, suite_name: str) -> Optional[Dict]:
        """Run a specific test suite"""
        
        if suite_name not in self.available_suites:
            print(f"❌ Unknown test suite: {suite_name}")
            print(f"Available suites: {', '.join(self.available_suites.keys())}")
            return None
        
        suite_config = self.available_suites[suite_name]
        
        print(f"\n{'🚀' * 60}")
        print(f"🚀 STARTING: {suite_config['name']}")
        print(f"🚀 {suite_config['description']}")
        print(f"🚀 Expected Duration: {suite_config['expected_duration']}")
        print(f"{'🚀' * 60}")
        
        suite_start_time = time.time()
        
        try:
            # Run the test suite
            results = suite_config['function']()
            suite_duration = time.time() - suite_start_time
            
            self.suite_execution_times[suite_name] = suite_duration
            
            if results:
                print(f"\n✅ {suite_config['name']} completed successfully!")
                print(f"⏱️ Duration: {suite_duration:.2f} seconds")
                print(f"📊 Tests run: {results['suite_summary']['total_tests']}")
                print(f"✅ Successful: {results['suite_summary']['successful_tests']}")
                
                self.test_results[suite_name] = {
                    'suite_config': suite_config,
                    'execution_time': suite_duration,
                    'results': results,
                    'success': True
                }
                
                return results
            else:
                print(f"\n❌ {suite_config['name']} failed!")
                self.test_results[suite_name] = {
                    'suite_config': suite_config,
                    'execution_time': suite_duration,
                    'results': None,
                    'success': False,
                    'error': 'Test suite returned no results'
                }
                return None
                
        except Exception as e:
            suite_duration = time.time() - suite_start_time
            print(f"\n❌ {suite_config['name']} failed with error: {e}")
            
            self.test_results[suite_name] = {
                'suite_config': suite_config,
                'execution_time': suite_duration,
                'results': None,
                'success': False,
                'error': str(e)
            }
            return None
    
    def run_all_suites(self) -> Dict:
        """Run all available test suites"""
        
        print(f"\n{'🌟' * 80}")
        print("🌟 COMPREHENSIVE CONVERSATION ROBUSTNESS TESTING")
        print("🌟 Testing ChatGPT/Claude.ai level conversation capabilities")
        print(f"🌟 Total Suites: {len(self.available_suites)}")
        print(f"🌟 Estimated Duration: 25-50 minutes")
        print(f"{'🌟' * 80}")
        
        self.start_time = time.time()
        
        # Run each test suite
        for suite_name in self.available_suites.keys():
            self.run_test_suite(suite_name)
            
            # Brief pause between suites
            if suite_name != list(self.available_suites.keys())[-1]:
                print(f"\n⏸️ Brief pause before next suite...")
                time.sleep(2)
        
        total_duration = time.time() - self.start_time
        
        # Generate comprehensive summary
        self.generate_final_summary(total_duration)
        
        return self.test_results
    
    def generate_final_summary(self, total_duration: float):
        """Generate comprehensive test summary"""
        
        print(f"\n{'📊' * 80}")
        print("📊 COMPREHENSIVE TEST SUITE SUMMARY")
        print(f"{'📊' * 80}")
        
        successful_suites = sum(1 for r in self.test_results.values() if r['success'])
        total_tests = sum(
            r['results']['suite_summary']['total_tests'] 
            for r in self.test_results.values() 
            if r['success'] and r['results']
        )
        successful_tests = sum(
            r['results']['suite_summary']['successful_tests']
            for r in self.test_results.values()
            if r['success'] and r['results']
        )
        
        print(f"🎯 Overall Results:")
        print(f"   Test Suites: {successful_suites}/{len(self.test_results)} successful")
        print(f"   Individual Tests: {successful_tests}/{total_tests} successful") 
        print(f"   Total Duration: {total_duration/60:.1f} minutes")
        
        print(f"\n📋 Suite-by-Suite Results:")
        for suite_name, results in self.test_results.items():
            status = "✅" if results['success'] else "❌"
            duration = f"{results['execution_time']:.1f}s"
            
            if results['success'] and results['results']:
                test_count = f"{results['results']['suite_summary']['successful_tests']}/{results['results']['suite_summary']['total_tests']}"
                print(f"   {status} {suite_name.capitalize()}: {test_count} tests, {duration}")
            else:
                error = results.get('error', 'Unknown error')[:50] + "..." if len(results.get('error', '')) > 50 else results.get('error', 'Unknown error')
                print(f"   {status} {suite_name.capitalize()}: Failed - {error}, {duration}")
        
        # Calculate conversation turn totals
        total_turns = 0
        for suite_results in self.test_results.values():
            if suite_results['success'] and suite_results['results']:
                for test_result in suite_results['results']['test_results'].values():
                    total_turns += test_result['execution_summary']['total_turns']
        
        print(f"\n🔢 Conversation Statistics:")
        print(f"   Total Conversation Turns: {total_turns}")
        print(f"   Average Turns per Test: {total_turns/max(total_tests, 1):.1f}")
        print(f"   Estimated Total Context Processed: {total_turns * 150} tokens") # Rough estimate
        
        # Performance insights
        print(f"\n⚡ Performance Insights:")
        print(f"   Fastest Suite: {min(self.suite_execution_times.items(), key=lambda x: x[1])[0]} ({min(self.suite_execution_times.values()):.1f}s)")
        print(f"   Slowest Suite: {max(self.suite_execution_times.items(), key=lambda x: x[1])[0]} ({max(self.suite_execution_times.values()):.1f}s)")
        
        if successful_tests == total_tests and successful_suites == len(self.test_results):
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
            print("The chat system demonstrates robust conversation management")
            print("capabilities comparable to ChatGPT/Claude.ai standards.")
        elif successful_tests >= total_tests * 0.8:
            print(f"\n✅ MOSTLY SUCCESSFUL!")
            print("The chat system shows strong conversation capabilities")
            print("with some areas for potential improvement.")
        else:
            print(f"\n⚠️ MIXED RESULTS")
            print("The chat system has basic functionality but may need")
            print("improvements for robust long-conversation management.")
        
        # Save comprehensive results
        self.save_comprehensive_results(total_duration)
    
    def save_comprehensive_results(self, total_duration: float):
        """Save comprehensive test results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"comprehensive_conversation_test_results_{timestamp}.json")
        
        comprehensive_results = {
            "test_suite_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_duration": total_duration,
                "total_suites": len(self.test_results),
                "successful_suites": sum(1 for r in self.test_results.values() if r['success']),
                "total_tests": sum(
                    r['results']['suite_summary']['total_tests'] 
                    for r in self.test_results.values() 
                    if r['success'] and r['results']
                ),
                "successful_tests": sum(
                    r['results']['suite_summary']['successful_tests']
                    for r in self.test_results.values()
                    if r['success'] and r['results']
                ),
                "suite_execution_times": self.suite_execution_times
            },
            "suite_results": self.test_results,
            "system_info": {
                "test_runner_version": "1.0.0",
                "test_categories": list(self.available_suites.keys()),
                "total_conversation_turns": sum(
                    sum(test['execution_summary']['total_turns'] for test in suite['results']['test_results'].values())
                    for suite in self.test_results.values()
                    if suite['success'] and suite['results']
                )
            }
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Comprehensive results saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error saving comprehensive results: {e}")
            return None

def main():
    """Main entry point for the master test runner"""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Conversation Robustness Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run all test suites
  %(prog)s --suite basic            # Run only basic tests
  %(prog)s --suite medium           # Run only medium tests  
  %(prog)s --suite tough            # Run only tough tests
  %(prog)s --suite specialized      # Run only specialized tests
  %(prog)s --output /path/to/dir    # Specify output directory
        """
    )
    
    parser.add_argument(
        '--suite', 
        choices=['basic', 'medium', 'tough', 'specialized'],
        help='Run specific test suite only'
    )
    
    parser.add_argument(
        '--output',
        help='Output directory for test results'
    )
    
    args = parser.parse_args()
    
    # Initialize master test runner
    runner = MasterTestRunner(output_dir=args.output)
    
    # Check server status first
    if not runner.check_server_status():
        print("❌ Cannot proceed without a working chat server")
        sys.exit(1)
    
    # Run specified tests
    if args.suite:
        print(f"Running {args.suite} test suite only...")
        results = runner.run_test_suite(args.suite)
        if results:
            print(f"✅ {args.suite} tests completed successfully!")
        else:
            print(f"❌ {args.suite} tests failed!")
            sys.exit(1)
    else:
        print("Running all test suites...")
        results = runner.run_all_suites()
        
        # Determine exit code based on results
        successful_suites = sum(1 for r in results.values() if r['success'])
        if successful_suites == len(results):
            print("🎉 All test suites completed successfully!")
            sys.exit(0)
        elif successful_suites > 0:
            print("⚠️ Some test suites had issues")
            sys.exit(1)
        else:
            print("❌ All test suites failed")
            sys.exit(2)

if __name__ == "__main__":
    main()
