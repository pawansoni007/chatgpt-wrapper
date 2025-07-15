#!/usr/bin/env python3
"""
Basic Level Conversation Tests (5-15 turns)
==========================================

Tests fundamental conversation abilities:
- Simple context retention
- Basic intent classification
- Short-term memory
- Response coherence
- Basic topic tracking

These tests validate that the system can handle simple conversations
similar to what a user might have in the first few minutes of interaction.
"""

import os
import sys
from typing import List, Dict

# Add the test runner to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from main_conversation_test_runner import ConversationTester

def create_basic_conversation_scenarios() -> List[Dict]:
    """Create basic level conversation test scenarios"""
    
    scenarios = []
    
    # Test 1: Simple Q&A Chain (Context Retention)
    scenarios.append({
        "name": "basic_qa_chain",
        "description": "Simple question-answer chain testing basic context retention",
        "test_type": "basic",
        "complexity_level": "basic",
        "expected_behaviors": [
            "Remember previous answers",
            "Maintain context across 5-8 turns", 
            "Accurate intent classification",
            "Coherent responses"
        ],
        "conversation_turns": [
            {
                "user_message": "Hi! I'm working on a Python project and need some help.",
                "expected_intent": "NEW_REQUEST",
                "expected_topics": ["python", "programming", "help"]
            },
            {
                "user_message": "I want to build a web scraper to collect product prices.",
                "expected_intent": "NEW_REQUEST", 
                "expected_topics": ["web scraping", "product prices", "project"]
            },
            {
                "user_message": "Which Python library would you recommend for this?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["python libraries", "recommendation"]
            },
            {
                "user_message": "How does Beautiful Soup work exactly?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["beautiful soup", "web scraping"]
            },
            {
                "user_message": "Can you show me a simple example of scraping a website?",
                "expected_intent": "ARTIFACT_GENERATION",
                "expected_topics": ["code example", "web scraping"]
            },
            {
                "user_message": "What about handling JavaScript-rendered content?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["javascript", "dynamic content"]
            },
            {
                "user_message": "Great! Now I understand the basics. Thank you!",
                "expected_intent": "CONVERSATIONAL_FILLER",
                "expected_topics": ["acknowledgment", "thanks"]
            }
        ]
    })
    
    # Test 2: Project Planning Conversation
    scenarios.append({
        "name": "basic_project_planning",
        "description": "Basic project planning conversation with simple topic progression",
        "test_type": "basic",
        "complexity_level": "basic", 
        "expected_behaviors": [
            "Track project requirements",
            "Remember user preferences",
            "Maintain planning context",
            "Suggest logical next steps"
        ],
        "conversation_turns": [
            {
                "user_message": "I want to create a personal budget tracker app.",
                "expected_intent": "NEW_REQUEST",
                "expected_topics": ["budget tracker", "app development"]
            },
            {
                "user_message": "What technology stack would work best for this?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["technology stack", "recommendations"]
            },
            {
                "user_message": "I prefer working with React. Is that a good choice?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["react", "frontend"]
            },
            {
                "user_message": "What about the backend? I need to store user data.",
                "expected_intent": "EXPLANATION", 
                "expected_topics": ["backend", "database", "user data"]
            },
            {
                "user_message": "Should I use a SQL or NoSQL database?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["database", "sql", "nosql"]
            },
            {
                "user_message": "Let's go with PostgreSQL. What are the main features I should implement first?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["postgresql", "features", "mvp"]
            },
            {
                "user_message": "Perfect! I think I have a clear plan now.",
                "expected_intent": "CONVERSATIONAL_FILLER",
                "expected_topics": ["completion", "satisfaction"]
            }
        ]
    })
    
    # Test 3: Problem-Solution Discussion
    scenarios.append({
        "name": "basic_problem_solving",
        "description": "Simple problem-solving conversation with context building",
        "test_type": "basic",
        "complexity_level": "basic",
        "expected_behaviors": [
            "Understand problem context",
            "Build on previous information",
            "Provide relevant solutions",
            "Remember problem details"
        ],
        "conversation_turns": [
            {
                "user_message": "My website is loading very slowly and I'm not sure why.",
                "expected_intent": "DEBUGGING",
                "expected_topics": ["website performance", "slow loading"]
            },
            {
                "user_message": "It's a React app with about 50 components and several large images.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["react app", "components", "images"]
            },
            {
                "user_message": "What could be causing the slow loading times?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["performance issues", "diagnosis"]
            },
            {
                "user_message": "How can I optimize the images?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["image optimization", "performance"]
            },
            {
                "user_message": "What about code splitting? Would that help?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["code splitting", "react optimization"]
            },
            {
                "user_message": "Should I implement lazy loading for the components?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["lazy loading", "components"]
            }
        ]
    })
    
    # Test 4: Learning Conversation  
    scenarios.append({
        "name": "basic_learning_session",
        "description": "Educational conversation testing concept explanation and follow-up",
        "test_type": "basic",
        "complexity_level": "basic",
        "expected_behaviors": [
            "Explain concepts clearly",
            "Build on previous explanations", 
            "Remember learning context",
            "Adapt to user understanding level"
        ],
        "conversation_turns": [
            {
                "user_message": "I'm new to machine learning. Can you explain what it is?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["machine learning", "basics", "introduction"]
            },
            {
                "user_message": "What's the difference between supervised and unsupervised learning?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["supervised learning", "unsupervised learning"]
            },
            {
                "user_message": "Can you give me examples of supervised learning?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["supervised learning", "examples"]
            },
            {
                "user_message": "What about neural networks? How do they fit in?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["neural networks", "machine learning"]
            },
            {
                "user_message": "This is a lot to take in. What should I learn first?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["learning path", "beginner advice"]
            },
            {
                "user_message": "Are there any good online courses you'd recommend?",
                "expected_intent": "EXPLANATION", 
                "expected_topics": ["online courses", "recommendations"]
            }
        ]
    })
    
    # Test 5: Quick Troubleshooting
    scenarios.append({
        "name": "basic_troubleshooting",
        "description": "Simple troubleshooting session with iterative problem solving",
        "test_type": "basic",
        "complexity_level": "basic",
        "expected_behaviors": [
            "Systematic problem diagnosis",
            "Remember previous attempts",
            "Build troubleshooting context",
            "Provide clear next steps"
        ],
        "conversation_turns": [
            {
                "user_message": "My Python script keeps giving me a 'ModuleNotFoundError' for requests.",
                "expected_intent": "DEBUGGING",
                "expected_topics": ["python error", "module not found", "requests"]
            },
            {
                "user_message": "I thought I installed it already with pip install requests.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["pip install", "installation"]
            },
            {
                "user_message": "How can I check if it's actually installed?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["package verification", "python packages"]
            },
            {
                "user_message": "It says requests 2.28.1 is installed, but I'm still getting the error.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["version conflict", "still error"]
            },
            {
                "user_message": "I'm using a virtual environment. Could that be related?",
                "expected_intent": "DEBUGGING",
                "expected_topics": ["virtual environment", "python environment"]
            },
            {
                "user_message": "How do I check which Python environment my script is using?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["python environment", "debugging"]
            },
            {
                "user_message": "That fixed it! The script was running in the wrong environment.",
                "expected_intent": "CONVERSATIONAL_FILLER",
                "expected_topics": ["resolution", "thanks"]
            }
        ]
    })
    
    return scenarios

def run_basic_level_tests():
    """Run all basic level conversation tests"""
    
    print("\n" + "🔵" * 20)
    print("🔵 BASIC LEVEL CONVERSATION TESTS")
    print("🔵" * 20)
    print("\nTesting fundamental conversation abilities:")
    print("- Context retention over 5-15 turns")
    print("- Basic intent classification")
    print("- Simple topic tracking")
    print("- Response coherence")
    
    # Create test scenarios
    scenarios = create_basic_conversation_scenarios()
    
    # Initialize tester
    tester = ConversationTester()
    
    # Check if server is running
    try:
        response = tester.session.get(f"{tester.base_url}/")
        print(f"✅ Connected to chat server: {response.json().get('message', 'OK')}")
    except Exception as e:
        print(f"❌ Cannot connect to chat server at {tester.base_url}")
        print("Please ensure the server is running with: python src/main.py")
        return None
    
    # Run tests
    results = tester.run_all_tests(scenarios)
    
    return results

if __name__ == "__main__":
    results = run_basic_level_tests()
    
    if results:
        print(f"\n✅ Basic level tests completed!")
        print(f"Results saved to: {results['suite_summary']['results_file']}")
    else:
        print(f"\n❌ Basic level tests failed!")
