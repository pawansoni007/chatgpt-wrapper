#!/usr/bin/env python3
"""
Enhanced Medium Level Conversation Tests
======================================

Uses the EnhancedConversationTester to properly calculate intent accuracy
and context quality metrics for medium-complexity conversations.
"""

import os
import sys
from typing import List, Dict

# Add the test runner to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from enhanced_test_runner import EnhancedConversationTester

def create_medium_conversation_scenarios() -> List[Dict]:
    """Create medium level conversation test scenarios"""
    
    scenarios = []
    
    # Test 1: Multi-Topic Development Discussion
    scenarios.append({
        "name": "medium_multi_topic_development", 
        "description": "Development discussion covering frontend, backend, database, and deployment",
        "test_type": "medium",
        "complexity_level": "medium",
        "expected_behaviors": [
            "Manage multiple related topics",
            "Switch between topics while retaining context",
            "Connect related concepts across topics",
            "Maintain project context throughout"
        ],
        "conversation_turns": [
            # Initial project setup
            {
                "user_message": "I'm building a full-stack e-commerce platform and need architectural guidance.",
                "expected_intent": "NEW_REQUEST",
                "expected_topics": ["e-commerce", "full-stack", "architecture"]
            },
            {
                "user_message": "For the frontend, I'm thinking React with TypeScript. What do you think?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["frontend", "react", "typescript"]
            },
            {
                "user_message": "What state management would you recommend for an e-commerce app?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["state management", "react", "e-commerce"]
            },
            {
                "user_message": "Redux seems complex. Is there a simpler alternative?",
                "expected_intent": "COMPARISON", 
                "expected_topics": ["redux", "state management", "alternatives"]
            },
            
            # Topic switch to backend
            {
                "user_message": "Now for the backend - I need to handle user authentication, product catalog, and orders.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["backend", "authentication", "orders", "catalog"]
            },
            {
                "user_message": "Should I use Node.js with Express or try something like FastAPI with Python?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["node.js", "express", "fastapi", "python"]
            },
            {
                "user_message": "For authentication, what's the best approach? JWT tokens?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["authentication", "jwt", "security"]
            },
            {
                "user_message": "How do I handle password hashing securely?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["password hashing", "security"]
            },
            
            # Topic switch to database
            {
                "user_message": "What about the database design? I need to store users, products, orders, and reviews.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["database design", "schema", "e-commerce"]
            },
            {
                "user_message": "Should I use PostgreSQL or MongoDB for this type of application?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["postgresql", "mongodb", "database choice"]
            },
            {
                "user_message": "How would you structure the product catalog tables?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["database schema", "product catalog"]
            },
            {
                "user_message": "What about handling product variants like size and color?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["product variants", "database design"]
            },
            
            # Topic switch to deployment/DevOps
            {
                "user_message": "For deployment, I'm considering AWS. What services should I use?",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["aws", "deployment", "cloud services"]
            },
            {
                "user_message": "Should I containerize the application with Docker?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["docker", "containerization"]
            },
            {
                "user_message": "What about CI/CD pipeline? I want automatic deployments.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["ci/cd", "automation", "deployment"]
            },
            
            # Integration questions that span multiple topics
            {
                "user_message": "How do I handle image uploads for products efficiently?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["image upload", "products", "performance"]
            },
            {
                "user_message": "Should the frontend communicate directly with the database or through the API?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["architecture", "frontend", "backend", "api"]
            },
            {
                "user_message": "What's the best way to handle real-time inventory updates across the system?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["real-time", "inventory", "architecture"]
            },
            
            # Performance and scaling
            {
                "user_message": "How do I optimize the app for performance as it scales?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["performance", "scaling", "optimization"]
            },
            {
                "user_message": "Should I implement caching? Where and how?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["caching", "performance", "redis"]
            },
            
            # Final integration question
            {
                "user_message": "Can you summarize the complete tech stack we've discussed?",
                "expected_intent": "ARTIFACT_GENERATION",
                "expected_topics": ["summary", "tech stack", "architecture"]
            }
        ]
    })
    
    # Test 2: Learning Journey with Topic Branching
    scenarios.append({
        "name": "medium_learning_journey",
        "description": "Educational conversation that branches into multiple related concepts",
        "test_type": "medium", 
        "complexity_level": "medium",
        "expected_behaviors": [
            "Track learning progression",
            "Connect related concepts",
            "Return to previous topics with context",
            "Build comprehensive understanding"
        ],
        "conversation_turns": [
            # Start with machine learning
            {
                "user_message": "I want to learn about artificial intelligence and machine learning from scratch.",
                "expected_intent": "NEW_REQUEST",
                "expected_topics": ["ai", "machine learning", "learning"]
            },
            {
                "user_message": "What's the difference between AI, ML, and deep learning?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["ai", "machine learning", "deep learning"]
            },
            {
                "user_message": "Let's focus on machine learning first. What are the main types?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["machine learning types", "supervised", "unsupervised"]
            },
            
            # Branch into supervised learning
            {
                "user_message": "Can you explain supervised learning in detail?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["supervised learning", "classification", "regression"]
            },
            {
                "user_message": "What's the difference between classification and regression?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["classification", "regression"]
            },
            {
                "user_message": "Can you give me examples of classification algorithms?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["classification algorithms", "examples"]
            },
            {
                "user_message": "How does a decision tree work exactly?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["decision tree", "algorithm"]
            },
            
            # Branch to math fundamentals
            {
                "user_message": "What math background do I need for machine learning?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["mathematics", "prerequisites", "linear algebra"]
            },
            {
                "user_message": "I'm rusty on linear algebra. What concepts are most important?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["linear algebra", "vectors", "matrices"]
            },
            {
                "user_message": "How are matrices used in machine learning specifically?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["matrices", "machine learning", "data representation"]
            },
            {
                "user_message": "What about statistics? Which concepts should I review?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["statistics", "probability", "distributions"]
            },
            
            # Branch to practical implementation
            {
                "user_message": "Now for the practical side - which programming language should I use?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["programming languages", "python", "r"]
            },
            {
                "user_message": "What Python libraries are essential for machine learning?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["python libraries", "scikit-learn", "pandas"]
            },
            {
                "user_message": "Can you explain what each library does? Start with pandas.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["pandas", "data manipulation"]
            },
            {
                "user_message": "How does scikit-learn fit into the workflow?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["scikit-learn", "ml workflow"]
            },
            
            # Connect back to earlier topics
            {
                "user_message": "Going back to decision trees - how would I implement one in scikit-learn?",
                "expected_intent": "ARTIFACT_GENERATION",
                "expected_topics": ["decision tree", "scikit-learn", "implementation"]
            },
            {
                "user_message": "What about the data preprocessing we'd need before training?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["data preprocessing", "feature engineering"]
            },
            
            # Branch to deep learning
            {
                "user_message": "When should I move from traditional ML to deep learning?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["deep learning", "neural networks", "when to use"]
            },
            {
                "user_message": "What's a neural network in simple terms?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["neural networks", "basics", "architecture"]
            },
            {
                "user_message": "How does backpropagation work?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["backpropagation", "training", "gradients"]
            },
            
            # Final synthesis
            {
                "user_message": "Can you create a learning roadmap based on everything we've discussed?",
                "expected_intent": "ARTIFACT_GENERATION",
                "expected_topics": ["learning roadmap", "summary", "plan"]
            },
            {
                "user_message": "What should be my first practical project to apply these concepts?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["first project", "practical application"]
            }
        ]
    })
    
    return scenarios

def run_enhanced_medium_level_tests():
    """Run all medium level conversation tests with enhanced metrics"""
    
    print("\n" + "🟡" * 20)
    print("🟡 ENHANCED MEDIUM LEVEL CONVERSATION TESTS")
    print("🟡" * 20)
    print("\nTesting intermediate conversation abilities with proper metrics:")
    print("- Multi-topic conversation management")
    print("- Context retention over 20-40 turns")
    print("- Topic switching and bridging")
    print("- Thread detection and management")
    
    # Create test scenarios
    scenarios = create_medium_conversation_scenarios()
    
    # Initialize enhanced tester
    tester = EnhancedConversationTester()
    
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
    results = run_enhanced_medium_level_tests()
    
    if results:
        print(f"\n✅ Enhanced medium level tests completed!")
        print(f"Results saved to: {results['suite_summary']['results_file']}")
    else:
        print(f"\n❌ Enhanced medium level tests failed!")
