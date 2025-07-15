#!/usr/bin/env python3
"""
Tough Level Conversation Tests (50-100+ turns)
==============================================

Tests advanced conversation abilities:
- Long conversation context management
- Complex interleaved topics
- Deep context dependencies
- Advanced thread management
- Complex intent patterns
- Memory-intensive scenarios
- Topic resurrection and complex branching

These tests validate that the system can handle very complex
conversations similar to extensive work sessions or deep
collaborative discussions that span multiple hours/sessions.
"""

import os
import sys
from typing import List, Dict

# Add the test runner to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from main_conversation_test_runner import ConversationTester

def create_tough_conversation_scenarios() -> List[Dict]:
    """Create tough level conversation test scenarios"""
    
    scenarios = []
    
    # Test 1: Complex Software Architecture Discussion
    scenarios.append({
        "name": "tough_complex_architecture_session",
        "description": "Extensive software architecture discussion with multiple interleaved concerns",
        "test_type": "tough",
        "complexity_level": "tough",
        "expected_behaviors": [
            "Manage 5+ concurrent topics",
            "Maintain context across 50+ turns",
            "Handle complex topic switching",
            "Remember decisions made earlier",
            "Connect disparate architectural concepts",
            "Provide consistent technical advice"
        ],
        "conversation_turns": [
            # Initial system overview
            {
                "user_message": "I'm designing a microservices architecture for a large-scale social media platform. We expect millions of users and need to handle posts, comments, likes, follows, messaging, and real-time notifications.",
                "expected_intent": "NEW_REQUEST",
                "expected_topics": ["microservices", "social media", "scalability", "architecture"]
            },
            {
                "user_message": "Let's start with the user service. What data should it handle and how should we structure it?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["user service", "data modeling", "microservices"]
            },
            {
                "user_message": "Should user authentication be part of the user service or separate?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["authentication", "service separation", "microservices"]
            },
            {
                "user_message": "For the database, should each microservice have its own database?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["database per service", "data consistency"]
            },
            
            # Branch to post service
            {
                "user_message": "Now for the post service - how do we handle the relationship between users and posts across services?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["post service", "cross-service relationships", "data consistency"]
            },
            {
                "user_message": "What about handling media uploads like images and videos in posts?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["media upload", "file storage", "cdn"]
            },
            {
                "user_message": "Should we process images asynchronously for different sizes?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["async processing", "image processing", "thumbnails"]
            },
            {
                "user_message": "How do we ensure posts appear in the correct order in user feeds?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["feed generation", "post ordering", "algorithms"]
            },
            
            # Introduce messaging complexity
            {
                "user_message": "The messaging system needs to be real-time. Should it be a separate service?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["messaging service", "real-time", "websockets"]
            },
            {
                "user_message": "How do we handle message encryption and privacy?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["message encryption", "privacy", "security"]
            },
            {
                "user_message": "What about group chats vs direct messages? Same service or different?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["group chat", "direct messages", "service design"]
            },
            
            # Notification complexity
            {
                "user_message": "For notifications, we need to notify users about likes, comments, follows, and messages. How do we coordinate this across services?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["notifications", "event-driven", "service coordination"]
            },
            {
                "user_message": "Should we use an event bus or message queue for inter-service communication?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["event bus", "message queue", "service communication"]
            },
            {
                "user_message": "How do we handle notification preferences and delivery channels?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["notification preferences", "delivery channels", "push notifications"]
            },
            
            # Performance and scaling concerns
            {
                "user_message": "Back to the user service - how do we cache user data for performance?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["caching", "user data", "performance", "redis"]
            },
            {
                "user_message": "What about caching for the post feed? That seems like a major performance bottleneck.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["feed caching", "performance optimization", "bottlenecks"]
            },
            {
                "user_message": "Should we pre-compute feeds or generate them on-demand?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["feed pre-computation", "on-demand generation", "trade-offs"]
            },
            {
                "user_message": "How do we handle the follow/unfollow actions affecting multiple users' feeds?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["follow relationships", "feed updates", "event propagation"]
            },
            
            # Database considerations
            {
                "user_message": "For the post service database, should we use SQL or NoSQL given the scale?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["sql vs nosql", "post storage", "scalability"]
            },
            {
                "user_message": "How do we handle database sharding for the user service?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["database sharding", "user data", "distribution"]
            },
            {
                "user_message": "What about cross-shard queries when users follow each other across shards?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["cross-shard queries", "data consistency", "follow relationships"]
            },
            
            # Security deep dive
            {
                "user_message": "Let's discuss security. How do we handle authentication tokens across all these services?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["authentication tokens", "service security", "jwt"]
            },
            {
                "user_message": "Should each service validate tokens independently or use a central auth service?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["token validation", "central auth", "distributed auth"]
            },
            {
                "user_message": "How do we protect against common attacks like SQL injection and XSS?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["security vulnerabilities", "sql injection", "xss"]
            },
            {
                "user_message": "What about rate limiting to prevent abuse?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["rate limiting", "abuse prevention", "api protection"]
            },
            
            # Monitoring and observability
            {
                "user_message": "How do we monitor the health of all these microservices?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["monitoring", "microservices health", "observability"]
            },
            {
                "user_message": "What kind of logging strategy should we implement across services?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["logging strategy", "distributed logging", "observability"]
            },
            {
                "user_message": "How do we trace requests that span multiple services?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["distributed tracing", "request tracing", "observability"]
            },
            
            # Deployment and DevOps
            {
                "user_message": "For deployment, should we use Kubernetes for orchestration?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["kubernetes", "container orchestration", "deployment"]
            },
            {
                "user_message": "How do we handle rolling updates and rollbacks for critical services like the user service?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["rolling updates", "rollbacks", "deployment strategy"]
            },
            {
                "user_message": "What about blue-green deployments vs canary deployments?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["blue-green deployment", "canary deployment", "deployment strategies"]
            },
            
            # Data consistency challenges
            {
                "user_message": "Going back to data consistency - how do we handle eventual consistency between the user service and post service?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["eventual consistency", "data consistency", "distributed systems"]
            },
            {
                "user_message": "What happens if a user is deleted but their posts still exist in the post service?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["data consistency", "cascade deletes", "orphaned data"]
            },
            {
                "user_message": "Should we implement saga patterns for complex transactions?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["saga pattern", "distributed transactions", "consistency"]
            },
            
            # Performance optimization callbacks
            {
                "user_message": "Earlier we discussed feed caching. How does that interact with the notification system when new posts are created?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["feed caching", "notifications", "cache invalidation"]
            },
            {
                "user_message": "For the media uploads we talked about - how do we optimize CDN performance globally?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["cdn optimization", "global performance", "media delivery"]
            },
            
            # Complex integration scenarios
            {
                "user_message": "How do we handle a scenario where a viral post causes a sudden spike in likes, comments, and notifications?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["viral content", "traffic spikes", "system resilience"]
            },
            {
                "user_message": "What if the notification service goes down but users are still creating posts and interactions?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["service failure", "resilience", "message queues"]
            },
            
            # API gateway considerations
            {
                "user_message": "Should we use an API gateway to expose these services to clients?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["api gateway", "service exposure", "client interface"]
            },
            {
                "user_message": "How does the API gateway handle authentication and routing to the right services?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["api gateway", "authentication", "service routing"]
            },
            
            # Final integration and summary
            {
                "user_message": "Considering everything we've discussed - the user service, posts, messaging, notifications, caching, security, and deployment - what are the biggest risks in this architecture?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["architecture risks", "system vulnerabilities", "trade-offs"]
            },
            {
                "user_message": "How do we prioritize the implementation? What should we build first?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["implementation priority", "mvp", "development roadmap"]
            },
            {
                "user_message": "Can you create a comprehensive technical summary of our entire architecture discussion?",
                "expected_intent": "ARTIFACT_GENERATION",
                "expected_topics": ["architecture summary", "technical documentation", "comprehensive overview"]
            }
        ]
    })
    
    return scenarios

def run_tough_level_tests():
    """Run all tough level conversation tests"""
    
    print("\n" + "🔴" * 20)
    print("🔴 TOUGH LEVEL CONVERSATION TESTS")
    print("🔴" * 20)
    print("\nTesting advanced conversation abilities:")
    print("- Complex multi-topic conversations (50+ turns)")
    print("- Deep context dependencies")
    print("- Advanced thread management")
    print("- Memory-intensive scenarios")
    print("- Complex topic interleaving and resurrection")
    
    # Create test scenarios
    scenarios = create_tough_conversation_scenarios()
    
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
    results = run_tough_level_tests()
    
    if results:
        print(f"\n✅ Tough level tests completed!")
        print(f"Results saved to: {results['suite_summary']['results_file']}")
    else:
        print(f"\n❌ Tough level tests failed!")
