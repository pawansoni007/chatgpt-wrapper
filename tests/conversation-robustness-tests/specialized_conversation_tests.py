#!/usr/bin/env python3
"""
Specialized Conversation Tests
=============================

Tests specific conversation types that require different
context management and response patterns:

1. Debugging Sessions - Technical problem-solving
2. Feature Development - Collaborative coding/planning  
3. Emotional Support - Vent/support conversations
4. Business Meetings - Professional discussions
5. Normal Chat - Casual conversations
6. Examination Sessions - Q&A format testing

These tests validate that the system can adapt its conversation
style and context management to different use cases while
maintaining consistency and relevance.
"""

import os
import sys
from typing import List, Dict

# Add the test runner to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from main_conversation_test_runner import ConversationTester

def create_debugging_session_scenario() -> Dict:
    """Create debugging session scenario"""
    
    return {
        "name": "specialized_debugging_session",
        "description": "Technical debugging session with systematic problem-solving",
        "test_type": "specialized_debugging", 
        "complexity_level": "medium",
        "expected_behaviors": [
            "Systematic debugging approach",
            "Remember previous attempts and findings",
            "Build technical context progressively",
            "Provide specific troubleshooting steps",
            "Connect related technical issues"
        ],
        "conversation_turns": [
            {
                "user_message": "I'm having a really frustrating issue with my React app. It keeps crashing with a memory leak.",
                "expected_intent": "DEBUGGING",
                "expected_topics": ["react", "memory leak", "crash"]
            },
            {
                "user_message": "The browser tab becomes unresponsive after about 5 minutes of usage.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["browser crash", "performance", "memory"]
            },
            {
                "user_message": "I checked the Chrome DevTools and the memory keeps growing. It's definitely a memory leak.",
                "expected_intent": "CONTINUATION", 
                "expected_topics": ["chrome devtools", "memory growth", "leak confirmation"]
            },
            {
                "user_message": "What are the most common causes of memory leaks in React applications?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["react memory leaks", "common causes"]
            },
            {
                "user_message": "I do have several useEffect hooks that set up event listeners. Could that be it?",
                "expected_intent": "DEBUGGING",
                "expected_topics": ["useEffect", "event listeners", "cleanup"]
            },
            {
                "user_message": "Here's one of my useEffect hooks: useEffect(() => { window.addEventListener('scroll', handleScroll); }, []). Is this wrong?",
                "expected_intent": "DEBUGGING",
                "expected_topics": ["useEffect cleanup", "event listener", "memory leak"]
            },
            {
                "user_message": "Oh! I need to clean up the event listener. How do I do that properly?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["cleanup function", "removeEventListener", "useEffect"]
            },
            {
                "user_message": "I fixed that one, but the memory leak is still happening. What else should I check?",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["still leaking", "additional causes", "debugging"]
            },
            {
                "user_message": "I'm using a state management library called Zustand. Could that cause memory leaks?",
                "expected_intent": "DEBUGGING",
                "expected_topics": ["zustand", "state management", "memory leaks"]
            },
            {
                "user_message": "I have some components that create subscriptions to the Zustand store. How do I check if they're being cleaned up?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["zustand subscriptions", "cleanup", "component unmount"]
            },
            {
                "user_message": "The subscriptions are in custom hooks. Should I be unsubscribing in the cleanup function?",
                "expected_intent": "DEBUGGING",
                "expected_topics": ["custom hooks", "subscriptions", "cleanup"]
            },
            {
                "user_message": "I think I found the issue! Some components were creating multiple subscriptions but only cleaning up one. Let me test the fix.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["found issue", "multiple subscriptions", "testing fix"]
            },
            {
                "user_message": "The memory leak is much better now, but there's still a small one. How can I identify what's still leaking?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["remaining leak", "identification", "profiling"]
            },
            {
                "user_message": "I used the Chrome DevTools memory profiler and found references to some old component instances. What does that mean?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["memory profiler", "component instances", "references"]
            },
            {
                "user_message": "Could it be related to closures capturing state from unmounted components?",
                "expected_intent": "DEBUGGING",
                "expected_topics": ["closures", "captured state", "unmounted components"]
            },
            {
                "user_message": "Perfect! I found the remaining leak. It was in a setTimeout that wasn't being cleared. Thank you for the systematic debugging approach!",
                "expected_intent": "CONVERSATIONAL_FILLER",
                "expected_topics": ["resolution", "setTimeout", "systematic debugging"]
            }
        ]
    }

def create_feature_development_scenario() -> Dict:
    """Create feature development scenario"""
    
    return {
        "name": "specialized_feature_development",
        "description": "Collaborative feature development session from planning to implementation",
        "test_type": "specialized_development",
        "complexity_level": "medium",
        "expected_behaviors": [
            "Track feature requirements and decisions",
            "Maintain technical context across implementation phases",
            "Connect design decisions to code implementation",
            "Remember architectural constraints and choices"
        ],
        "conversation_turns": [
            {
                "user_message": "I need to add a real-time chat feature to my existing web app. Let's plan this out together.",
                "expected_intent": "NEW_REQUEST",
                "expected_topics": ["real-time chat", "feature planning", "web app"]
            },
            {
                "user_message": "The app is built with React frontend and Node.js backend using REST APIs. Users should be able to send messages instantly.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["react", "nodejs", "rest api", "instant messaging"]
            },
            {
                "user_message": "What's the best way to implement real-time functionality? WebSockets or Server-Sent Events?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["websockets", "server-sent events", "real-time"]
            },
            {
                "user_message": "Let's go with WebSockets. What changes do I need to make to my existing backend?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["websockets", "backend changes", "implementation"]
            },
            {
                "user_message": "I'm using Express.js. Should I integrate Socket.IO or use native WebSocket APIs?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["socket.io", "native websockets", "express"]
            },
            {
                "user_message": "Alright, Socket.IO it is. Can you help me set up the basic server-side Socket.IO integration?",
                "expected_intent": "ARTIFACT_GENERATION",
                "expected_topics": ["socket.io setup", "server-side", "code implementation"]
            },
            {
                "user_message": "Now for the frontend - how do I connect the React app to the Socket.IO server?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["react", "socket.io client", "frontend connection"]
            },
            {
                "user_message": "Should I create a custom hook to manage the socket connection?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["custom hook", "socket management", "react patterns"]
            },
            {
                "user_message": "Great! Can you show me how to implement the custom hook for socket management?",
                "expected_intent": "ARTIFACT_GENERATION",
                "expected_topics": ["custom hook", "socket management", "react code"]
            },
            {
                "user_message": "For the UI, I want a simple chat interface with a message list and input field. Should I create separate components?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["ui design", "react components", "chat interface"]
            },
            {
                "user_message": "How do I handle message state management? Should I use React's useState or integrate with my existing Redux store?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["state management", "useState", "redux", "messages"]
            },
            {
                "user_message": "Let's use Redux for consistency. How do I handle real-time message updates in Redux?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["redux", "real-time updates", "message handling"]
            },
            {
                "user_message": "I need to handle different message types - text, images, and maybe files later. How should I structure the message data?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["message structure", "data modeling", "file types"]
            },
            {
                "user_message": "What about user authentication with the socket connection? How do I verify users?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["socket authentication", "user verification", "security"]
            },
            {
                "user_message": "I'm using JWT tokens for my REST API. Can I reuse those for socket authentication?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["jwt tokens", "socket auth", "token reuse"]
            },
            {
                "user_message": "Should I implement chat rooms or just direct messaging for now?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["chat rooms", "direct messaging", "feature scope"]
            },
            {
                "user_message": "Let's start with direct messaging. How do I route messages between specific users?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["message routing", "direct messaging", "user targeting"]
            },
            {
                "user_message": "What about persisting chat messages? Should I save them to my existing database?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["message persistence", "database", "chat history"]
            },
            {
                "user_message": "How do I load chat history when a user opens a conversation?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["chat history", "message loading", "initial data"]
            },
            {
                "user_message": "This is coming together nicely! Can you help me create a simple implementation plan with the key components we've discussed?",
                "expected_intent": "ARTIFACT_GENERATION",
                "expected_topics": ["implementation plan", "components", "development roadmap"]
            }
        ]
    }

def create_emotional_support_scenario() -> Dict:
    """Create emotional support/vent session scenario"""
    
    return {
        "name": "specialized_emotional_support",
        "description": "Emotional support conversation where user needs to vent and seek advice",
        "test_type": "specialized_support",
        "complexity_level": "medium",
        "expected_behaviors": [
            "Show empathy and understanding",
            "Remember emotional context and previous statements",
            "Provide supportive responses",
            "Maintain helpful and caring tone",
            "Connect related emotional themes"
        ],
        "conversation_turns": [
            {
                "user_message": "I'm having a really tough time at work lately and just need someone to talk to.",
                "expected_intent": "CONVERSATIONAL_FILLER",
                "expected_topics": ["work stress", "emotional support", "venting"]
            },
            {
                "user_message": "My manager has been micromanaging everything I do and it's driving me crazy.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["micromanagement", "manager issues", "frustration"]
            },
            {
                "user_message": "I used to love my job, but now I dread going to work every morning.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["job satisfaction", "work dread", "emotional state"]
            },
            {
                "user_message": "They question every decision I make, even though I've been doing this job successfully for three years.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["lack of trust", "experience", "competence questioning"]
            },
            {
                "user_message": "How do I deal with a manager who doesn't trust my judgment?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["manager relationship", "trust issues", "advice"]
            },
            {
                "user_message": "I've tried having a conversation with them about it, but they just say they're 'staying involved' in projects.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["communication attempts", "manager response", "staying involved"]
            },
            {
                "user_message": "It's affecting my confidence. I second-guess myself on things I used to handle easily.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["confidence loss", "self-doubt", "emotional impact"]
            },
            {
                "user_message": "Should I talk to HR about this? I'm not sure if this qualifies as a real issue.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["hr involvement", "workplace issue", "uncertainty"]
            },
            {
                "user_message": "I'm also worried that complaining might make things worse or hurt my career.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["fear of retaliation", "career concerns", "workplace politics"]
            },
            {
                "user_message": "My friends say I should just look for another job, but I really like the company and my coworkers.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["job searching", "company attachment", "coworker relationships"]
            },
            {
                "user_message": "Is it normal to feel this stressed about work? Sometimes I can't sleep because I'm thinking about the next day.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["work stress", "sleep problems", "anxiety"]
            },
            {
                "user_message": "Maybe I'm being too sensitive? Other people seem to handle difficult managers just fine.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["self-blame", "sensitivity", "comparison to others"]
            },
            {
                "user_message": "What would you do in my situation? I feel stuck between staying and being miserable or leaving and starting over.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["decision advice", "staying vs leaving", "feeling stuck"]
            },
            {
                "user_message": "Thank you for listening. It helps to talk about this with someone who doesn't know all the office politics.",
                "expected_intent": "CONVERSATIONAL_FILLER",
                "expected_topics": ["gratitude", "outside perspective", "emotional relief"]
            }
        ]
    }

def create_business_meeting_scenario() -> Dict:
    """Create business meeting/professional discussion scenario"""
    
    return {
        "name": "specialized_business_meeting",
        "description": "Professional business discussion with strategic planning and decision-making",
        "test_type": "specialized_business",
        "complexity_level": "medium",
        "expected_behaviors": [
            "Maintain professional tone",
            "Track business decisions and requirements",
            "Connect strategic concepts",
            "Provide business-focused advice",
            "Remember key stakeholders and constraints"
        ],
        "conversation_turns": [
            {
                "user_message": "We need to discuss our Q4 product roadmap and make some strategic decisions for next year.",
                "expected_intent": "NEW_REQUEST",
                "expected_topics": ["q4 roadmap", "strategic planning", "product decisions"]
            },
            {
                "user_message": "Our current product has 10,000 active users, but growth has plateaued over the last two quarters.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["user metrics", "growth plateau", "performance"]
            },
            {
                "user_message": "The engineering team is split between adding new features and improving performance of existing ones.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["engineering priorities", "new features", "performance"]
            },
            {
                "user_message": "From a business perspective, should we focus on acquisition of new users or retention of existing ones?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["user acquisition", "user retention", "business strategy"]
            },
            {
                "user_message": "Our churn rate is 5% monthly, which seems high compared to industry benchmarks.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["churn rate", "industry benchmarks", "retention"]
            },
            {
                "user_message": "What are the most effective strategies for reducing churn in SaaS products?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["churn reduction", "saas strategies", "retention tactics"]
            },
            {
                "user_message": "We're also considering expanding to a new market segment - small businesses instead of just individual users.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["market expansion", "small business", "b2b transition"]
            },
            {
                "user_message": "What are the key differences in selling to businesses vs individual consumers?",
                "expected_intent": "COMPARISON",
                "expected_topics": ["b2b sales", "b2c sales", "sales strategy"]
            },
            {
                "user_message": "The sales team says businesses want enterprise features like SSO and advanced analytics.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["enterprise features", "sso", "analytics", "business requirements"]
            },
            {
                "user_message": "How do we prioritize these enterprise features against improvements for our current user base?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["feature prioritization", "enterprise vs consumer", "resource allocation"]
            },
            {
                "user_message": "Our budget for Q4 is $500K for development and $200K for marketing. How should we allocate these resources?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["budget allocation", "development budget", "marketing budget"]
            },
            {
                "user_message": "The marketing team wants to run a big campaign to acquire enterprise customers, but that would use most of the marketing budget.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["marketing campaign", "enterprise acquisition", "budget constraints"]
            },
            {
                "user_message": "What metrics should we track to measure the success of our strategy changes?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["success metrics", "kpis", "performance tracking"]
            },
            {
                "user_message": "Can you help me create a summary of our strategic options and recommendations for the board presentation?",
                "expected_intent": "ARTIFACT_GENERATION",
                "expected_topics": ["strategic summary", "board presentation", "recommendations"]
            }
        ]
    }

def create_casual_chat_scenario() -> Dict:
    """Create normal casual conversation scenario"""
    
    return {
        "name": "specialized_casual_chat",
        "description": "Natural casual conversation testing social interaction capabilities",
        "test_type": "specialized_casual",
        "complexity_level": "basic",
        "expected_behaviors": [
            "Maintain natural conversation flow",
            "Remember personal details shared",
            "Show appropriate social responses",
            "Handle topic transitions naturally",
            "Display conversational intelligence"
        ],
        "conversation_turns": [
            {
                "user_message": "Hey! How's it going? I'm just taking a break from work.",
                "expected_intent": "CONVERSATIONAL_FILLER",
                "expected_topics": ["greeting", "casual", "work break"]
            },
            {
                "user_message": "I've been working on this project for weeks and finally made a breakthrough today.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["project", "breakthrough", "accomplishment"]
            },
            {
                "user_message": "It's a machine learning model for predicting customer behavior. Pretty exciting stuff!",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["machine learning", "customer behavior", "excitement"]
            },
            {
                "user_message": "Do you ever get excited about technical achievements, or is that just a human thing?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["technical excitement", "human emotions", "ai experience"]
            },
            {
                "user_message": "That's interesting. By the way, what's your favorite programming language if you had to pick one?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["programming languages", "preferences", "technical discussion"]
            },
            {
                "user_message": "Python is great! I use it for almost everything these days. What got me into programming was actually video games.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["python", "programming background", "video games"]
            },
            {
                "user_message": "I wanted to create my own games when I was a kid. Never quite made it as a game developer though!",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["game development", "childhood dreams", "career path"]
            },
            {
                "user_message": "What about you - if you could have any job in the world, what would it be?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["ideal job", "hypothetical", "preferences"]
            },
            {
                "user_message": "That's a thoughtful answer. Speaking of helping people, I've been thinking about volunteering somewhere.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["volunteering", "helping others", "personal growth"]
            },
            {
                "user_message": "Any suggestions for good volunteer opportunities for someone with tech skills?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["tech volunteering", "volunteer opportunities", "skills application"]
            },
            {
                "user_message": "Teaching coding to kids sounds amazing! I remember how magical it felt when I wrote my first 'Hello World' program.",
                "expected_intent": "CONTINUATION",
                "expected_topics": ["teaching kids", "coding education", "first program memory"]
            },
            {
                "user_message": "Thanks for the chat! This was a nice break. Sometimes it's good to just talk about random stuff.",
                "expected_intent": "CONVERSATIONAL_FILLER",
                "expected_topics": ["gratitude", "casual conversation", "mental break"]
            }
        ]
    }

def create_examination_scenario() -> Dict:
    """Create examination/Q&A session scenario"""
    
    return {
        "name": "specialized_examination_session",
        "description": "Examination-style Q&A session with various question types and difficulty levels",
        "test_type": "specialized_examination",
        "complexity_level": "medium",
        "expected_behaviors": [
            "Handle different question types appropriately",
            "Maintain academic/testing context",
            "Provide educational explanations",
            "Track question difficulty progression",
            "Connect related concepts across questions"
        ],
        "conversation_turns": [
            {
                "user_message": "I'd like to test my knowledge of machine learning concepts. Can you ask me some questions?",
                "expected_intent": "NEW_REQUEST",
                "expected_topics": ["machine learning", "knowledge testing", "examination"]
            },
            {
                "user_message": "Start with some basic questions and then make them progressively harder.",
                "expected_intent": "META_INSTRUCTION",
                "expected_topics": ["progressive difficulty", "question structure", "learning assessment"]
            },
            {
                "user_message": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["ml definition", "ai subset", "algorithms", "data learning"]
            },
            {
                "user_message": "The three main types are supervised learning, unsupervised learning, and reinforcement learning.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["ml types", "supervised", "unsupervised", "reinforcement"]
            },
            {
                "user_message": "In supervised learning, you have input data with known correct outputs, like email classification where emails are labeled as spam or not spam.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["supervised learning", "labeled data", "email classification"]
            },
            {
                "user_message": "Overfitting occurs when a model learns the training data too well, including noise and irrelevant patterns, so it performs poorly on new, unseen data.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["overfitting", "training data", "generalization", "model performance"]
            },
            {
                "user_message": "You can prevent overfitting by using techniques like cross-validation, regularization, early stopping, or getting more training data.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["overfitting prevention", "cross-validation", "regularization", "early stopping"]
            },
            {
                "user_message": "A neural network with multiple hidden layers. The 'deep' refers to the depth of the network - how many layers it has.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["deep learning", "neural networks", "hidden layers", "network depth"]
            },
            {
                "user_message": "Gradient descent is an optimization algorithm that finds the minimum of a function by iteratively moving in the direction of steepest descent.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["gradient descent", "optimization", "steepest descent", "minimum finding"]
            },
            {
                "user_message": "Backpropagation is the algorithm used to train neural networks. It calculates gradients by working backwards through the network layers.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["backpropagation", "neural network training", "gradient calculation", "backwards pass"]
            },
            {
                "user_message": "A CNN uses convolutional layers, pooling layers, and fully connected layers. It's designed to process grid-like data such as images.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["cnn architecture", "convolutional layers", "pooling", "image processing"]
            },
            {
                "user_message": "I think the vanishing gradient problem occurs when gradients become very small in deep networks, making it hard to train the earlier layers effectively.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["vanishing gradient", "deep networks", "gradient magnitude", "training difficulty"]
            },
            {
                "user_message": "I'm not sure about this one. Could you explain the bias-variance tradeoff?",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["bias-variance tradeoff", "model complexity", "uncertainty"]
            },
            {
                "user_message": "That makes sense. High bias means the model is too simple and underfits, while high variance means it's too complex and overfits.",
                "expected_intent": "EXPLANATION",
                "expected_topics": ["bias-variance understanding", "underfitting", "overfitting", "model complexity"]
            },
            {
                "user_message": "How did I do overall? What areas should I focus on studying more?",
                "expected_intent": "STATUS_CHECK",
                "expected_topics": ["performance assessment", "study recommendations", "knowledge gaps"]
            }
        ]
    }

def create_specialized_scenarios() -> List[Dict]:
    """Create all specialized conversation scenarios"""
    
    return [
        create_debugging_session_scenario(),
        create_feature_development_scenario(), 
        create_emotional_support_scenario(),
        create_business_meeting_scenario(),
        create_casual_chat_scenario(),
        create_examination_scenario()
    ]

def run_specialized_tests():
    """Run all specialized conversation tests"""
    
    print("\n" + "🟣" * 20)
    print("🟣 SPECIALIZED CONVERSATION TESTS")
    print("🟣" * 20)
    print("\nTesting context adaptation for different use cases:")
    print("- Debugging sessions (technical problem-solving)")
    print("- Feature development (collaborative coding)")
    print("- Emotional support (vent/counseling sessions)")
    print("- Business meetings (professional discussions)")
    print("- Casual chat (social conversations)")
    print("- Examination sessions (Q&A format)")
    
    # Create test scenarios
    scenarios = create_specialized_scenarios()
    
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
    results = run_specialized_tests()
    
    if results:
        print(f"\n✅ Specialized tests completed!")
        print(f"Results saved to: {results['suite_summary']['results_file']}")
    else:
        print(f"\n❌ Specialized tests failed!")
