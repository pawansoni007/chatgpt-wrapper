import os
import json
import sys
import asyncio
from datetime import datetime
from typing import List, Dict

# Add parent directory to path to import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your conversation manager (adjust import path as needed)
from src.main import ConversationManager, redis_client
from dotenv import load_dotenv

load_dotenv()

class RedisDebugTester:
    def __init__(self):
        self.conv_manager = ConversationManager()
        self.test_conv_id = None
        
    def print_section(self, title: str):
        """Print a formatted section header"""
        print("\n" + "="*60)
        print(f"🔍 {title}")
        print("="*60)
    
    def print_redis_state(self, step: str):
        """Print current Redis state for the conversation"""
        print(f"\n📊 REDIS STATE AFTER: {step}")
        print("-" * 50)
        
        # Get conversation messages
        conv_key = self.conv_manager.get_conversation_key(self.test_conv_id)
        conversation_data = redis_client.get(conv_key)
        
        if conversation_data:
            messages = json.loads(conversation_data)
            print(f"📝 Conversation Messages ({len(messages)} total):")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:100] + "..." if len(msg.get('content', '')) > 100 else msg.get('content', '')
                timestamp = msg.get('timestamp', 'no timestamp')
                print(f"   [{i}] {role}: {content}")
                print(f"       📅 {timestamp}")
        else:
            print("📝 No conversation data found")
        
        # Get metadata
        meta_key = self.conv_manager.get_metadata_key(self.test_conv_id)
        metadata = redis_client.get(meta_key)
        
        if metadata:
            meta_dict = json.loads(metadata)
            print(f"\n📋 Metadata:")
            for key, value in meta_dict.items():
                print(f"   {key}: {value}")
        else:
            print("\n📋 No metadata found")
        
        # Get embeddings
        embedding_keys = redis_client.keys(f"embedding:{self.test_conv_id}:*")
        print(f"\n🧠 Embeddings ({len(embedding_keys)} total):")
        for key in sorted(embedding_keys):
            embedding_data = redis_client.get(key)
            if embedding_data:
                embedding = json.loads(embedding_data)
                # Show first few dimensions of embedding
                preview = embedding[:5] if len(embedding) > 5 else embedding
                print(f"   {key}: [{', '.join([f'{x:.3f}' for x in preview])}, ...] (dim: {len(embedding)})")
            else:
                print(f"   {key}: No data")
    
    def print_context_selection(self, user_message: str, context: List[Dict], tokens_used: int):
        """Print context selection details"""
        print(f"\n🎯 CONTEXT SELECTION FOR: '{user_message}'")
        print("-" * 50)
        print(f"📊 Total tokens used: {tokens_used}")
        print(f"📝 Context messages ({len(context)} total):")
        
        for i, msg in enumerate(context):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:80] + "..." if len(msg.get('content', '')) > 80 else msg.get('content', '')
            token_count = self.conv_manager.count_tokens(msg.get('content', ''))
            print(f"   [{i}] {role} ({token_count} tokens): {content}")
    
    def test_semantic_selection(self, user_message: str):
        """Test and debug semantic message selection"""
        print(f"\n🔍 SEMANTIC SELECTION DEBUG FOR: '{user_message}'")
        print("-" * 50)
        
        # Get relevant messages with debug info
        relevant_messages = self.conv_manager.get_relevant_messages(self.test_conv_id, user_message)
        
        print(f"🧠 Found {len(relevant_messages)} relevant messages:")
        for msg_data in relevant_messages:
            idx = msg_data['index']
            similarity = msg_data['similarity']
            message = msg_data['message']
            role = message.get('role', 'unknown')
            content = message.get('content', '')[:60] + "..." if len(message.get('content', '')) > 60 else message.get('content', '')
            
            print(f"   [{idx}] {role} (similarity: {similarity:.3f}): {content}")
        
        # Show threshold info
        print(f"\n📏 Similarity threshold: {self.conv_manager.semantic_similarity_threshold}")
        print(f"🔢 Simple conversation threshold: {self.conv_manager.simple_conversation_threshold}")
    
    async def run_full_test(self):
        """Run the complete test scenario"""
        
        self.print_section("STARTING REDIS DEBUG TEST")
        
        # Generate test conversation ID
        self.test_conv_id = self.conv_manager.generate_conversation_id()
        print(f"🆔 Test Conversation ID: {self.test_conv_id}")
        
        # Test messages with context switches
        test_messages = [
            "I'm building a React app and need help with state management. Should I use useState or Redux?",
            "Actually, let me switch topics. Can you explain how machine learning algorithms work?",
            "Back to React - what's the difference between useEffect and useLayoutEffect?",
            "One more ML question - what's the difference between supervised and unsupervised learning?"
        ]
        
        for i, user_msg in enumerate(test_messages, 1):
            self.print_section(f"MESSAGE {i}: SENDING USER MESSAGE")
            print(f"💬 User Message: '{user_msg}'")
            
            # Show context selection BEFORE sending
            conversation = self.conv_manager.get_conversation(self.test_conv_id)
            print(f"📊 Current conversation length: {len(conversation)}")
            
            # Test context preparation
            context, tokens_used = self.conv_manager.prepare_context(self.test_conv_id, user_msg)
            self.print_context_selection(user_msg, context, tokens_used)
            
            # Test semantic selection if conversation is long enough
            if len(conversation) > self.conv_manager.simple_conversation_threshold:
                self.test_semantic_selection(user_msg)
            else:
                print(f"\n📝 Using simple context selection (conversation too short: {len(conversation)} <= {self.conv_manager.simple_conversation_threshold})")
            
            # Simulate AI response (we'll create a mock response instead of calling Cerebras)
            mock_responses = [
                "For small React apps, useState is usually sufficient. Redux adds overhead and complexity that's often unnecessary. Consider useState with useContext for medium-sized apps, and only use Redux for large, complex state management needs.",
                "Machine learning algorithms are computational methods that enable systems to learn patterns from data. They work by finding mathematical relationships in training data and using those patterns to make predictions on new, unseen data.",
                "useEffect runs after the DOM has been painted, while useLayoutEffect runs synchronously after all DOM mutations but before the browser paints. Use useLayoutEffect when you need to measure DOM elements or make visual changes that users shouldn't see.",
                "Supervised learning uses labeled training data (input-output pairs) to learn patterns, like email spam detection. Unsupervised learning finds hidden patterns in unlabeled data, like customer segmentation or anomaly detection."
            ]
            
            assistant_response = mock_responses[i-1]
            print(f"🤖 Assistant Response: '{assistant_response[:100]}...'")
            
            # Add the exchange (this will store embeddings)
            self.conv_manager.add_exchange(self.test_conv_id, user_msg, assistant_response)
            
            # Show Redis state after this exchange
            self.print_redis_state(f"Message {i} Exchange")
            
            # Add a small delay to make timestamps different
            await asyncio.sleep(0.1)
        
        # Final analysis
        self.print_section("FINAL ANALYSIS")
        
        # Test final semantic selection
        test_query = "Tell me more about React hooks"
        print(f"🔍 Testing semantic selection for: '{test_query}'")
        self.test_semantic_selection(test_query)
        
        # Show final context selection
        final_context, final_tokens = self.conv_manager.prepare_context(self.test_conv_id, test_query)
        self.print_context_selection(test_query, final_context, final_tokens)
        
        print(f"\n✅ Test completed! Conversation ID: {self.test_conv_id}")
        print(f"🗑️  To clean up, run: redis-cli DEL conversation:{self.test_conv_id} metadata:{self.test_conv_id} embedding:{self.test_conv_id}:*")

    def cleanup_test_data(self):
        """Clean up test data from Redis"""
        if self.test_conv_id:
            try:
                # Delete conversation and metadata
                redis_client.delete(self.conv_manager.get_conversation_key(self.test_conv_id))
                redis_client.delete(self.conv_manager.get_metadata_key(self.test_conv_id))
                
                # Delete all embeddings
                embedding_keys = redis_client.keys(f"embedding:{self.test_conv_id}:*")
                if embedding_keys:
                    redis_client.delete(*embedding_keys)
                
                print(f"🗑️  Cleaned up test data for conversation: {self.test_conv_id}")
            except Exception as e:
                print(f"❌ Error cleaning up: {e}")

async def main():
    """Run the debug test"""
    tester = RedisDebugTester()
    
    try:
        # Test Redis connection
        if not redis_client.ping():
            print("❌ Redis connection failed!")
            return
        
        print("✅ Redis connection successful!")
        
        # Run the test
        await tester.run_full_test()
        
        # Ask if user wants to cleanup
        cleanup = input("\n🗑️  Do you want to cleanup test data? (y/n): ").lower().strip()
        if cleanup == 'y':
            tester.cleanup_test_data()
        else:
            print(f"📝 Test data preserved. Conversation ID: {tester.test_conv_id}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())