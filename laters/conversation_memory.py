import json
from datetime import datetime
from typing import List, Dict, Optional

class ConversationMemory:
    """Handles persistent conversation memory and summarization"""
    
    def __init__(self, redis_client, cerebras_client):
        self.redis_client = redis_client
        self.cerebras_client = cerebras_client
        
        # Memory configuration
        self.memory_update_threshold = 4  # Update memory every 4 new messages
        self.max_key_events = 10         # Maximum key events to store
        self.max_topics = 8              # Maximum topics to track
        self.summary_token_limit = 300   # Token limit for memory summaries
        
    def get_memory_key(self, conv_id: str) -> str:
        """Get Redis key for conversation memory"""
        return f"memory:{conv_id}"
    
    def get_conversation_memory(self, conv_id: str) -> Dict:
        """Get existing conversation memory from Redis"""
        try:
            memory_data = self.redis_client.get(self.get_memory_key(conv_id))
            if memory_data:
                return json.loads(memory_data)
            
            # Return empty memory structure if none exists
            return self.create_empty_memory()
        except Exception as e:
            print(f"Error getting conversation memory: {e}")
            return self.create_empty_memory()
    
    def create_empty_memory(self) -> Dict:
        """Create empty memory structure"""
        return {
            "key_events": [],
            "topics_discussed": {},
            "user_context": {
                "preferences": [],
                "expertise_level": "unknown",
                "common_requests": []
            },
            "conversation_state": "new",
            "last_updated": datetime.now().isoformat(),
            "memory_version": 1
        }
    
    def save_conversation_memory(self, conv_id: str, memory: Dict):
        """Save conversation memory to Redis"""
        try:
            memory["last_updated"] = datetime.now().isoformat()
            self.redis_client.set(
                self.get_memory_key(conv_id),
                json.dumps(memory)
            )
            print(f"✅ Updated conversation memory for {conv_id}")
        except Exception as e:
            print(f"❌ Error saving conversation memory: {e}")
    
    def should_update_memory(self, conv_id: str, current_message_count: int) -> bool:
        """Determine if memory should be updated based on message count"""
        memory = self.get_conversation_memory(conv_id)
        
        # Check if we have enough new messages since last update
        last_known_count = len(memory.get("key_events", [])) * 2  # Rough estimate
        new_messages = current_message_count - last_known_count
        
        return new_messages >= self.memory_update_threshold
    
    async def generate_conversation_summary(self, messages: List[Dict]) -> Dict:
        """Generate a structured summary of recent conversation exchanges"""
        
        # Prepare messages for summarization
        recent_messages = messages[-8:] if len(messages) > 8 else messages
        
        # Build conversation text for LLM
        conversation_text = self.build_conversation_text(recent_messages)
        
        # Create summarization prompt
        summary_prompt = f"""
Analyze this conversation and extract structured information:

{conversation_text}

Respond with a JSON object in this exact format:
{{
    "key_events": ["Brief description of each significant interaction"],
    "topics_discussed": {{"topic_name": ["subtopic1", "subtopic2"]}},
    "user_preferences": ["What the user seems to prefer or focus on"],
    "conversation_state": "brief description of current conversation state"
}}

Focus on:
- What actually happened in the conversation
- What the user asked for and what was provided  
- Technical topics and implementation details
- User's apparent expertise level and interests

DO NOT include anything that didn't actually happen. Be factual and specific.
Your entire response must be valid JSON only.
"""

        try:
            # Get summary from LLM
            response = await self.cerebras_client.chat.completions.create(
                model="llama-3.3-70b",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,  # Lower temperature for more consistent summaries
                max_tokens=400
            )
            
            summary_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                summary_data = json.loads(summary_text)
                print(f"✅ Generated conversation summary: {len(summary_data.get('key_events', []))} events")
                return summary_data
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse summary JSON: {e}")
                print(f"Raw response: {summary_text}")
                return self.create_fallback_summary(recent_messages)
                
        except Exception as e:
            print(f"❌ Error generating conversation summary: {e}")
            return self.create_fallback_summary(recent_messages)
    
    def build_conversation_text(self, messages: List[Dict]) -> str:
        """Build formatted conversation text for summarization"""
        conversation_lines = []
        
        for msg in messages:
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '')
            
            # Truncate very long messages for summarization
            if len(content) > 200:
                content = content[:200] + "..."
            
            conversation_lines.append(f"{role}: {content}")
        
        return "\n".join(conversation_lines)
    
    def create_fallback_summary(self, messages: List[Dict]) -> Dict:
        """Create a basic fallback summary when LLM summarization fails"""
        print("🔄 Using fallback summary generation")
        
        # Extract basic info from messages
        key_events = []
        topics = {}
        
        for i in range(0, len(messages), 2):  # Process in Q&A pairs
            if i + 1 < len(messages):
                user_msg = messages[i].get('content', '')
                assistant_msg = messages[i + 1].get('content', '')
                
                # Create basic event description
                if len(user_msg) > 0:
                    event = f"User asked about {self.extract_topic_keywords(user_msg)}"
                    key_events.append(event)
        
        return {
            "key_events": key_events[-self.max_key_events:],
            "topics_discussed": topics,
            "user_preferences": [],
            "conversation_state": "active discussion"
        }
    
    def extract_topic_keywords(self, text: str) -> str:
        """Extract key topic words from a message"""
        # Simple keyword extraction (can be enhanced with NLP)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'can', 'you', 'how', 'what', 'when', 'where', 'why'}
        words = text.lower().split()
        keywords = [word for word in words[:10] if word not in common_words and len(word) > 2]
        return ' '.join(keywords[:3])  # Return first 3 relevant keywords
    
    def merge_memory_updates(self, existing_memory: Dict, new_summary: Dict) -> Dict:
        """Merge new summary data with existing memory"""
        
        # Merge key events (avoid duplicates)
        existing_events = existing_memory.get("key_events", [])
        new_events = new_summary.get("key_events", [])
        
        merged_events = existing_events + new_events
        merged_events = merged_events[-self.max_key_events:]  # Keep only recent events
        
        # Merge topics
        existing_topics = existing_memory.get("topics_discussed", {})
        new_topics = new_summary.get("topics_discussed", {})
        
        for topic, subtopics in new_topics.items():
            if topic in existing_topics:
                # Merge subtopics
                existing_topics[topic] = list(set(existing_topics[topic] + subtopics))
            else:
                existing_topics[topic] = subtopics
        
        # Keep only recent topics (limit storage)
        if len(existing_topics) > self.max_topics:
            # Keep the most recent topics (simple approach - can be enhanced)
            topic_items = list(existing_topics.items())
            existing_topics = dict(topic_items[-self.max_topics:])
        
        # Merge user preferences
        existing_prefs = existing_memory.get("user_context", {}).get("preferences", [])
        new_prefs = new_summary.get("user_preferences", [])
        merged_prefs = list(set(existing_prefs + new_prefs))
        
        # Build merged memory
        merged_memory = {
            "key_events": merged_events,
            "topics_discussed": existing_topics,
            "user_context": {
                "preferences": merged_prefs,
                "expertise_level": existing_memory.get("user_context", {}).get("expertise_level", "unknown"),
                "common_requests": existing_memory.get("user_context", {}).get("common_requests", [])
            },
            "conversation_state": new_summary.get("conversation_state", "ongoing"),
            "memory_version": existing_memory.get("memory_version", 1) + 1
        }
        
        return merged_memory
    
    async def update_conversation_memory(self, conv_id: str, messages: List[Dict]) -> bool:
        """Update conversation memory with recent messages"""
        try:
            print(f"🧠 Updating conversation memory for {conv_id}")
            
            # Get existing memory
            existing_memory = self.get_conversation_memory(conv_id)
            
            # Generate summary of recent conversation
            new_summary = await self.generate_conversation_summary(messages)
            
            # Merge with existing memory
            updated_memory = self.merge_memory_updates(existing_memory, new_summary)
            
            # Save updated memory
            self.save_conversation_memory(conv_id, updated_memory)
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating conversation memory: {e}")
            return False
    
    def get_memory_context_string(self, conv_id: str) -> str:
        """Get memory as a formatted string for context"""
        memory = self.get_conversation_memory(conv_id)
        
        if not memory or not memory.get("key_events"):
            return "No previous conversation history."
        
        context_parts = []
        
        # Add key events
        events = memory.get("key_events", [])
        if events:
            context_parts.append("Previous conversation:")
            for event in events[-5:]:  # Last 5 events
                context_parts.append(f"- {event}")
        
        # Add current topics
        topics = memory.get("topics_discussed", {})
        if topics:
            topic_list = list(topics.keys())
            context_parts.append(f"Topics discussed: {', '.join(topic_list)}")
        
        # Add conversation state
        state = memory.get("conversation_state", "")
        if state and state != "new":
            context_parts.append(f"Current context: {state}")
        
        return "\n".join(context_parts)


# Integration with existing ConversationManager
class EnhancedConversationManager:
    """Enhanced conversation manager with memory capabilities"""
    
    def __init__(self, existing_conv_manager, cerebras_client, redis_client):
        # Keep existing functionality
        self.base_manager = existing_conv_manager
        
        # Add memory management
        self.memory_manager = ConversationMemory(redis_client, cerebras_client)
        
        # Copy existing attributes
        self.max_tokens = existing_conv_manager.max_tokens
        self.reserved_tokens = existing_conv_manager.reserved_tokens
        self.simple_conversation_threshold = existing_conv_manager.simple_conversation_threshold
        self.semantic_similarity_threshold = existing_conv_manager.semantic_similarity_threshold
    
    def __getattr__(self, name):
        """Delegate unknown attributes to base manager"""
        return getattr(self.base_manager, name)
    
    async def add_exchange_with_memory(self, conv_id: str, user_message: str, assistant_message: str):
        """Enhanced add_exchange that updates conversation memory"""
        
        # Add exchange using existing logic
        self.base_manager.add_exchange(conv_id, user_message, assistant_message)
        
        # Check if memory should be updated
        conversation = self.base_manager.get_conversation(conv_id)
        
        if self.memory_manager.should_update_memory(conv_id, len(conversation)):
            print(f"🧠 Triggering memory update for conversation {conv_id}")
            await self.memory_manager.update_conversation_memory(conv_id, conversation)
        
        return True
    
    def prepare_context_with_memory(self, conv_id: str, user_message: str) -> tuple[List[Dict], int]:
        """Enhanced context preparation that includes conversation memory"""
        
        # Get memory context
        memory_context = self.memory_manager.get_memory_context_string(conv_id)
        
        # Get regular context using existing logic
        context, tokens_used = self.base_manager.prepare_context(conv_id, user_message)
        
        # Enhance system message with memory
        if context and context[0].get('role') == 'system':
            enhanced_system_content = f"{context[0]['content']}\n\nConversation Memory:\n{memory_context}"
            context[0]['content'] = enhanced_system_content
            
            # Recalculate tokens
            memory_tokens = self.base_manager.count_tokens(memory_context)
            tokens_used += memory_tokens
            
            print(f"🧠 Added {memory_tokens} tokens from conversation memory")
        
        return context, tokens_used