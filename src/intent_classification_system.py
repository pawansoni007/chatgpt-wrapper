import json
import hashlib
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

class IntentCategory(Enum):
    """11-category intent classification system"""
    
    # Task-Oriented Intents
    NEW_REQUEST = "NEW_REQUEST"           # Start new project/feature
    CONTINUATION = "CONTINUATION"         # Add to existing work  
    REFINEMENT = "REFINEMENT"            # Improve existing code
    DEBUGGING = "DEBUGGING"              # Fix errors/problems
    ARTIFACT_GENERATION = "ARTIFACT_GENERATION"  # Create supporting content (tests, docs, tool use)
    
    # Knowledge-Seeking Intents
    STATUS_CHECK = "STATUS_CHECK"        # Check project progress/state
    EXPLANATION = "EXPLANATION"          # Understand concepts/theory
    COMPARISON = "COMPARISON"            # Explore alternatives/options
    
    # Context-Management Intents
    CORRECTION = "CORRECTION"            # Fix AI misunderstandings
    META_INSTRUCTION = "META_INSTRUCTION"  # Control conversation behavior
    CONVERSATIONAL_FILLER = "CONVERSATIONAL_FILLER"  # Social responses

@dataclass
class IntentClassification:
    """Result of intent classification"""
    intent: IntentCategory
    confidence: float  # 0.0 to 1.0
    reasoning: str
    secondary_intent: Optional[IntentCategory] = None
    requires_clarification: bool = False
    clarification_reason: Optional[str] = None

class EnhancedIntentClassifier:
    """
    Azure OpenAI powered intent classification with stability features:
    - Uses GPT-4o-mini for consistent, cost-effective intent classification
    - Very low temperature for deterministic behavior
    - Intent caching for similar messages
    - Intent smoothing to prevent oscillation
    """
    
    def __init__(self, azure_client):
        # Use Azure OpenAI client
        self.azure_client = azure_client
        
        # Configuration for stability
        self.confidence_threshold = 0.7
        self.clarification_threshold = 0.5
        self.max_context_messages = 4
        
        # Stability features
        self.intent_cache = {}  # Cache for similar messages
        self.conversation_intent_history = {}  # Track intent patterns per conversation
        self.cache_similarity_threshold = 0.85  # High similarity = use cached intent
        self.smoothing_window = 3  # Consider last 3 intents for smoothing
        
        # Intent patterns for quick detection
        self.intent_patterns = self._build_intent_patterns()
    
    def _build_intent_patterns(self) -> Dict[IntentCategory, Dict]:
        """Build pattern definitions for each intent category"""
        return {
            IntentCategory.NEW_REQUEST: {
                "keywords": ["create", "build", "make", "start", "new", "begin", "implement", "develop"],
                "patterns": ["can you help me", "i want to", "let's create", "i need to build"],
                "context_clues": ["project", "from scratch", "new feature", "new topic"]
            },
            
            IntentCategory.CONTINUATION: {
                "keywords": ["continue", "next", "also", "and", "add", "extend", "more"],
                "patterns": ["continue with", "next step", "also add", "let's also"],
                "context_clues": ["building on", "in addition to"]
            },
            
            IntentCategory.REFINEMENT: {
                "keywords": ["improve", "optimize", "better", "enhance", "refactor", "clean", "efficient"],
                "patterns": ["make it better", "can we improve", "optimize this", "more efficient"],
                "context_clues": ["performance", "cleaner code", "best practices"]
            },
            
            IntentCategory.DEBUGGING: {
                "keywords": ["error", "bug", "fix", "problem", "issue", "broken", "not working", "fails"],
                "patterns": ["getting error", "doesn't work", "something wrong", "fix this"],
                "context_clues": ["stack trace", "exception", "failed"]
            },
            
            IntentCategory.ARTIFACT_GENERATION: {
                "keywords": ["test", "documentation", "readme", "example", "demo", "tutorial"],
                "patterns": ["write tests", "create docs", "add readme", "show example"],
                "context_clues": ["unit test", "api docs", "documentation"]
            },
            
            IntentCategory.STATUS_CHECK: {
                "keywords": ["status", "progress", "done", "complete", "finished", "where are we"],
                "patterns": ["what's the status", "are we done", "how's progress", "where do we stand"],
                "context_clues": ["completion", "remaining work"]
            },
            
            IntentCategory.EXPLANATION: {
                "keywords": ["explain", "how", "what", "why", "understand", "learn", "clarify"],
                "patterns": ["can you explain", "how does", "what is", "why did", "help me understand"],
                "context_clues": ["concept", "theory", "mechanism"]
            },
            
            IntentCategory.COMPARISON: {
                "keywords": ["compare", "versus", "vs", "alternative", "option", "choice", "difference"],
                "patterns": ["compare with", "vs", "which is better", "alternatives to"],
                "context_clues": ["pros and cons", "trade-offs", "options"]
            },
            
            IntentCategory.CORRECTION: {
                "keywords": ["no", "not", "wrong", "actually", "instead", "correction", "mistake"],
                "patterns": ["that's not right", "actually it's", "i meant", "correction:"],
                "context_clues": ["misunderstood", "clarification"]
            },
            
            IntentCategory.META_INSTRUCTION: {
                "keywords": ["please", "always", "never", "remember", "format", "style", "behavior"],
                "patterns": ["please always", "don't", "remember to", "format it as"],
                "context_clues": ["instruction", "preference", "guideline"]
            },
            
            IntentCategory.CONVERSATIONAL_FILLER: {
                "keywords": ["thanks", "thank you", "ok", "okay", "got it", "perfect", "great"],
                "patterns": ["thank you", "got it", "makes sense", "that's perfect"],
                "context_clues": ["acknowledgment", "appreciation"]
            }
        }
    
    def _build_classification_prompt(self, user_message: str, recent_context: List[Dict]) -> str:
        """Build the prompt for intent classification"""
        
        # Build context string
        context_str = ""
        if recent_context:
            context_str = "Recent conversation context:\n"
            for i, msg in enumerate(recent_context[-self.max_context_messages:]):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:200]  # Truncate long messages
                context_str += f"{role}: {content}\n"
            context_str += "\n"
        
        # Intent definitions for the LLM
        intent_definitions = """
            Intent Categories:

            **Task-Oriented Intents:**
            1. NEW_REQUEST - Starting a new project, feature, or task from scratch
            2. CONTINUATION - Adding to or continuing existing work or discussion
            3. REFINEMENT - Improving, optimizing, or enhancing existing code/work
            4. DEBUGGING - Fixing errors, bugs, or problems
            5. ARTIFACT_GENERATION - Creating supporting content (tests, docs, examples)

            **Knowledge-Seeking Intents:**
            6. STATUS_CHECK - Checking progress, completion status, or current state
            7. EXPLANATION - Understanding concepts, theory, or how things work
            8. COMPARISON - Exploring alternatives, options, or comparing approaches

            **Context-Management Intents:**
            9. CORRECTION - Fixing misunderstandings or providing corrections
            10. META_INSTRUCTION - Controlling conversation behavior or setting preferences
            11. CONVERSATIONAL_FILLER - Social responses, acknowledgments, thanks
        """
        
        return f"""{context_str}User's current message: "{user_message}"

        {intent_definitions}

        Analyze the user's message and classify their intent. Consider:
        1. What is the user trying to accomplish?
        2. Are they continuing previous work or starting something new?
        3. Do they need information or want action taken?
        4. Is this a technical request or social interaction?

        Respond with a JSON object in this exact format:
        {{
            "intent": "INTENT_CATEGORY_NAME",
            "confidence": 0.85,
            "reasoning": "Brief explanation of why this intent was chosen",
            "secondary_intent": "OPTIONAL_SECONDARY_INTENT",
            "requires_clarification": false,
            "clarification_reason": null
        }}

        Guidelines:
        - Use confidence 0.9+ for very clear intents
        - Use confidence 0.7-0.9 for clear intents  
        - Use confidence 0.5-0.7 for somewhat unclear intents
        - Set requires_clarification=true and confidence below 0.5 if the intent is genuinely ambiguous
        - Include secondary_intent if there's a clear secondary purpose
        - Be specific and accurate in your reasoning

        Your entire response must be valid JSON only."""

    def classify_intent(self, user_message: str, recent_context: List[Dict] = None, conv_id: str = None) -> IntentClassification:
        """
        Azure OpenAI powered intent classification with stability enhancements
        
        Args:
            user_message: The user's current message
            recent_context: List of recent conversation messages for context
            conv_id: Conversation ID for intent smoothing
            
        Returns:
            IntentClassification object with intent, confidence, and reasoning
        """
        
        if not user_message or not user_message.strip():
            return IntentClassification(
                intent=IntentCategory.CONVERSATIONAL_FILLER,
                confidence=0.5,
                reasoning="Empty or whitespace-only message"
            )
        
        recent_context = recent_context or []
        
        # Check cache for similar messages first
        cached_intent = self._check_intent_cache(user_message)
        if cached_intent:
            print(f"🎯 Using cached intent: {cached_intent.intent.value}")
            return cached_intent
        
        try:
            # Build classification prompt
            prompt = self._build_classification_prompt(user_message, recent_context)
            
            # Use Azure OpenAI with very low temperature for consistency
            response = asyncio.run(self.azure_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                task_type="intent",  # This will use GPT-4o-mini
                temperature=0.05,    # VERY low temperature for maximum consistency
                max_tokens=300
            ))
            
            print(f"✅ Intent classified using Azure OpenAI {response.model_used}")
            
            response_text = response.content.strip()
            
            # Parse JSON response
            try:
                classification_data = json.loads(response_text)
                raw_intent = self._parse_classification_response(classification_data, user_message)
                
                # Apply intent smoothing based on conversation history
                smoothed_intent = self._apply_intent_smoothing(raw_intent, conv_id) if conv_id else raw_intent
                
                # Cache the result
                self._cache_intent(user_message, smoothed_intent)
                
                # Update conversation history
                if conv_id:
                    self._update_conversation_intent_history(conv_id, smoothed_intent)
                
                return smoothed_intent
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse classification JSON: {e}")
                print(f"Raw response: {response_text}")
                return self._fallback_classification(user_message, recent_context)
                
        except Exception as e:
            print(f"❌ Error in intent classification: {e}")
            return self._fallback_classification(user_message, recent_context)
    
    def _check_intent_cache(self, user_message: str) -> Optional[IntentClassification]:
        """Check if we have a cached intent for similar message"""
        message_hash = self._hash_message(user_message)
        
        # Check for exact match first
        if message_hash in self.intent_cache:
            return self.intent_cache[message_hash]['intent']
        
        # Check for similar messages (simplified similarity check)
        for cached_hash, cached_data in self.intent_cache.items():
            cached_message = cached_data['message']
            if self._simple_similarity(user_message, cached_message) > self.cache_similarity_threshold:
                print(f"🔄 Found similar cached message")
                return cached_data['intent']
        
        return None
    
    def _apply_intent_smoothing(self, current_intent: IntentClassification, conv_id: str) -> IntentClassification:
        """Apply smoothing based on recent intent history"""
        if not conv_id:
            return current_intent
            
        history = self.conversation_intent_history.get(conv_id, [])
        
        if len(history) < 2:
            return current_intent  # Not enough history for smoothing
        
        # Check if current intent is dramatically different from recent pattern
        recent_intents = [h.intent for h in history[-self.smoothing_window:]]
        most_common_recent = max(set(recent_intents), key=recent_intents.count)
        
        # If confidence is low and recent pattern is strong, bias toward recent pattern
        if (current_intent.confidence < 0.7 and 
            recent_intents.count(most_common_recent) >= 2 and
            current_intent.intent != most_common_recent):
            
            print(f"🔄 Intent smoothing: {current_intent.intent.value} → {most_common_recent.value}")
            
            # Create smoothed intent with adjusted confidence
            return IntentClassification(
                intent=most_common_recent,
                confidence=0.7,  # Moderate confidence for smoothed intent
                reasoning=f"Smoothed from {current_intent.intent.value} based on recent pattern",
                secondary_intent=current_intent.intent
            )
        
        return current_intent
    
    def _cache_intent(self, user_message: str, intent: IntentClassification):
        """Cache the intent result"""
        message_hash = self._hash_message(user_message)
        self.intent_cache[message_hash] = {
            'message': user_message,
            'intent': intent,
            'timestamp': datetime.now()
        }
        
        # Keep cache size reasonable
        if len(self.intent_cache) > 100:
            # Remove oldest entries
            oldest_items = sorted(self.intent_cache.items(), key=lambda x: x[1]['timestamp'])
            for old_hash, _ in oldest_items[:20]:
                del self.intent_cache[old_hash]
    
    def _update_conversation_intent_history(self, conv_id: str, intent: IntentClassification):
        """Update conversation intent history"""
        if conv_id not in self.conversation_intent_history:
            self.conversation_intent_history[conv_id] = []
        
        self.conversation_intent_history[conv_id].append(intent)
        
        # Keep history size reasonable
        if len(self.conversation_intent_history[conv_id]) > 20:
            self.conversation_intent_history[conv_id] = self.conversation_intent_history[conv_id][-15:]
    
    def _hash_message(self, message: str) -> str:
        """Create hash for message caching"""
        normalized = message.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _simple_similarity(self, msg1: str, msg2: str) -> float:
        """Simple similarity check for caching"""
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _parse_classification_response(self, data: Dict, user_message: str) -> IntentClassification:
        """Parse the LLM's classification response into IntentClassification object"""
        
        try:
            # Get primary intent
            intent_str = data.get("intent", "CONVERSATIONAL_FILLER")
            try:
                intent = IntentCategory(intent_str)
            except ValueError:
                print(f"⚠️ Unknown intent category: {intent_str}, using CONVERSATIONAL_FILLER")
                intent = IntentCategory.CONVERSATIONAL_FILLER
            
            # Get secondary intent if provided
            secondary_intent = None
            if data.get("secondary_intent"):
                try:
                    secondary_intent = IntentCategory(data["secondary_intent"])
                except ValueError:
                    print(f"⚠️ Unknown secondary intent: {data.get('secondary_intent')}")
            
            # Validate confidence
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            
            return IntentClassification(
                intent=intent,
                confidence=confidence,
                reasoning=data.get("reasoning", "Azure OpenAI classification"),
                secondary_intent=secondary_intent,
                requires_clarification=bool(data.get("requires_clarification", False)),
                clarification_reason=data.get("clarification_reason")
            )
            
        except Exception as e:
            print(f"❌ Error parsing classification response: {e}")
            return self._fallback_classification(user_message, [])
    
    def _fallback_classification(self, user_message: str, recent_context: List[Dict]) -> IntentClassification:
        """Provide fallback classification using simple pattern matching"""
        
        print("🔄 Using fallback pattern-based classification")
        
        message_lower = user_message.lower()
        
        # Check each intent pattern
        best_match = None
        max_score = 0
        
        for intent_category, patterns in self.intent_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in message_lower:
                    score += 1
            
            # Check patterns
            for pattern in patterns["patterns"]:
                if pattern in message_lower:
                    score += 2  # Patterns worth more than keywords
            
            # Update best match
            if score > max_score:
                max_score = score
                best_match = intent_category
        
        # If no clear pattern match, use heuristics
        if not best_match or max_score == 0:
            if any(word in message_lower for word in ["error", "bug", "problem", "fix"]):
                best_match = IntentCategory.DEBUGGING
            elif any(word in message_lower for word in ["how", "what", "why", "explain"]):
                best_match = IntentCategory.EXPLANATION
            elif any(word in message_lower for word in ["continue", "next", "also"]):
                best_match = IntentCategory.CONTINUATION
            elif any(word in message_lower for word in ["create", "build", "make", "new"]):
                best_match = IntentCategory.NEW_REQUEST
            else:
                best_match = IntentCategory.CONVERSATIONAL_FILLER
        
        # Calculate confidence based on pattern match strength
        confidence = min(0.7, max_score * 0.1 + 0.3)  # 0.3 to 0.7 range
        
        return IntentClassification(
            intent=best_match,
            confidence=confidence,
            reasoning=f"Fallback pattern matching (score: {max_score})",
            requires_clarification=confidence < self.clarification_threshold
        )
    
    def classify_intent_sync(self, user_message: str, recent_context: List[Dict] = None) -> IntentClassification:
        """Kept for backward compatibility - now just calls classify_intent directly"""
        return self.classify_intent(user_message, recent_context)
    
    def get_intent_statistics(self) -> Dict:
        """Get statistics about intent patterns (for debugging)"""
        return {
            "total_categories": len(IntentCategory),
            "categories": [intent.value for intent in IntentCategory],
            "patterns_loaded": len(self.intent_patterns),
            "config": {
                "confidence_threshold": self.confidence_threshold,
                "clarification_threshold": self.clarification_threshold,
                "max_context_messages": self.max_context_messages,
                "cache_size": len(self.intent_cache),
                "conversations_tracked": len(self.conversation_intent_history)
            },
            "stability_features": {
                "intent_caching": "enabled",
                "intent_smoothing": "enabled", 
                "llm_provider": "Azure OpenAI GPT-4o-mini",
                "temperature": 0.05
            }
        }

# Example usage and testing functions
def test_intent_classifier(azure_client):
    """Test the intent classifier with sample messages"""
    
    classifier = EnhancedIntentClassifier(azure_client)
    
    test_cases = [
        "I want to make my new feature work, currently having errors",
        "Can you help me build a new React app?",
        "There's an error in my Python code",
        "Continue with the authentication system",
        "How does JWT work?",
        "Make this code more efficient",
        "Thanks, that's perfect!",
        "What's the status of our project?",
        "Compare React vs Vue",
        "Actually, I meant Express not React",
        "Please always format code in markdown"
    ]
    
    print("🧪 Testing Azure OpenAI Intent Classification System")
    print("=" * 60)
    
    for i, message in enumerate(test_cases, 1):
        result = classifier.classify_intent(message, conv_id="test_conversation")
        
        print(f"\n{i}. Message: \"{message}\"")
        print(f"   Intent: {result.intent.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Reasoning: {result.reasoning}")
        
        if result.secondary_intent:
            print(f"   Secondary: {result.secondary_intent.value}")
        
        if result.requires_clarification:
            print(f"   ⚠️  Needs clarification: {result.clarification_reason}")
    
    print("\n" + "=" * 60)
    print("✅ Azure OpenAI Intent classification testing complete!")
    print(f"📊 Final stats: {classifier.get_intent_statistics()}")

if __name__ == "__main__":
    # This allows testing the classifier independently
    print("Azure OpenAI Intent Classification System - Standalone Test Mode")
    
    from llm_client import AzureOpenAIClient
    
    try:
        azure_client = AzureOpenAIClient()
        test_intent_classifier(azure_client)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Azure OpenAI credentials are set in your .env file")
