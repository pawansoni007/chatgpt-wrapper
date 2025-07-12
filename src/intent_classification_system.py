import json
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
    Intent classification system using Cerebras LLM with function calling.
    Classifies user messages into 11 intent categories for better response routing.
    """
    
    def __init__(self, cerebras_client):
        self.cerebras_client = cerebras_client
        
        # Configuration
        self.confidence_threshold = 0.7  # Threshold for high-confidence classification
        self.clarification_threshold = 0.5  # Below this, ask for clarification
        self.max_context_messages = 4  # Recent messages to use for context
        
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

    def classify_intent(self, user_message: str, recent_context: List[Dict] = None) -> IntentClassification:
        """
        Classify the intent of a user message
        
        Args:
            user_message: The user's current message
            recent_context: List of recent conversation messages for context
            
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
        
        try:
            # Build classification prompt
            prompt = self._build_classification_prompt(user_message, recent_context)
            
            # Get classification from Cerebras (synchronous call like in main.py)
            response = self.cerebras_client.chat.completions.create(
                model="llama-3.3-70b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for more consistent classification
                max_tokens=300
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                classification_data = json.loads(response_text)
                return self._parse_classification_response(classification_data, user_message)
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse classification JSON: {e}")
                print(f"Raw response: {response_text}")
                return self._fallback_classification(user_message, recent_context)
                
        except Exception as e:
            print(f"❌ Error in intent classification: {e}")
            return self._fallback_classification(user_message, recent_context)
    
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
                reasoning=data.get("reasoning", "LLM classification"),
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
                "max_context_messages": self.max_context_messages
            }
        }

# Example usage and testing functions
def test_intent_classifier(cerebras_client):
    """Test the intent classifier with sample messages"""
    
    classifier = EnhancedIntentClassifier(cerebras_client)
    
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
    
    print("🧪 Testing Intent Classification System")
    print("=" * 50)
    
    for i, message in enumerate(test_cases, 1):
        result = classifier.classify_intent(message)
        
        print(f"\n{i}. Message: \"{message}\"")
        print(f"   Intent: {result.intent.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Reasoning: {result.reasoning}")
        
        if result.secondary_intent:
            print(f"   Secondary: {result.secondary_intent.value}")
        
        if result.requires_clarification:
            print(f"   ⚠️  Needs clarification: {result.clarification_reason}")
    
    print("\n" + "=" * 50)
    print("✅ Intent classification testing complete!")

if __name__ == "__main__":
    # This allows testing the classifier independently
    print("Intent Classification System - Standalone Test Mode")
    print("Note: This requires CEREBRAS_API_KEY environment variable")
    
    import os
    from cerebras.cloud.sdk import Cerebras
    from dotenv import load_dotenv
    
    load_dotenv()
    
    try:
        cerebras_client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        test_intent_classifier(cerebras_client)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure CEREBRAS_API_KEY is set in your .env file")
