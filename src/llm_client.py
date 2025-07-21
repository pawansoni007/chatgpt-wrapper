"""
Azure OpenAI Only LLM Client
============================

Clean, simple Azure OpenAI client without fallbacks or complexity.
Handles chat completions and embeddings.
"""

import os
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class LLMResponse:
    """Response from Azure OpenAI"""
    content: str
    tokens_used: int
    model_used: str

class AzureOpenAIClient:
    """
    Simple Azure OpenAI client for chat and embeddings
    """
    
    def __init__(self):
        # Azure OpenAI configuration
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        
        # Model deployment names
        self.chat_model = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o")
        self.intent_model = os.getenv("AZURE_OPENAI_INTENT_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Initialize client
        self._initialize_client()
        
        # Token counter
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client"""
        if not self.api_key or not self.endpoint:
            raise ValueError("Azure OpenAI API key and endpoint must be provided")
        
        try:
            from openai import AzureOpenAI
            
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
            print("✅ Azure OpenAI client initialized successfully")
            
        except Exception as e:
            raise Exception(f"Failed to initialize Azure OpenAI client: {e}")

    async def generate_summary(self, content: List[str]) -> str:
        """
        Generate a concise, well-formatted summary of the provided content.
        """
        summary_prompt = (
            "Summarize the following conversation history concisely, preserving key points and tasks. "
            "Format the summary with bullet points or numbered lists for clarity:\n\n"
            + "\n".join(content[:10])  # Limit to first 10 for efficiency
        )
        response = await self.chat_completion(
            [{"role": "user", "content": summary_prompt}],
            task_type="chat",
            temperature=0.2,
            max_tokens=500
        )
        return response.content

    async def chat_completion(
        self, 
        messages: List[Dict],
        task_type: str = "chat",  # "chat", "intent"
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Get chat completion from Azure OpenAI
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            task_type: "chat" for main responses, "intent" for classification
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMResponse with content, tokens, and model info
        """
        
        # Select model based on task type
        model_name = self.intent_model if task_type == "intent" else self.chat_model
        
        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = 4000 if "gpt-4o" in model_name else 2000
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                model_used=model_name
            )
            
        except Exception as e:
            raise Exception(f"Azure OpenAI API error: {str(e)}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding from Azure OpenAI (replaces Cohere)
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error getting Azure OpenAI embedding: {e}")
            return []
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch (more efficient)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            print(f"Error getting batch embeddings: {e}")
            return []
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.encoding.encode(text))
        except:
            return len(text) // 4  # Rough approximation
    
    def get_status(self) -> Dict:
        """Get client status and configuration"""
        return {
            "provider": "Azure OpenAI",
            "models": {
                "chat": self.chat_model,
                "intent": self.intent_model,
                "embedding": self.embedding_model
            },
            "endpoint": self.endpoint.replace(self.api_key, "***") if self.endpoint else None,
            "api_version": self.api_version
        }

# Example usage
if __name__ == "__main__":
    async def test_client():
        client = AzureOpenAIClient()
        
        print("🧪 Testing Azure OpenAI Client")
        print("Status:", client.get_status())
        
        # Test chat completion
        messages = [{"role": "user", "content": "Hello! What's 2+2?"}]
        
        try:
            response = await client.chat_completion(messages, task_type="chat")
            print(f"✅ Chat Response: {response.content[:100]}...")
            print(f"📊 Tokens: {response.tokens_used}, Model: {response.model_used}")
        except Exception as e:
            print(f"❌ Chat Error: {e}")
        
        # Test embedding
        try:
            embedding = client.get_embedding("Hello world")
            print(f"✅ Embedding: {len(embedding)} dimensions")
        except Exception as e:
            print(f"❌ Embedding Error: {e}")
    
    asyncio.run(test_client())
