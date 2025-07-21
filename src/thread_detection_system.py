"""
ROBUST Thread Detection System with Azure OpenAI and Token Management

Enhanced features:
- DBSCAN clustering for semantic thread detection
- Token-aware embedding handling (fixes truncation issues)
- Thread summaries for optimization
- Robust context selection with token limits
- Automatic summarization when context exceeds limits
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio

# Install scikit-learn if not already installed: pip install scikit-learn
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

class ThreadStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"  # Simple lifecycle: archive inactive threads
    COMPLETED = "completed"  # For API compatibility

@dataclass
class ConversationThread:
    """Enhanced thread representation with summaries and cached embeddings"""
    thread_id: str
    messages: List[int]  # Message indices
    topic: str
    status: ThreadStatus
    created_at: datetime
    last_activity: datetime
    confidence_score: float = 0.8
    summary: str = ""  # Add this for compressed representation
    cached_embedding: Optional[List[float]] = None  # Cache thread embedding
    
    def to_dict(self) -> Dict:
        return {
            "thread_id": self.thread_id,
            "messages": self.messages,
            "topic": self.topic,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "confidence_score": self.confidence_score,
            "summary": self.summary,
            "cached_embedding": self.cached_embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            thread_id=data["thread_id"],
            messages=data["messages"],
            topic=data["topic"],
            status=ThreadStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            confidence_score=data.get("confidence_score", 0.8),
            summary=data.get("summary", ""),
            cached_embedding=data.get("cached_embedding")
        )

class RobustTopicDetector:
    def __init__(self, azure_client, similarity_threshold=0.7, max_tokens=6000, 
                 dbscan_eps=0.5, dbscan_min_samples=3, redis_client=None):
        self.azure_client = azure_client
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.redis_client = redis_client
        
        # Metrics tracking
        self.metrics = {
            "clustering_successes": 0,
            "clustering_fallbacks": 0,
            "embedding_failures": 0,
            "threads_created": 0
        }

    def _get_metrics_key(self, conv_id: str = "global") -> str:
        """Get Redis key for storing metrics"""
        return f"thread_metrics:{conv_id}"

    async def _log_metrics(self, conv_id: str = "global"):
        """Log metrics to Redis for monitoring"""
        if self.redis_client:
            try:
                key = self._get_metrics_key(conv_id)
                self.redis_client.hset(key, mapping=self.metrics)
            except Exception as e:
                print(f"Failed to log metrics: {e}")

    def _safe_get_embedding(self, text: str) -> List[float]:
        """Get embedding with token-aware handling and caching"""
        try:
            # Check for cached embedding first
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_key = f"embedding_cache:{text_hash}"
            
            if self.redis_client:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            
            token_count = self.azure_client.count_tokens(text)
            
            if token_count > self.max_tokens:
                print(f"⚠️ Text has {token_count} tokens, applying smart truncation...")
                text = self._smart_truncate(text, self.max_tokens)
            
            embedding = self.azure_client.get_embedding(text)
            
            # Cache the embedding
            if self.redis_client and embedding:
                self.redis_client.setex(cache_key, 3600, json.dumps(embedding))  # 1 hour cache
            
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding: {e}")
            self.metrics["embedding_failures"] += 1
            return []

    def _validate_embeddings(self, embeddings: List[List[float]], min_valid_ratio=0.5) -> bool:
        """Check if we have enough valid embeddings for clustering"""
        valid_embeddings = [emb for emb in embeddings if emb and len(emb) > 0]
        ratio = len(valid_embeddings) / len(embeddings) if embeddings else 0
        return ratio >= min_valid_ratio and len(valid_embeddings) >= 6

    async def create_threads(self, messages: List[Dict], conv_id: str = "unknown") -> List[ConversationThread]:
        if len(messages) < 4:
            return []

        # Embed messages (with chunking for long ones)
        embeddings = []
        valid_indices = []
        
        for i, msg in enumerate(messages):
            content = msg.get("content", "").strip()
            if not content:  # Skip empty messages
                continue
                
            embedding = self._safe_get_embedding(content)
            if embedding:
                embeddings.append(embedding)
                valid_indices.append(i)

        # Validate embeddings before proceeding
        if not self._validate_embeddings(embeddings):
            print(f"⚠️ Insufficient valid embeddings ({len(embeddings)}/{len(messages)}), falling back to simple grouping")
            self.metrics["clustering_fallbacks"] += 1
            threads = await self._create_simple_threads(messages)
            await self._log_metrics(conv_id)
            return threads

        if not SKLEARN_AVAILABLE or len(embeddings) < 6:
            # Fallback to simple grouping
            self.metrics["clustering_fallbacks"] += 1
            threads = await self._create_simple_threads(messages)
            await self._log_metrics(conv_id)
            return threads

        # Auto-tune DBSCAN parameters based on data
        eps = self._auto_tune_eps(embeddings)
        
        # Cluster embeddings semantically
        embeddings_array = np.array(embeddings)
        clustering = DBSCAN(eps=eps, min_samples=self.dbscan_min_samples, metric="cosine")
        labels = clustering.fit_predict(embeddings_array)

        # Group into threads
        threads = []
        thread_id = 1
        
        for label in set(labels):
            if label == -1: continue  # Noise points (outliers)
            
            # Map back to original message indices
            cluster_indices = [valid_indices[i] for i, l in enumerate(labels) if l == label]
            
            if len(cluster_indices) < 3: continue

            # Generate topic via LLM summary of first few
            topic_text = " ".join([messages[i]["content"][:200] for i in cluster_indices[:3]])
            topic = await self._llm_call("Generate a concise topic for this: " + topic_text)

            # Summarize thread for token efficiency with better prompting
            summary = await self._generate_enhanced_summary([messages[i] for i in cluster_indices])

            # Create thread with cached embedding
            thread_text = summary or " ".join([messages[i]["content"][:100] for i in cluster_indices[:3]])
            thread_embedding = self._safe_get_embedding(thread_text)

            thread = ConversationThread(
                thread_id=f"thread_{thread_id}",
                messages=cluster_indices,
                topic=topic,
                status=ThreadStatus.ACTIVE,
                created_at=datetime.now() - timedelta(days=1),
                last_activity=datetime.now(),
                summary=summary,
                cached_embedding=thread_embedding
            )
            threads.append(thread)
            thread_id += 1

        self.metrics["clustering_successes"] += 1
        self.metrics["threads_created"] += len(threads)
        await self._log_metrics(conv_id)
        
        print(f"✅ Created {len(threads)} clustered threads with auto-tuned eps={eps:.3f}")
        return threads

    def _auto_tune_eps(self, embeddings: List[List[float]]) -> float:
        """Auto-tune DBSCAN eps parameter based on embedding variance"""
        try:
            embeddings_array = np.array(embeddings)
            # Calculate pairwise distances
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(embeddings_array)
            
            # Use median distance as eps (with bounds)
            median_distance = np.median(distances[distances > 0])
            eps = max(0.3, min(0.8, median_distance))
            
            return eps
        except Exception:
            return self.dbscan_eps  # Fallback to default

    async def _generate_enhanced_summary(self, messages: List[Dict]) -> str:
        """Generate structured summary preserving key information"""
        try:
            if not messages:
                return ""
            
            # Combine messages with smart truncation
            combined = ' '.join([msg.get("content", "") for msg in messages])
            if self.azure_client.count_tokens(combined) > 2000:
                combined = self._smart_truncate(combined, 2000)
            
            # Enhanced prompt for better summaries
            prompt = f"""Summarize this conversation thread in 2-3 sentences with the following structure:
• Main topic/task discussed
• Key technical details or decisions made  
• Current status or next steps

Conversation:
{combined}

Summary:"""
            
            messages_for_llm = [
                {"role": "system", "content": "Create structured summaries that preserve key technical details and action items."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.azure_client.chat_completion(
                messages=messages_for_llm,
                task_type="intent",
                temperature=0.3,
                max_tokens=150  # Slightly more tokens for structured output
            )
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating enhanced summary: {e}")
            return "Thread summary unavailable"

    async def _create_simple_threads(self, messages: List[Dict]) -> List[ConversationThread]:
        """Fallback simple threading"""
        threads = []
        thread_id = 1
        cluster_size = 8
        
        for i in range(0, len(messages), cluster_size):
            end_idx = min(i + cluster_size, len(messages))
            if end_idx - i < 3:
                continue
                
            message_indices = list(range(i, end_idx))
            topic_text = " ".join([messages[idx]["content"][:200] for idx in message_indices[:3]])
            topic = await self._llm_call("Generate a concise topic for this: " + topic_text)
            summary = await self._generate_summary([messages[idx] for idx in message_indices])
            
            thread = ConversationThread(
                thread_id=f"thread_{thread_id}",
                messages=message_indices,
                topic=topic,
                status=ThreadStatus.ACTIVE,
                created_at=datetime.now() - timedelta(minutes=thread_id*5),
                last_activity=datetime.now(),
                summary=summary
            )
            threads.append(thread)
            thread_id += 1
            
        return threads

    async def _llm_call(self, prompt: str) -> str:
        """Generate topic using Azure OpenAI"""
        try:
            messages = [
                {"role": "system", "content": "Generate concise topics for conversation segments. Respond with just the topic, 2-4 words max."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.azure_client.chat_completion(
                messages=messages,
                task_type="intent",
                temperature=0.3,
                max_tokens=50
            )
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating topic: {e}")
            return "General Discussion"

    async def _generate_summary(self, messages: List[Dict]) -> str:
        """Generate thread summary"""
        try:
            content = " ".join([msg["content"] for msg in messages])[:2000]  # Limit length
            
            prompt = f"Summarize this conversation thread in 1-2 sentences: {content}"
            
            messages_for_llm = [
                {"role": "system", "content": "Summarize conversation threads concisely, preserving key technical details."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.azure_client.chat_completion(
                messages=messages_for_llm,
                task_type="intent",
                temperature=0.3,
                max_tokens=100
            )
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Thread summary unavailable"

class RobustThreadContextSelector:
    def __init__(self, base_manager, azure_client, intent_classifier,
                 continuation_threshold=0.85, semantic_threshold=0.75,
                 max_threads_to_check=10, intent_confidence_threshold=0.7):
        self.base_manager = base_manager
        self.azure_client = azure_client
        self.intent_classifier = intent_classifier
        
        # Configurable thresholds
        self.continuation_threshold = continuation_threshold
        self.semantic_threshold = semantic_threshold
        self.max_threads_to_check = max_threads_to_check
        self.intent_confidence_threshold = intent_confidence_threshold
        
        # Metrics tracking
        self.metrics = {
            "continuation_strategy_used": 0,
            "semantic_strategy_used": 0,
            "hybrid_strategy_used": 0,
            "compression_events": 0,
            "avg_similarity_scores": [],
            "token_savings": 0
        }

    async def select_thread_context(
        self, 
        conv_id: str, 
        threads: List[ConversationThread], 
        user_message: str, 
        intent_result=None,
        token_limit=100000
    ) -> Tuple[List[Dict], str]:
        """
        Enhanced context selection with confidence checks and hybrid strategies
        """
        conversation = self.base_manager.get_conversation(conv_id)
        
        if not threads:
            return self._get_recent_context(conversation, user_message), "recent_fallback"

        # Classify intent if not provided
        if intent_result is None:
            recent_context = conversation[-6:] if conversation else []
            intent_result = await self.intent_classifier.classify_intent(user_message, recent_context, conv_id)

        # Enhanced intent handling with confidence checks
        strategy_decision = self._determine_strategy(intent_result, user_message)
        
        if strategy_decision == "continuation":
            self.metrics["continuation_strategy_used"] += 1
            context_msgs, strategy = await self._continuation_strategy(
                conversation, threads, user_message, token_limit
            )
        elif strategy_decision == "hybrid":
            self.metrics["hybrid_strategy_used"] += 1
            context_msgs, strategy = await self._hybrid_strategy(
                conversation, threads, user_message, intent_result, token_limit
            )
        else:  # semantic
            self.metrics["semantic_strategy_used"] += 1
            context_msgs, strategy = await self._semantic_strategy(
                conversation, threads, user_message, token_limit
            )

        return context_msgs, strategy

    def _determine_strategy(self, intent_result, user_message: str) -> str:
        """
        Enhanced strategy determination with confidence checks and hybrid fallbacks
        """
        # Check intent confidence
        if intent_result.confidence < self.intent_confidence_threshold:
            print(f"⚠️ Low intent confidence ({intent_result.confidence:.2f}), using hybrid strategy")
            return "hybrid"
        
        # High confidence continuation
        if intent_result.intent.value == "CONTINUATION" and intent_result.confidence >= 0.8:
            return "continuation"
        
        # Medium confidence continuation - use hybrid
        if intent_result.intent.value == "CONTINUATION" and intent_result.confidence >= self.intent_confidence_threshold:
            return "hybrid"
        
        # All other high-confidence intents use semantic
        return "semantic"

    async def _hybrid_strategy(
        self,
        conversation: List[Dict],
        threads: List[ConversationThread], 
        user_message: str,
        intent_result,
        token_limit: int
    ) -> Tuple[List[Dict], str]:
        """
        Hybrid strategy: Blend recent context with semantic relevance
        """
        print("🔀 Using HYBRID strategy - blending recent + semantic context")
        
        # Get both types of context
        recent_context, _ = await self._continuation_strategy(
            conversation, threads, user_message, int(token_limit * 0.6)  # 60% for recent
        )
        
        semantic_context, _ = await self._semantic_strategy(
            conversation, threads, user_message, int(token_limit * 0.4)  # 40% for semantic
        )
        
        # Merge contexts intelligently
        context_msgs = []
        
        # Add system message combining both approaches
        context_msgs.append({
            "role": "system",
            "content": f"Context: Continuing recent work (intent: {intent_result.intent.value}, confidence: {intent_result.confidence:.2f}) with relevant background."
        })
        
        # Take recent messages from continuation strategy (remove system msg)
        recent_msgs = [msg for msg in recent_context if msg.get("role") != "system"][-8:]
        
        # Take semantic thread summary from semantic strategy  
        semantic_system = [msg for msg in semantic_context if msg.get("role") == "system"]
        if semantic_system:
            context_msgs.extend(semantic_system[:1])  # Just the first system message
        
        # Add recent messages
        context_msgs.extend(recent_msgs)
        
        # Add user message
        context_msgs.append({"role": "user", "content": user_message})
        
        # Token check
        total_tokens = sum(self.azure_client.count_tokens(msg.get("content", "")) for msg in context_msgs)
        strategy = "hybrid_balanced"
        
        if total_tokens > token_limit * 0.8:
            context_msgs = await self._compress_hybrid_context(context_msgs, user_message, token_limit)
            strategy = "hybrid_compressed"
        
        print(f"✅ Hybrid context: {len(context_msgs)} messages, ~{total_tokens} tokens")
        return context_msgs, strategy

    async def _continuation_strategy(
        self, 
        conversation: List[Dict], 
        threads: List[ConversationThread], 
        user_message: str,
        token_limit: int
    ) -> Tuple[List[Dict], str]:
        """
        Enhanced continuation strategy with dynamic thresholds
        """
        print("🔄 Using CONTINUATION strategy - prioritizing recent context")
        
        # Dynamic threshold based on conversation length
        threshold = self._get_dynamic_threshold(len(conversation), "continuation")
        
        # Use last 2-3 active threads (temporal approach)
        recent_threads = [t for t in threads if t.status.value == "ACTIVE"][-3:]
        
        context_msgs = []
        strategy = "continuation_temporal"
        
        # Add lightweight system message
        context_msgs.append({
            "role": "system", 
            "content": "Continue the ongoing conversation naturally, building on recent context."
        })
        
        # Enhanced message selection based on conversation length
        if len(conversation) <= 20:
            # Short conversation: include everything
            context_msgs.extend(conversation)
            strategy += "_full"
            
        elif len(conversation) <= 50:
            # Medium conversation: recent + key thread messages  
            recent_thread_indices = set()
            for thread in recent_threads:
                recent_thread_indices.update(thread.messages[-10:])  # More from recent threads
            
            # Combine with global recent messages
            global_recent_start = max(0, len(conversation) - 18)  # Increased from 15
            combined_indices = recent_thread_indices.union(
                set(range(global_recent_start, len(conversation)))
            )
            
            # Sort by original order and add to context
            for i in sorted(combined_indices):
                if i < len(conversation):
                    context_msgs.append(conversation[i])
            strategy += "_hybrid"
            
        else:
            # Long conversation: aggressive recent focus + summaries
            context_msgs.extend(await self._build_long_continuation_context(
                conversation, recent_threads, token_limit
            ))
            strategy += "_compressed"
        
        # Add user message
        context_msgs.append({"role": "user", "content": user_message})
        
        # Enhanced token check with dynamic threshold
        total_tokens = sum(self.azure_client.count_tokens(msg.get("content", "")) for msg in context_msgs)
        
        if total_tokens > token_limit * threshold:
            print(f"⚠️ Continuation context too long ({total_tokens} tokens), applying compression")
            original_tokens = total_tokens
            context_msgs = await self._compress_continuation_context(context_msgs, user_message, token_limit)
            new_tokens = sum(self.azure_client.count_tokens(msg.get("content", "")) for msg in context_msgs)
            self.metrics["compression_events"] += 1
            self.metrics["token_savings"] += (original_tokens - new_tokens)
            strategy += "_final_compressed"
        
        print(f"✅ Continuation context: {len(context_msgs)} messages, ~{total_tokens} tokens")
        return context_msgs, strategy

    async def _semantic_strategy(
        self,
        conversation: List[Dict],
        threads: List[ConversationThread], 
        user_message: str,
        token_limit: int
    ) -> Tuple[List[Dict], str]:
        """
        Enhanced semantic strategy with thread capping and cached embeddings
        """
        print("🎯 Using SEMANTIC strategy - finding most relevant historical context")
        
        # Cap threads to avoid performance issues
        threads_to_check = threads[-self.max_threads_to_check:] if len(threads) > self.max_threads_to_check else threads
        
        # Get user embedding
        user_emb = await asyncio.to_thread(self.azure_client.get_embedding, user_message)

        # Find most relevant thread with cached embeddings
        similarities = []
        for thread in threads_to_check:
            # Use cached embedding if available
            if thread.cached_embedding:
                thread_emb = thread.cached_embedding
            else:
                # Generate and cache embedding
                if thread.summary:
                    thread_text = thread.summary
                else:
                    thread_msgs = [conversation[i].get("content", "") for i in thread.messages[:3] if i < len(conversation)]
                    thread_text = " ".join(thread_msgs)[:1000]
                
                if thread_text.strip():
                    thread_emb = await asyncio.to_thread(self.azure_client.get_embedding, thread_text)
                    # Update cached embedding
                    thread.cached_embedding = thread_emb
                else:
                    thread_emb = []
            
            if thread_emb:
                sim = await asyncio.to_thread(self._cosine_similarity, user_emb, thread_emb)
                similarities.append(sim)
            else:
                similarities.append(0.0)

        best_thread_idx = np.argmax(similarities) if similarities else 0
        best_thread = threads_to_check[best_thread_idx]
        best_similarity = similarities[best_thread_idx] if similarities else 0.0
        
        # Track similarity metrics
        self.metrics["avg_similarity_scores"].append(best_similarity)
        if len(self.metrics["avg_similarity_scores"]) > 100:  # Keep last 100
            self.metrics["avg_similarity_scores"] = self.metrics["avg_similarity_scores"][-100:]

        print(f"🎯 Selected thread '{best_thread.topic}' (similarity: {best_similarity:.3f})")

        # Build semantic context with dynamic threshold
        threshold = self._get_dynamic_threshold(len(conversation), "semantic")
        context_msgs = []
        strategy = "semantic"
        
        # Add thread summary as system context
        if best_thread.summary:
            context_msgs.append({
                "role": "system", 
                "content": f"Relevant context from '{best_thread.topic}' (similarity: {best_similarity:.2f}): {best_thread.summary}"
            })
            strategy += "_with_summary"
        
        # Add key messages from selected thread (increased from 8 to 10)
        thread_messages = []
        for idx in best_thread.messages[-10:]:
            if idx < len(conversation):
                thread_messages.append(conversation[idx])
        
        context_msgs.extend(thread_messages)
        
        # Add some recent global context (increased from 6 to 8)
        recent_global = conversation[-8:]
        
        # Avoid duplicates
        used_indices = set(best_thread.messages[-10:])
        recent_start = max(0, len(conversation) - 8)
        for i, msg in enumerate(conversation[recent_start:], start=recent_start):
            if i not in used_indices:
                context_msgs.append(msg)
        
        # Add user message
        context_msgs.append({"role": "user", "content": user_message})
        
        # Token check with dynamic threshold
        total_tokens = sum(self.azure_client.count_tokens(msg.get("content", "")) for msg in context_msgs)
        
        if total_tokens > token_limit * threshold:
            print(f"⚠️ Semantic context too long ({total_tokens} tokens), applying compression")
            original_tokens = total_tokens
            context_msgs = await self._compress_semantic_context(context_msgs, best_thread, user_message, token_limit)
            new_tokens = sum(self.azure_client.count_tokens(msg.get("content", "")) for msg in context_msgs)
            self.metrics["compression_events"] += 1
            self.metrics["token_savings"] += (original_tokens - new_tokens)
            strategy += "_compressed"
        
        print(f"✅ Semantic context: {len(context_msgs)} messages, ~{total_tokens} tokens")
        return context_msgs, strategy

    def _get_dynamic_threshold(self, conversation_length: int, strategy_type: str) -> float:
        """
        Calculate dynamic token thresholds based on conversation length
        """
        base_threshold = self.continuation_threshold if strategy_type == "continuation" else self.semantic_threshold
        
        # Looser thresholds for shorter conversations
        if conversation_length <= 20:
            return min(0.95, base_threshold + 0.1)
        elif conversation_length <= 50:
            return base_threshold + 0.05
        else:
            return base_threshold

    async def _compress_hybrid_context(
        self,
        context_msgs: List[Dict],
        user_message: str,
        token_limit: int
    ) -> List[Dict]:
        """Compress hybrid context maintaining balance"""
        
        system_msgs = [msg for msg in context_msgs if msg.get("role") == "system"]
        conversation_msgs = [msg for msg in context_msgs if msg.get("role") in ["user", "assistant"]]
        
        # Keep more recent messages for hybrid (balance between continuation and semantic)
        keep_recent = conversation_msgs[-5:] if len(conversation_msgs) > 5 else conversation_msgs
        to_compress = conversation_msgs[:-5] if len(conversation_msgs) > 5 else []
        
        compressed_context = system_msgs.copy()
        
        if to_compress:
            summary = await self._generate_summary_for_context([{"role": "system", "content": " ".join([f"{msg['role']}: {msg['content'][:200]}" for msg in to_compress])}])
            compressed_context.append({
                "role": "system",
                "content": f"Context summary (hybrid approach): {summary}"
            })
        
        compressed_context.extend(keep_recent)
        compressed_context.append({"role": "user", "content": user_message})
        
        return compressed_context

    async def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Async-wrapped cosine similarity calculation"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception:
            return 0.0

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for monitoring"""
        avg_similarity = np.mean(self.metrics["avg_similarity_scores"]) if self.metrics["avg_similarity_scores"] else 0.0
        
        return {
            **self.metrics,
            "avg_similarity_score": round(avg_similarity, 3),
            "compression_rate": round(self.metrics["compression_events"] / max(1, sum([
                self.metrics["continuation_strategy_used"],
                self.metrics["semantic_strategy_used"], 
                self.metrics["hybrid_strategy_used"]
            ])) * 100, 2)
        }

    async def _continuation_strategy(
        self, 
        conversation: List[Dict], 
        threads: List[ConversationThread], 
        user_message: str,
        token_limit: int
    ) -> Tuple[List[Dict], str]:
        """
        Continuation strategy: Prioritize recent context and temporal flow
        """
        print("🔄 Using CONTINUATION strategy - prioritizing recent context")
        
        # For continuation, we want MORE recent context
        # Use last 2-3 active threads (temporal approach)
        recent_threads = [t for t in threads if t.status.value == "ACTIVE"][-3:]
        
        context_msgs = []
        strategy = "continuation_temporal"
        
        # Add lightweight system message
        context_msgs.append({
            "role": "system", 
            "content": "Continue the ongoing conversation naturally, building on recent context."
        })
        
        # Prioritize recent messages with smart token allocation
        if len(conversation) <= 20:
            # Short conversation: include everything
            context_msgs.extend(conversation)
            strategy += "_full"
            
        elif len(conversation) <= 50:
            # Medium conversation: recent + key thread messages
            # Get messages from recent threads
            recent_thread_indices = set()
            for thread in recent_threads:
                recent_thread_indices.update(thread.messages[-8:])  # More from recent threads
            
            # Combine with global recent messages
            global_recent_start = max(0, len(conversation) - 15)
            combined_indices = recent_thread_indices.union(
                set(range(global_recent_start, len(conversation)))
            )
            
            # Sort by original order and add to context
            for i in sorted(combined_indices):
                if i < len(conversation):
                    context_msgs.append(conversation[i])
            strategy += "_hybrid"
            
        else:
            # Long conversation: aggressive recent focus + summaries
            context_msgs.extend(await self._build_long_continuation_context(
                conversation, recent_threads, token_limit
            ))
            strategy += "_compressed"
        
        # Add user message
        context_msgs.append({"role": "user", "content": user_message})
        
        # Final token check
        total_tokens = sum(self.azure_client.count_tokens(msg.get("content", "")) for msg in context_msgs)
        
        if total_tokens > token_limit * 0.85:  # 85% threshold for continuation
            print(f"⚠️ Continuation context too long ({total_tokens} tokens), applying compression")
            context_msgs = await self._compress_continuation_context(context_msgs, user_message, token_limit)
            strategy += "_final_compressed"
        
        print(f"✅ Continuation context: {len(context_msgs)} messages, ~{total_tokens} tokens")
        return context_msgs, strategy

    async def _semantic_strategy(
        self,
        conversation: List[Dict],
        threads: List[ConversationThread], 
        user_message: str,
        token_limit: int
    ) -> Tuple[List[Dict], str]:
        """
        Semantic strategy: Find most relevant thread using similarity
        """
        print("🎯 Using SEMANTIC strategy - finding most relevant historical context")
        
        # Embed user message for similarity search
        user_emb = self.azure_client.get_embedding(user_message)

        # Find most relevant thread (cosine similarity)
        similarities = []
        for thread in threads:
            if thread.summary:
                thread_text = thread.summary
            else:
                # Fallback to first few messages
                thread_msgs = [conversation[i].get("content", "") for i in thread.messages[:3] if i < len(conversation)]
                thread_text = " ".join(thread_msgs)[:1000]
            
            if thread_text.strip():
                thread_emb = self.azure_client.get_embedding(thread_text)
                sim = self._cosine_similarity(user_emb, thread_emb)
                similarities.append(sim)
            else:
                similarities.append(0.0)

        best_thread_idx = np.argmax(similarities) if similarities else 0
        best_thread = threads[best_thread_idx]
        best_similarity = similarities[best_thread_idx] if similarities else 0.0

        print(f"🎯 Selected thread '{best_thread.topic}' (similarity: {best_similarity:.3f})")

        # Build semantic context
        context_msgs = []
        strategy = "semantic"
        
        # Add thread summary as system context
        if best_thread.summary:
            context_msgs.append({
                "role": "system", 
                "content": f"Relevant context from '{best_thread.topic}': {best_thread.summary}"
            })
            strategy += "_with_summary"
        
        # Add key messages from selected thread
        thread_messages = []
        for idx in best_thread.messages[-8:]:  # More messages for semantic queries
            if idx < len(conversation):
                thread_messages.append(conversation[idx])
        
        context_msgs.extend(thread_messages)
        
        # Add some recent global context (fewer than continuation)
        recent_global = conversation[-6:]  # Fewer recent messages
        
        # Avoid duplicates
        used_indices = set(best_thread.messages[-8:])
        recent_start = max(0, len(conversation) - 6)
        for i, msg in enumerate(conversation[recent_start:], start=recent_start):
            if i not in used_indices:
                context_msgs.append(msg)
        
        # Add user message
        context_msgs.append({"role": "user", "content": user_message})
        
        # Token check with different threshold for semantic queries
        total_tokens = sum(self.azure_client.count_tokens(msg.get("content", "")) for msg in context_msgs)
        
        if total_tokens > token_limit * 0.75:  # 75% threshold for semantic (more compression)
            print(f"⚠️ Semantic context too long ({total_tokens} tokens), applying compression")
            context_msgs = await self._compress_semantic_context(context_msgs, best_thread, user_message, token_limit)
            strategy += "_compressed"
        
        print(f"✅ Semantic context: {len(context_msgs)} messages, ~{total_tokens} tokens")
        return context_msgs, strategy

    async def _build_long_continuation_context(
        self,
        conversation: List[Dict],
        recent_threads: List[ConversationThread],
        token_limit: int
    ) -> List[Dict]:
        """Build context for long conversations using continuation strategy"""
        
        context_msgs = []
        
        # Add compact summaries of recent threads (not the best one, but recent ones)
        for thread in recent_threads[:-1]:  # All but the most recent
            if thread.summary:
                context_msgs.append({
                    "role": "system",
                    "content": f"Previous context - {thread.topic}: {thread.summary[:200]}..."
                })
        
        # Add more detailed context from the most recent thread
        if recent_threads:
            latest_thread = recent_threads[-1]
            latest_messages = []
            for idx in latest_thread.messages[-10:]:  # More from latest thread
                if idx < len(conversation):
                    latest_messages.append(conversation[idx])
            context_msgs.extend(latest_messages)
        
        # Add very recent global messages
        context_msgs.extend(conversation[-8:])
        
        return context_msgs

    async def _compress_continuation_context(
        self,
        context_msgs: List[Dict],
        user_message: str,
        token_limit: int
    ) -> List[Dict]:
        """Compress context while preserving continuation flow"""
        
        # Keep system messages and very recent messages
        system_msgs = [msg for msg in context_msgs if msg.get("role") == "system"]
        conversation_msgs = [msg for msg in context_msgs if msg.get("role") in ["user", "assistant"]]
        
        # Keep last 6 messages as-is for continuation
        keep_recent = conversation_msgs[-6:] if len(conversation_msgs) > 6 else conversation_msgs
        to_compress = conversation_msgs[:-6] if len(conversation_msgs) > 6 else []
        
        compressed_context = system_msgs.copy()
        
        if to_compress:
            # Summarize older part
            summary_text = "\n".join([f"{msg['role']}: {msg['content'][:300]}" for msg in to_compress])
            summary = await self._generate_summary_for_context([{"role": "system", "content": summary_text}])
            
            compressed_context.append({
                "role": "system",
                "content": f"Earlier conversation summary (for continuation): {summary}"
            })
        
        # Add recent messages
        compressed_context.extend(keep_recent)
        compressed_context.append({"role": "user", "content": user_message})
        
        return compressed_context

    async def _compress_semantic_context(
        self,
        context_msgs: List[Dict],
        selected_thread: ConversationThread,
        user_message: str,
        token_limit: int
    ) -> List[Dict]:
        """Compress context for semantic queries (more aggressive)"""
        
        # Keep system message and compress everything else
        system_msgs = [msg for msg in context_msgs if msg.get("role") == "system"]
        conversation_msgs = [msg for msg in context_msgs if msg.get("role") in ["user", "assistant"]]
        
        # For semantic queries, keep fewer recent messages
        keep_recent = conversation_msgs[-3:]
        to_compress = conversation_msgs[:-3] if len(conversation_msgs) > 3 else []
        
        compressed_context = system_msgs.copy()
        
        if to_compress:
            # More aggressive summarization for semantic queries
            summary_text = f"Thread topic: {selected_thread.topic}\n"
            summary_text += "\n".join([f"{msg['role']}: {msg['content'][:200]}" for msg in to_compress])
            
            summary = await self._generate_summary_for_context([{"role": "system", "content": summary_text}])
            
            compressed_context.append({
                "role": "system", 
                "content": f"Relevant thread summary: {summary}"
            })
        
        compressed_context.extend(keep_recent)
        compressed_context.append({"role": "user", "content": user_message})
        
        return compressed_context

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception:
            return 0.0

    def _get_recent_context(self, conversation: List[Dict], user_message: str) -> List[Dict]:
        """Fallback to recent messages"""
        recent_messages = conversation[-10:] if conversation else []
        
        system_msg = {
            "role": "system",
            "content": "You are a helpful assistant. Provide thoughtful responses based on the conversation history."
        }
        
        context = [system_msg]
        context.extend(recent_messages)
        context.append({"role": "user", "content": user_message})
        
        return context

    async def _generate_summary_for_context(self, messages: List[Dict]) -> str:
        """Generate summary for context compression"""
        try:
            content = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])[:2000]
            
            prompt = f"Summarize this conversation preserving key context: {content}"
            
            response = await self.azure_client.chat_completion(
                messages=[
                    {"role": "system", "content": "Create concise summaries preserving technical context."},
                    {"role": "user", "content": prompt}
                ],
                task_type="intent",
                temperature=0.3,
                max_tokens=300
            )
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating context summary: {e}")
            return "Previous conversation context"

class SimplifiedThreadLifecycleManager:
    """Thread persistence manager"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
    
    def get_threads_key(self, conv_id: str) -> str:
        return f"threads:{conv_id}"
    
    def load_threads(self, conv_id: str) -> List[ConversationThread]:
        """Load threads from Redis"""
        try:
            data = self.redis_client.get(self.get_threads_key(conv_id))
            if data:
                threads_data = json.loads(data)
                return [ConversationThread.from_dict(t) for t in threads_data]
            return []
        except Exception as e:
            print(f"Error loading threads: {e}")
            return []
    
    def save_threads(self, conv_id: str, threads: List[ConversationThread]):
        """Save threads to Redis"""
        try:
            threads_data = [thread.to_dict() for thread in threads]
            self.redis_client.set(
                self.get_threads_key(conv_id),
                json.dumps(threads_data)
            )
        except Exception as e:
            print(f"Error saving threads: {e}")

class ThreadAwareConversationManager:
    """
    Enhanced Thread-aware conversation manager with configurable parameters and monitoring
    """
    
    def __init__(self, base_manager, azure_client, redis_client, intent_classifier=None,
                 # Configurable parameters
                 dbscan_eps=0.5, dbscan_min_samples=3, 
                 continuation_threshold=0.85, semantic_threshold=0.75,
                 max_threads_to_check=10, intent_confidence_threshold=0.7):
        self.base_manager = base_manager
        self.azure_client = azure_client
        self.intent_classifier = intent_classifier
        self.redis_client = redis_client
        
        # Enhanced components with configurable parameters
        self.boundary_detector = RobustTopicDetector(
            azure_client, 
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            redis_client=redis_client
        )
        self.lifecycle_manager = SimplifiedThreadLifecycleManager(redis_client)
        self.context_selector = RobustThreadContextSelector(
            base_manager, 
            azure_client, 
            intent_classifier,
            continuation_threshold=continuation_threshold,
            semantic_threshold=semantic_threshold,
            max_threads_to_check=max_threads_to_check,
            intent_confidence_threshold=intent_confidence_threshold
        )
    
    def add_exchange_with_threads(self, conv_id: str, user_message: str, assistant_message: str):
        """Add exchange with enhanced thread handling"""
        
        # Add exchange using base manager
        self.base_manager.add_exchange(conv_id, user_message, assistant_message)
        
        print(f"✅ Added exchange for conversation {conv_id} with robust thread handling")
        return True
    
    async def analyze_conversation_threads(self, conv_id: str) -> List[ConversationThread]:
        """Enhanced thread analysis with clustering and metrics"""
        
        conversation = self.base_manager.get_conversation(conv_id)
        
        if len(conversation) < 4:
            return []
        
        threads = await self.boundary_detector.create_threads(conversation, conv_id)
        
        print(f"📊 Created {len(threads)} robust threads for conversation {conv_id}")
        
        return threads
    
    async def prepare_context_with_threads(
        self, 
        conv_id: str, 
        user_message: str,
        intent_result=None,
        token_limit: int = 100000
    ) -> Tuple[List[Dict], int, str]:
        """
        Enhanced context preparation with intent-aware strategy selection and monitoring
        
        Returns:
            (context, total_tokens, strategy_used)
        """
        
        # Load or create threads
        threads = self.lifecycle_manager.load_threads(conv_id)
        
        if not threads:
            threads = await self.analyze_conversation_threads(conv_id)
            if threads:
                # Save threads with cached embeddings
                self.lifecycle_manager.save_threads(conv_id, threads)
        
        # Select context using intent-aware logic
        context, strategy = await self.context_selector.select_thread_context(
            conv_id, threads, user_message, intent_result, token_limit
        )
        
        # Calculate token usage
        total_tokens = sum(self.azure_client.count_tokens(msg.get("content", "")) for msg in context)
        
        print(f"🔍 Intent-aware context ({strategy}): {len(context)} messages, {total_tokens} tokens")
        
        return context, total_tokens, strategy
    
    def update_threads_with_new_message(self, conv_id: str, message_index: int):
        """No-op for API compatibility"""
        pass

    def get_performance_metrics(self, conv_id: str = "global") -> Dict:
        """
        Get comprehensive performance metrics for monitoring
        """
        # Get context selector metrics
        context_metrics = self.context_selector.get_performance_metrics()
        
        # Get boundary detector metrics from Redis
        boundary_metrics = {}
        if self.redis_client:
            try:
                metrics_key = self.boundary_detector._get_metrics_key(conv_id)
                boundary_metrics = self.redis_client.hgetall(metrics_key)
                # Convert string values back to appropriate types
                for key, value in boundary_metrics.items():
                    try:
                        boundary_metrics[key] = float(value) if '.' in str(value) else int(value)
                    except:
                        pass
            except Exception as e:
                print(f"Error retrieving boundary metrics: {e}")
        
        return {
            "thread_detection": boundary_metrics,
            "context_selection": context_metrics,
            "system_health": {
                "sklearn_available": SKLEARN_AVAILABLE,
                "redis_connected": bool(self.redis_client and self.redis_client.ping()),
                "intent_classifier_available": self.intent_classifier is not None
            }
        }

    def tune_parameters(self, **kwargs):
        """
        Runtime parameter tuning for optimization
        """
        # Update boundary detector parameters
        if 'dbscan_eps' in kwargs:
            self.boundary_detector.dbscan_eps = kwargs['dbscan_eps']
        if 'dbscan_min_samples' in kwargs:
            self.boundary_detector.dbscan_min_samples = kwargs['dbscan_min_samples']
            
        # Update context selector parameters
        if 'continuation_threshold' in kwargs:
            self.context_selector.continuation_threshold = kwargs['continuation_threshold']
        if 'semantic_threshold' in kwargs:
            self.context_selector.semantic_threshold = kwargs['semantic_threshold']
        if 'max_threads_to_check' in kwargs:
            self.context_selector.max_threads_to_check = kwargs['max_threads_to_check']
        if 'intent_confidence_threshold' in kwargs:
            self.context_selector.intent_confidence_threshold = kwargs['intent_confidence_threshold']
            
        print(f"🔧 Updated parameters: {kwargs}")

# Backward compatibility aliases
TopicBoundaryDetector = RobustTopicDetector
SimplifiedTopicDetector = RobustTopicDetector
SimplifiedThreadContextSelector = RobustThreadContextSelector
