#!/usr/bin/env python3
"""
Async Conversation Logger
========================

Provides async logging capabilities for conversation context and strategy tracking.
Logs context selection strategies, intent classifications, and conversation details
without blocking the main thread.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import logging

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class ConversationLogEntry:
    """Single conversation log entry"""
    timestamp: datetime
    conversation_id: str
    message_index: int
    user_message: str
    assistant_response: str
    context_strategy: str
    intent_classification: Dict
    context_details: Dict
    performance_metrics: Dict
    thread_info: Dict
    log_level: LogLevel = LogLevel.INFO
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "conversation_id": self.conversation_id,
            "message_index": self.message_index,
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "context_strategy": self.context_strategy,
            "intent_classification": self.intent_classification,
            "context_details": self.context_details,
            "performance_metrics": self.performance_metrics,
            "thread_info": self.thread_info,
            "log_level": self.log_level.value
        }

class ConversationLogger:
    """Async conversation logger that tracks context strategies and conversation flow"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.ensure_log_directory()
        
        # Initialize async logging components
        self.log_queue = Queue()
        self.worker_thread = None
        self.is_running = False
        
        # Strategy tracking
        self.strategy_stats = {}
        self.intent_accuracy_log = []
        self.context_performance_log = []
        
        # Log files
        self.conversation_log_file = os.path.join(log_dir, f"conversation_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
        self.strategy_stats_file = os.path.join(log_dir, f"strategy_stats_{datetime.now().strftime('%Y%m%d')}.json")
        self.performance_log_file = os.path.join(log_dir, f"performance_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
        
        # Start the async worker
        self.start_worker()
    
    def ensure_log_directory(self):
        """Create log directory if it doesn't exist"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"📁 Created log directory: {self.log_dir}")
    
    def start_worker(self):
        """Start the async logging worker thread"""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("🚀 Async conversation logger started")
    
    def stop_worker(self):
        """Stop the async logging worker thread"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        print("🛑 Async conversation logger stopped")
    
    def _worker_loop(self):
        """Main worker loop that processes log entries"""
        while self.is_running:
            try:
                # Process all queued log entries
                while not self.log_queue.empty():
                    log_entry = self.log_queue.get()
                    self._write_log_entry(log_entry)
                    self._update_strategy_stats(log_entry)
                    self._update_performance_tracking(log_entry)
                
                # Brief sleep to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error in logger worker: {e}")
                time.sleep(1)
    
    def _write_log_entry(self, entry: ConversationLogEntry):
        """Write log entry to file"""
        try:
            with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"❌ Error writing log entry: {e}")
    
    def _update_strategy_stats(self, entry: ConversationLogEntry):
        """Update strategy usage statistics"""
        strategy = entry.context_strategy
        
        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = {
                "usage_count": 0,
                "total_response_time": 0.0,
                "total_tokens": 0,
                "avg_response_time": 0.0,
                "avg_tokens": 0.0,
                "conversations": set()
            }
        
        stats = self.strategy_stats[strategy]
        stats["usage_count"] += 1
        stats["total_response_time"] += entry.performance_metrics.get("response_time", 0)
        stats["total_tokens"] += entry.performance_metrics.get("tokens_used", 0)
        stats["conversations"].add(entry.conversation_id)
        
        # Calculate averages
        stats["avg_response_time"] = stats["total_response_time"] / stats["usage_count"]
        stats["avg_tokens"] = stats["total_tokens"] / stats["usage_count"]
        
        # Save updated stats
        self._save_strategy_stats()
    
    def _save_strategy_stats(self):
        """Save strategy statistics to file"""
        try:
            # Convert set to list for JSON serialization
            stats_for_json = {}
            for strategy, stats in self.strategy_stats.items():
                stats_copy = stats.copy()
                stats_copy["conversations"] = list(stats_copy["conversations"])
                stats_copy["unique_conversations"] = len(stats["conversations"])
                stats_for_json[strategy] = stats_copy
            
            with open(self.strategy_stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "strategy_statistics": stats_for_json,
                    "total_messages": sum(s["usage_count"] for s in self.strategy_stats.values()),
                    "strategies_ranked": sorted(
                        stats_for_json.keys(),
                        key=lambda x: stats_for_json[x]["usage_count"],
                        reverse=True
                    )
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving strategy stats: {e}")
    
    def _update_performance_tracking(self, entry: ConversationLogEntry):
        """Update performance tracking metrics"""
        performance_entry = {
            "timestamp": entry.timestamp.isoformat(),
            "conversation_id": entry.conversation_id,
            "strategy": entry.context_strategy,
            "response_time": entry.performance_metrics.get("response_time", 0),
            "tokens_used": entry.performance_metrics.get("tokens_used", 0),
            "context_size": entry.context_details.get("context_size", 0),
            "intent_confidence": entry.intent_classification.get("confidence", 0),
            "threads_active": entry.thread_info.get("active_threads", 0)
        }
        
        # Write to performance log
        try:
            with open(self.performance_log_file, 'a', encoding='utf-8') as f:
                json.dump(performance_entry, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"❌ Error writing performance log: {e}")
    
    def log_conversation_turn(self, 
                            conversation_id: str,
                            message_index: int,
                            user_message: str,
                            assistant_response: str,
                            context_strategy: str,
                            intent_classification: Dict,
                            context_details: Dict,
                            performance_metrics: Dict,
                            thread_info: Dict,
                            log_level: LogLevel = LogLevel.INFO):
        """Log a conversation turn asynchronously"""
        
        log_entry = ConversationLogEntry(
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            message_index=message_index,
            user_message=user_message,
            assistant_response=assistant_response,
            context_strategy=context_strategy,
            intent_classification=intent_classification,
            context_details=context_details,
            performance_metrics=performance_metrics,
            thread_info=thread_info,
            log_level=log_level
        )
        
        # Add to queue for async processing
        self.log_queue.put(log_entry)
    
    def log_context_selection(self,
                            conversation_id: str,
                            user_message: str,
                            selected_strategy: str,
                            context_result: Dict,
                            performance_metrics: Dict):
        """Log context selection details"""
        
        context_log = {
            "timestamp": datetime.now().isoformat(),
            "conversation_id": conversation_id,
            "user_message": user_message[:100] + "..." if len(user_message) > 100 else user_message,
            "selected_strategy": selected_strategy,
            "context_result": context_result,
            "performance_metrics": performance_metrics
        }
        
        # This is a special log entry for context selection
        try:
            context_log_file = os.path.join(self.log_dir, f"context_selection_{datetime.now().strftime('%Y%m%d')}.jsonl")
            with open(context_log_file, 'a', encoding='utf-8') as f:
                json.dump(context_log, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"❌ Error logging context selection: {e}")
    
    def get_strategy_rankings(self) -> List[Dict]:
        """Get strategy rankings based on usage and performance"""
        rankings = []
        
        for strategy, stats in self.strategy_stats.items():
            ranking = {
                "strategy": strategy,
                "usage_count": stats["usage_count"],
                "avg_response_time": stats["avg_response_time"],
                "avg_tokens": stats["avg_tokens"],
                "unique_conversations": len(stats["conversations"]),
                "efficiency_score": self._calculate_efficiency_score(stats)
            }
            rankings.append(ranking)
        
        # Sort by efficiency score (higher is better)
        rankings.sort(key=lambda x: x["efficiency_score"], reverse=True)
        
        return rankings
    
    def _calculate_efficiency_score(self, stats: Dict) -> float:
        """Calculate efficiency score for strategy ranking"""
        # Simple scoring algorithm - can be improved
        usage_weight = 0.3
        response_time_weight = 0.4  # Lower is better
        token_efficiency_weight = 0.3  # Lower is better
        
        # Normalize values (simple approach)
        max_usage = max(s["usage_count"] for s in self.strategy_stats.values())
        max_response_time = max(s["avg_response_time"] for s in self.strategy_stats.values())
        max_tokens = max(s["avg_tokens"] for s in self.strategy_stats.values())
        
        usage_score = stats["usage_count"] / max(max_usage, 1)
        response_time_score = 1 - (stats["avg_response_time"] / max(max_response_time, 1))
        token_efficiency_score = 1 - (stats["avg_tokens"] / max(max_tokens, 1))
        
        return (usage_weight * usage_score + 
                response_time_weight * response_time_score + 
                token_efficiency_weight * token_efficiency_score)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            "total_strategies": len(self.strategy_stats),
            "total_logged_messages": sum(s["usage_count"] for s in self.strategy_stats.values()),
            "strategy_rankings": self.get_strategy_rankings(),
            "log_files": {
                "conversation_log": self.conversation_log_file,
                "strategy_stats": self.strategy_stats_file,
                "performance_log": self.performance_log_file
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def __del__(self):
        """Cleanup when logger is destroyed"""
        self.stop_worker()

# Global logger instance
_global_logger = None

def get_conversation_logger(log_dir: str = "logs") -> ConversationLogger:
    """Get the global conversation logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = ConversationLogger(log_dir)
    
    return _global_logger

def log_conversation_turn(conversation_id: str,
                        message_index: int,
                        user_message: str,
                        assistant_response: str,
                        context_strategy: str,
                        intent_classification: Dict,
                        context_details: Dict,
                        performance_metrics: Dict,
                        thread_info: Dict,
                        log_level: LogLevel = LogLevel.INFO):
    """Convenience function for logging conversation turns"""
    
    logger = get_conversation_logger()
    logger.log_conversation_turn(
        conversation_id=conversation_id,
        message_index=message_index,
        user_message=user_message,
        assistant_response=assistant_response,
        context_strategy=context_strategy,
        intent_classification=intent_classification,
        context_details=context_details,
        performance_metrics=performance_metrics,
        thread_info=thread_info,
        log_level=log_level
    )

def log_context_selection(conversation_id: str,
                        user_message: str,
                        selected_strategy: str,
                        context_result: Dict,
                        performance_metrics: Dict):
    """Convenience function for logging context selection"""
    
    logger = get_conversation_logger()
    logger.log_context_selection(
        conversation_id=conversation_id,
        user_message=user_message,
        selected_strategy=selected_strategy,
        context_result=context_result,
        performance_metrics=performance_metrics
    )

def get_strategy_rankings() -> List[Dict]:
    """Get strategy usage rankings"""
    logger = get_conversation_logger()
    return logger.get_strategy_rankings()

def get_performance_summary() -> Dict:
    """Get performance summary"""
    logger = get_conversation_logger()
    return logger.get_performance_summary()