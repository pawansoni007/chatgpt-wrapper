from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    tokens_used: int
    total_messages: int

class ConversationSummary(BaseModel):
    conversation_id: str
    message_count: int
    created_at: str
    last_updated: str