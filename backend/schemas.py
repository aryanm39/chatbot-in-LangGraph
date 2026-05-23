from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    session_id: str
    is_resume: bool = False

class ChatResponse(BaseModel):
    answer: str
    session_id: str



