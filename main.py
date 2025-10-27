from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

from agent import agent

load_dotenv()

class ChatRequest(BaseModel):
    question: str 
    chat_history: List[Dict[str, str]] # e.g., [{"type": "human", "content": "hi"}]



app = FastAPI(title="Botivate Rag Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

@app.head("/")
async def status_check():
    return Response(status_code=200)

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    The main endpoint to interact with the agent.
    It accepts a question and the conversation history.
    """

    history_messages = []

    for msg in request.chat_history:
        if msg.get("type") == "human":
            history_messages.append(HumanMessage(content=msg.get("content")))
        elif msg.get("type") == "ai":
            history_messages.append(AIMessage(content=msg.get("content")))
        
    initial_state = {
        "question": request.question,
        "chat_history": history_messages
    }

    final_state = agent.invoke(initial_state)

    return {"answer": final_state.get('answer', "Sorry, I encountered an error.")}
