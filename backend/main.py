from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from schemas import ChatRequest, ChatResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from services import chatbot
import json
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        def generate_data():
            config = {"configurable": {"thread_id": request.session_id}}
            if getattr(request, "is_resume", False):
                result = chatbot.invoke(Command(resume=request.question), config=config)
            else:
                result = chatbot.invoke({"messages": [HumanMessage(content=request.question)]},config=config)
            interrupts = result.get("__interrupt__", [])
            if interrupts:
                interrupt_value = interrupts[0].value
                payload = {"type": "interrupt", "message": interrupt_value}
                yield json.dumps(payload)
                return

            last_message = result["messages"][-1]
            payload = {"type": "message", "message": last_message.content}
            yield json.dumps(payload)
        return StreamingResponse(generate_data(),media_type="application/json")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

#==============================================================================================
# testing 
#==============================================================================================
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse
from pathlib import Path
from rag import ingest_pdf, DATA_DIR

# =========================
# UPLOAD PDF
# =========================
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    try:
        pdf_bytes = await file.read()

        result = ingest_pdf(
            file_bytes=pdf_bytes,
            thread_id=session_id,
            filename=file.filename,
        )

        return {
            "status": "success",
            "session_id": session_id,
            "document": result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# LIST PDFs
# =========================
@app.get("/pdfs")
def list_pdfs():
    files = []

    for file in DATA_DIR.glob("*.pdf"):
        files.append({
            "filename": file.name,
            "url": f"/pdf/{file.name}"
        })

    return {
        "count": len(files),
        "files": files,
    }
#==============================================================================================
#==============================================================================================

# @app.post("/chat")
# def chat(request: ChatRequest):
#     try:
#         def generate_data():
#             for message_chunk, metadata in chatbot.stream(
#                 {"messages": [HumanMessage(content=request.question)]},
#                 config= {"configurable": {"thread_id": request.session_id}},
#                 stream_mode='messages'
#                 ):
#                 if message_chunk.content:
#                     print(message_chunk.content, end=" ", flush=True)
#                     yield message_chunk.content
#         return StreamingResponse(generate_data(),media_type="text/plain")

#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=str(exc))


