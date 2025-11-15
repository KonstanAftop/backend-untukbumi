from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import uvicorn
import pathlib
import logging

class RagRequest(BaseModel):
    question: str
    userProfile: dict
    activities: dict

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store_path = pathlib.Path(__file__).parent / "vector-store"
store = FAISS.load_local(str(vector_store_path), embeddings, allow_dangerous_deserialization=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

app = FastAPI(title="UntukBumi API")


default_origins = "http://localhost:5173,http://localhost:3000,http://localhost:8080"
allow_origins = os.environ.get("CORS_ORIGINS", default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post("/api/rag")
async def rag(req: RagRequest):
    try:
        logging.info(f"Processing RAG request for question: {req.question[:50]}...")

        retriever = store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(f"""
            Pertanyaan: {req.question}
            Profil: {req.userProfile}
            Aktivitas: {req.activities}
        """)

        if not docs:
            raise HTTPException(status_code=404, detail="No relevant documents found")

        context = "\n\n".join(f"({i+1}) {d.page_content}" for i, d in enumerate(docs))
        prompt = f"""
Anda UntukBumi AI. Jawab dalam tiga bagian (Analisis Personal, Rekomendasi, Edukasi).
Gunakan bahasa Indonesia, referensikan dokumen bila relevan.

Untuk bagian Edukasi: Jelaskan konsep perubahan iklim dan jika relevan sertakan penjelasan tentang proses meteorologi, oseanografi, atau kimia yang terkait.

Konteks:
{context}

Pertanyaan/Profil:
{req.question}
Profil: {req.userProfile}
Aktivitas: {req.activities}
"""
        response = llm.invoke(prompt)

        return {"answer": response.content, "sources": [
            {"id": i+1, "content": d.page_content, "metadata": d.metadata}
            for i, d in enumerate(docs)
        ]}

    except Exception as e:
        logging.error(f"Error processing RAG request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    # IMPORTANT: use Railway's PORT env instead of hardcoding 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)