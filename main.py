import os
import requests
import msal
import openai
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
@app.get("/")
def home():
    return {"message": "Chatbot API is running!"}

# Load biến môi trường
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
azure_client_id = os.getenv("AZURE_CLIENT_ID")
azure_tenant_id = os.getenv("AZURE_TENANT_ID")
azure_client_secret = os.getenv("AZURE_CLIENT_SECRET")
drive_id = os.getenv("SHAREPOINT_DRIVE_ID")

# Kết nối ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="training_docs")

class ChatRequest(BaseModel):
    question: str

def get_access_token():
    app = msal.ConfidentialClientApplication(
        azure_client_id,
        authority=f"https://login.microsoftonline.com/{azure_tenant_id}",
        client_credential=azure_client_secret
    )
    token = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    return token.get("access_token")

def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return response["data"][0]["embedding"]

def search_in_chroma(query, top_k=3):
    embedding = get_embedding(query)
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    return [result[0] for result in results["documents"]]

def generate_answer(query):
    context = "\n".join(search_in_chroma(query))
    if context:
        prompt = f"Dựa trên thông tin sau:\n\n{context}\n\nHãy trả lời câu hỏi: {query}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response['choices'][0]['message']['content'].strip()
    return "Xin lỗi, tôi không tìm thấy thông tin phù hợp."

@app.post("/chat")
def chat(request: ChatRequest):
    return {"answer": generate_answer(request.question)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
