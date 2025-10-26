import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import time
import random
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Initialize Google Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
 
INDEX_PATH = os.path.join(os.getcwd(), "faiss_index")

 
def load_or_create_faiss():
    try:
        store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded existing FAISS index from: {INDEX_PATH}")
        return store
    except Exception as e:
        print(f"Creating new FAISS Web index at: {INDEX_PATH}")
        return FAISS.from_texts(["init"], embedding=embeddings)

vector_store = load_or_create_faiss()
print("FAISS index ready!")


def scrape_url(url):
    headers_req = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }
    
    time.sleep(random.uniform(0.5, 1.5))
    response = requests.get(url, headers=headers_req, timeout=15)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    
    text = soup.get_text(separator=' ', strip=True)
    return ' '.join(text.split())


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    if not text:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=overlap
    )
    return text_splitter.split_text(text)


def store_embedding(chunks: list, url: str, url_id: str):
    global vector_store
    
    if not chunks:
        print(" No chunks to process")
        return
    
    print(f"Storing {len(chunks)} chunks...")
    
    metadatas = [
        {"url": url, "url_id": url_id, "chunk_index": i} 
        for i in range(len(chunks))
    ]
    ids = [f"{url_id}_{i}" for i in range(len(chunks))]
    
    # Add to FAISS and save
    vector_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
    vector_store.save_local(INDEX_PATH)
    
    print(f" Stored {len(chunks)} chunks successfully!")


def search_similar(query: str, k: int = 5, url_id: str = None):
    if not vector_store:
        raise ValueError("FAISS index not initialized")
    
    results = vector_store.similarity_search_with_score(query, k=k)
    
    print(results)
    
     
     
    formatted_results = {
        'documents': [[doc.page_content for doc, _ in results]]
    }
    
    return formatted_results


def get_prompt(context, question):
    prompt_template = f"""
You are an expert assistant. Analyze the user’s question carefully and provide a clear, helpful answer using ONLY the information in the context below. 
Follow these steps:

1. Identify the main intent of the question.
2. Check if the answer exists in the context.
3. Highlight the parts of the context that support your answer.
4. Provide a concise, clear answer (50-100 words).
5. If the answer is not in the context, say: "Answer not available in the context."
6 if its greeeting greet back
7. If only partial information is available, give a short, relevant answer with reasoning.

Context:
{context}

Question:
{question}

Answer (include reasoning and relevant context references):
"""
    return prompt_template
