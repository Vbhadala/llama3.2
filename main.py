import json
from fastapi import FastAPI, HTTPException, Body,File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import httpx
import uvicorn


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from utils import llm, gpt

import os
import tempfile


app = FastAPI()



async def get_generated_text(prompt: str, model: str):

    url = "http://localhost:11434/api/generate"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json={"model": model, "prompt": prompt})
            response.raise_for_status()

            # Combine all responses into a single string
            combined_response = ""
            for line in response.text.splitlines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            combined_response += data["response"]
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON

            return {"response": combined_response}

        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Error communicating with the server: {str(e)}")
        

@app.post("/pdf_url/")
async def prompt_pdf(file_url:str,prompt:str = 'What is summary of this text'):


    # Save the file temporarily
    try:

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(file_url)
        docs = loader.load()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in provided file url: {str(e)}")


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text_splitter.split_documents(docs)[:5]
    documents=text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model='llama3.2',base_url='http://localhost:11434')
    db = FAISS.from_documents(documents[:10],embeddings)

    print(db)

    response = gpt(prompt,db)

    return ({'response':response})




@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):

    # Check if the uploaded file is a PDF
    if file.content_type != "application/pdf":
        return JSONResponse(content={"error": "File must be a PDF"}, status_code=400)

    # Save the file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

    finally:
        # Ensure the temporary file is deleted after use
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text_splitter.split_documents(docs)[:5]
    documents=text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model='llama3.2',base_url='http://localhost:11434')
    db = FAISS.from_documents(documents[:10],embeddings)

    # query="An attention function can be described as mapping a query "
    # result = db.similarity_search(query)
    # response = result[0].page_content

    response = gpt('What is this document summary',db)

    return ({'response':response})






@app.post("/upload-pdf/prompt")
async def upload_pdf_prompt(file: UploadFile = File(...),prompt = 'What is this document summary'):

    # Check if the uploaded file is a PDF
    if file.content_type != "application/pdf":
        return JSONResponse(content={"error": "File must be a PDF"}, status_code=400)

    # Save the file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

    finally:
        # Ensure the temporary file is deleted after use
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text_splitter.split_documents(docs)[:5]
    documents=text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model='llama3.2',base_url='http://localhost:11434')
    db = FAISS.from_documents(documents[:10],embeddings)

    # query="An attention function can be described as mapping a query "
    # result = db.similarity_search(query)
    # response = result[0].page_content

    response = gpt(prompt,db)

    return ({'response':response})





@app.post("/api/prompt")
async def generate_tex_prompt(prompt:str):

        response = await get_generated_text(prompt, 'llama3.2')
        return JSONResponse(response)



@app.post("/api/models/download")
async def download_model(llm_name: str = Body(..., embed=True)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/pull",
                json={"name": llm_name}
            )
            response.raise_for_status()
            return {"message": f"Model {llm_name} downloaded successfully"}
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error downloading model: {str(e)}")
    


@app.get("/api/models")
async def list_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            return {"models": response.json()["models"]}
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
