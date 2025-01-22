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

from langchain_ollama import OllamaLLM 

from utils import gpt,get_chain,get_vectors,chat

import os
import tempfile



base_url :str = "http://localhost:11434"
model :str = 'llama3.2'

db = None
chain = None

app = FastAPI()

## Load Ollama LAMA2 LLM model
llm = OllamaLLM(model=model)


async def get_generated_text(prompt: str):


    #this is defualt ollama endpoint
    url = f"{base_url}/api/generate"
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

        except Exception as e:
            print(str(e))
            raise RuntimeError(f"Error connecting with model: {str(e)}")
        

@app.get('/')
async def root():

    return({'message':'Hello'})

@app.get("/query")
async def get_vector(query:str):

    try:
        result = db.similarity_search(query)
        response = result[0].page_content

        return ({'response':response})

    except Exception as e:
        print(str(e))
        raise RuntimeError(f"Error connecting with vector store: {str(e)}")


@app.post("/upload-pdf")
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

    global db 
    db = get_vectors(base_url,model,docs)
   
    return ({'response':'success'})


@app.post("/pdf_url")
async def prompt_pdf(file_url:str,prompt:str = 'What is summary of this text'):


    # Save the file temporarily
    try:

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(file_url)
        docs = loader.load()

    except Exception as e:
            print(str(e))
            raise RuntimeError(f"Error connecting with model: {str(e)}")


    global db,chain 
    db = get_vectors(base_url,model,docs)
    chain = get_chain(db,llm)

    return ({'response':'success'})




@app.post("/prompt-pdf")
async def prompt_pdf(prompt = 'What is this document summary'):
    
    response = chat(prompt,chain)

    return ({'response':response})



@app.post("/prompt-model")
async def generate_tex_prompt(prompt:str):
        
    try:

        response = await get_generated_text(prompt)
        return JSONResponse(response)

    except Exception as e:
            print(str(e))
            raise RuntimeError(f"Error connecting with model: {str(e)}")



@app.post("/api/models/download")
async def download_model(llm_name: str = Body(..., embed=True)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f'{base_url}/api/pull',json={"name": llm_name})
            response.raise_for_status()
            return {"message": f"Model {llm_name} downloaded successfully"}
    except Exception as e:
            print(str(e))
            raise RuntimeError(f"Error connecting with model: {str(e)}")
    


@app.get("/api/models")
async def list_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f'{base_url}/api/tags')
            response.raise_for_status()
            return {"models": response.json()["models"]}
    except Exception as e:
            print(str(e))
            raise RuntimeError(f"Error connecting with model: {str(e)}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
