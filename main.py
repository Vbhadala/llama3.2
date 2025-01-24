import json
from fastapi import FastAPI, HTTPException, Body,File, UploadFile,BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import httpx
import uvicorn
import uuid


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaLLM 

from utils import gpt,get_chain,get_vectors,chat, process_pdf

import os
import tempfile



base_url :str = "http://localhost:11434"
model :str = 'llama3.2'

db :object= None
chain :object = None

# Fake in-memory database
fake_db: Dict[str, Dict] = {}

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
async def upload_pdf(background_tasks: BackgroundTasks,file: UploadFile = File(...)):

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

    # Generate a unique ID for the task
    task_id = str(uuid.uuid4())

    # Add a placeholder for the task in the database
    fake_db[task_id] = {"status": "processing", "db": None, "chain": None,'response':None}

    # Add the background task
    background_tasks.add_task(process_pdf, task_id,docs,base_url,model,llm,fake_db)

    return {"task_id": task_id, "message": "Processing started. Use the task_id to query the status."}



@app.post("/pdf-url")
async def prompt_pdf(file_url: str, background_tasks: BackgroundTasks):
    """
    Endpoint to process a PDF file asynchronously.
    """

    try:

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(file_url)
        docs = loader.load()

    except Exception as e:
            print(str(e))
            raise RuntimeError(f"Error reading PDF: {str(e)}")

    # Generate a unique ID for the task
    task_id = str(uuid.uuid4())

    # Add a placeholder for the task in the database
    fake_db[task_id] = {"status": "processing", "db": None, "chain": None,'response':None}

    # Add the background task
    background_tasks.add_task(process_pdf, task_id,docs,base_url,model,llm,fake_db)

    return {"task_id": task_id, "message": "Processing started. Use the task_id to query the status."}



@app.get("/task-status/{task_id}")
async def get_status(task_id: str):

    if task_id not in fake_db:
        return {"error": "Invalid task_id."}

    task_data = fake_db[task_id]

    if task_data["status"] != "completed":
        return {"status": task_data["status"], "message": "Task is not completed yet."}
    
    return {"status": "completed","response": task_data["response"] }



@app.post("/prompt-pdf/{task_id}")
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
    


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

