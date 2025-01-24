from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaLLM 

## Design ChatPrompt Template
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


async def process_pdf(task_id: str, docs: str,base_url:str,model:str,llm:object,fake_db:dict):
    """
    Process the PDF in the background and update the fake_db.
    """
    try:
        # Generate db and chain
        db = get_vectors(base_url, model, docs)
        chain = get_chain(db, llm)

        prompt = ''' The unstructured text includes 5 fields that are required to be extracted. These fields are:
        Amount and currency of second charge mortgage to be granted,
        Duration of the second charge mortgage,
        The total amount to be repaid,
        Broker Fee,
        Added to Loan,
        Lender Fee,
        Lender Name,
        Interest Rate,
        This document produced for,
        UK Mortgage Lending Ltd will pay us a commission,
        Initial monthly instalment

        Please extract the values into the fields with the same name.

        '''

        response = chat(prompt,chain)

        # Update the fake_db with the results
        fake_db[task_id] = {"status": "completed", "db": db, "chain": chain,'response':response}


    except Exception as e:
        # Update the fake_db with an error status
        fake_db[task_id] = {"status": "error", "error": str(e)}


def gpt(input:str,db:object,llm=object):

    # Construct the prompt
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context. 
        Think step by step before providing a detailed answer. 
        I will tip you $1000 if the user finds the answer helpful. 
        <context>
        {context}
        </context>
        Question: {input} 
        """
    )

    #Flow
    #any input inquiry will go to retriver. Retriver will retrive context from vector store
    #retrived chain go to llm with context and prompt.

    ## Create Stuff Docment Chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()  #Interface for vector store
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    response = retrieval_chain.invoke({"input":input})

    return response['answer']


def get_chain(db:object,llm=object):

    # Construct the prompt
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context. 
        Think step by step before providing a detailed answer. 
        I will tip you $1000 if the user finds the answer helpful. 
        <context>
        {context}
        </context>
        Question: {input} 
        """
    )

    #Flow
    #any input inquiry will go to retriver. Retriver will retrive context from vector store
    #retrived chain go to llm with context and prompt.

    ## Create Stuff Docment Chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()  #Interface for vector store
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    return retrieval_chain


def chat(prompt:str,chain:object):

    response = chain.invoke({"input":prompt})

    return response['answer']


def get_vectors(base_url,model,docs):

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text_splitter.split_documents(docs)[:5]
    documents=text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=model,base_url=base_url)
    db = FAISS.from_documents(documents[:10],embeddings)

    return db
