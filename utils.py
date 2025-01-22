from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaLLM 

## Design ChatPrompt Template
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain



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
