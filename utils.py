from langchain_ollama import OllamaLLM 

## Load Ollama LAMA2 LLM model
llm = OllamaLLM(model="llama3.2")

## Design ChatPrompt Template
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def gpt(chat:str,db:object):

    prompt = ChatPromptTemplate.from_template(

    """
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    I will tip you $1000 if the user finds the answer helpful. 
    <context>
    {context}
    </context>
    Question: {{chat}} 
    """
    )

    print(prompt)

    ## Create Stuff Docment Chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    """
    Retrievers: A retriever is an interface that returns documents given
    an unstructured query. It is more general than a vector store.
    A retriever does not need to be able to store documents, only to 
    return (or retrieve) them. Vector stores can be used as the backbone
    of a retriever, but there are other types of retrievers as well. 
    https://python.langchain.com/docs/modules/data_connection/retrievers/   

    """
 
    retriever = db.as_retriever()


    """
    Retrieval chain:This chain takes in a user inquiry, which is then
    passed to the retriever to fetch relevant documents. Those documents 
    (and original inputs) are then passed to an LLM to generate a response
    https://python.langchain.com/docs/modules/chains/

    """

    retrieval_chain=create_retrieval_chain(retriever,document_chain)


    response=retrieval_chain.invoke({"input":"What is this circular about"})

    return response['answer']
