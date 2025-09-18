import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

def load_document(file_path):
    """
    Loads a document (PDF, DOCX, or TXT) from the given file path.
    """
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension.lower() == '.docx':
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_extension.lower() == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF, DOCX, and TXT are supported.")
    documents = loader.load()
    return documents

def split_text_into_chunks(documents):
    """
    Splits the loaded documents into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks using HuggingFace embeddings.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vector_store

def get_doctor_persona(vector_store, groq_api_key):
    """
    Determines the appropriate doctor persona based on the document content.
    """
    # Using a fast and capable model for persona detection
    llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

    prompt_template = """
    Based on the following medical report context, what is the most relevant medical specialty or type of doctor for this case?
    For example: Cardiologist, Oncologist, Neurologist, Orthopedic Surgeon, etc.
    Be very specific and concise. Just state the specialty.

    CONTEXT:
    {context}

    SPECIALTY:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # This chain is specifically for determining the persona
    persona_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    query = "Determine the medical specialty from this report."
    result = persona_chain.invoke({"query": query})
    return result["result"].strip()


def create_conversational_chain(vector_store, groq_api_key, persona):
    """
    Creates the main conversational RAG chain with the determined doctor persona.
    """
    # Using a more powerful model for detailed answers
    llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

    prompt_template = f"""
    You are a helpful and empathetic AI medical assistant. Your name is DocTalk.
    You are role-playing as a highly knowledgeable **{persona}**.
    Your task is to answer the user's questions based ONLY on the context of the provided medical report.
    Do not provide medical advice, diagnoses, or treatment plans. You can explain what is in the report, but you cannot interpret it.
    If the answer to a question is not found within the provided context, you must clearly state, "I cannot find information about that in the provided medical report."
    Be concise, clear, and use easy-to-understand language.

    CONTEXT:
    {{context}}

    QUESTION:
    {{question}}

    ANSWER:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

