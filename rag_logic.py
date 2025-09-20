import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- File Handling ---
def save_temp_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

# --- Document Loading ---
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.docx':
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")
    return loader.load()

# --- Text Splitting ---
def split_text(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# --- Embedding and Vector Store ---
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def build_vector_store(chunks):
    embeddings = create_embeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)

# --- Full Pipeline: Create Vector Store from Uploaded File ---
def create_vector_store(uploaded_file):
    if uploaded_file is None:
        return None

    tmp_file_path = save_temp_file(uploaded_file)

    try:
        documents = load_document(tmp_file_path)
        if not documents:
            return None

        chunks = split_text(documents)
        if not chunks:
            return None

        return build_vector_store(chunks)

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# --- Persona Detection ---
def detect_doctor_persona(llm, retriever):
    prompt = PromptTemplate(
        template="""
        Based on the following medical report context, identify the primary medical specialty of the doctor who would analyze this.
        Your answer MUST be ONLY the name of the specialty (e.g., "Cardiologist", "Neurologist", "Oncologist").

        CONTEXT: {context}

        SPECIALTY:
        """,
        input_variables=["context"]
    )
    relevant_docs = retriever.get_relevant_documents("What is the main subject of this medical report?")
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"context": context_text})
    return result['text'].strip()

# --- Main Conversational RAG Chain ---
def get_conversational_chain(vector_store, groq_api_key):
    llm = ChatGroq(temperature=0.2, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    doctor_persona = detect_doctor_persona(llm, retriever)

    rag_prompt = PromptTemplate(
        template=f"""
        You are an expert **{doctor_persona}**. Your name is DocTalk.
        Your role is to answer questions about a patient's medical report in a clear, empathetic, and professional manner.
        Use ONLY the provided context from the report to answer the user's question accurately.
        you can also provide medical advice or interpretation bt it must be legit and based on the context.
        If the answer is not in the context, state that the information is not available in the report.

        CONTEXT:
        {{context}}

        QUESTION: {{input}}

        ANSWER:
        """,
        input_variables=["context", "input"]
    )

    qa_chain = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain, doctor_persona
