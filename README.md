🩺 DocTalk: AI-Powered Medical Report Chatbot
DocTalk is an intelligent chatbot application that allows users to upload their medical reports (PDF or Word documents) and ask questions about them. The application uses a Retrieval-Augmented Generation (RAG) pipeline to provide accurate and context-aware answers.

A key feature of DocTalk is its ability to dynamically adopt the persona of a relevant medical specialist (e.g., Cardiologist, Oncologist) based on the content of the uploaded report.

Features
File Upload: Supports both PDF and DOCX file formats.

Dynamic Persona: Automatically analyzes the report to determine the most relevant medical specialty and tailors its responses accordingly.

Contextual Q&A: Answers questions based solely on the information present in the uploaded document.

User-Friendly Interface: Built with Streamlit for a clean and interactive user experience.

High-Speed LLM: Powered by Groq for near-instantaneous responses.

State-of-the-Art Embeddings: Utilizes Hugging Face's sentence-transformers for efficient document analysis.

Privacy-Focused: Documents are processed in-memory for the duration of the session and are not stored.

Tech Stack
Frontend: Streamlit

Core Logic: Python

LLM Orchestration: LangChain

LLM Provider: Groq (Llama 3)

Embeddings: Hugging Face Sentence Transformers

Vector Store: FAISS (Facebook AI Similarity Search)

Document Loading: PyPDF, python-docx

Setup and Installation
1. Clone the Repository
git clone <repository_url>
cd <repository_name>

2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies
Install all the required Python packages using the requirements.txt file.

pip install -r requirements.txt

4. Set Up Environment Variables
You'll need an API key from Groq to use their language models.

Create a file named .env in the root directory of the project.

Add your Groq API key to the .env file as follows:

GROQ_API_KEY="your_groq_api_key_here"

How to Run the Application
Once you have completed the setup, you can run the Streamlit application with a single command:

streamlit run app.py

This will open the DocTalk application in your default web browser.

How It Works
Upload: The user uploads a medical report via the Streamlit sidebar.

Process: The application temporarily saves the file, loads its content, and splits it into smaller, manageable text chunks.

Embed & Store: Each text chunk is converted into a numerical vector (embedding) using a Hugging Face model. These embeddings are stored in a FAISS vector store for efficient searching.

Persona Detection: The app queries the vector store to find the most relevant context and uses the Groq LLM to determine the appropriate medical specialist persona.

Chat: The main conversational agent, now role-playing as the determined specialist, is ready. When the user asks a question, the app retrieves the most relevant chunks from the report, combines them with the user's question and a specialized prompt, and sends them to the Groq LLM to generate a final, context-aware answer.

Disclaimer: This application is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.