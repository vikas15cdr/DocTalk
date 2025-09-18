import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from rag_logic import (
    load_document,
    split_text_into_chunks,
    create_vector_store,
    get_doctor_persona,
    create_conversational_chain,
)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(page_title="DocTalk", page_icon="🩺", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .st-chat-message {
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .st-chat-message[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"]>p:first-child:before) {
            background-color: #DCF8C6; /* User message color */
        }
        .st-chat-message[data-testid="stChatMessage"]:not(:has(div[data-testid="stChatMessageContent"]>p:first-child:before)) {
            background-color: #FFFFFF; /* Bot message color */
        }
        h1 {
            color: #1E3A8A; /* A deep blue color */
            text-align: center;
        }
        .st-emotion-cache-1y4p8pa {
            max-width: 80%;
            margin: auto;
        }
        .stTextInput > div > div > input {
            border-radius: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🩺 DocTalk: Your Medical Report Assistant")
    st.markdown("<h4 style='text-align: center; color: #555;'>Upload your medical report and ask questions with a specialized AI assistant.</h4>", unsafe_allow_html=True)
    st.markdown("---")

    # Initialize session state variables
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "persona" not in st.session_state:
        st.session_state.persona = "General Practitioner"

    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("Upload Your Report")
        uploaded_file = st.file_uploader(
            "Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"]
        )

        if uploaded_file:
            if st.button("Process Document"):
                if not GROQ_API_KEY:
                    st.error("GROQ_API_KEY is not set. Please add it to your environment variables.")
                else:
                    with st.spinner("Processing your document... This may take a moment."):
                        try:
                            # Use a temporary file to handle the upload
                            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name

                            # 1. Load Document
                            documents = load_document(tmp_file_path)

                            # 2. Split into chunks
                            text_chunks = split_text_into_chunks(documents)
                            if not text_chunks:
                                st.error("Could not extract text from the document. Please try another file.")
                                return

                            # 3. Create Vector Store
                            vector_store = create_vector_store(text_chunks)

                            # 4. Determine Persona
                            st.session_state.persona = get_doctor_persona(vector_store, GROQ_API_KEY)

                            # 5. Create Conversational Chain
                            st.session_state.conversation_chain = create_conversational_chain(
                                vector_store, GROQ_API_KEY, st.session_state.persona
                            )
                            
                            # 6. Reset chat history and notify user
                            st.session_state.chat_history = []
                            st.success(f"Document processed successfully! I am now acting as a **{st.session_state.persona}**.")
                            st.info("You can now ask questions about your report in the main chat window.")

                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                        finally:
                            # Clean up the temporary file
                            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                                os.remove(tmp_file_path)

        st.info("Your uploaded documents are not stored and will be deleted after the session ends.")
        st.warning("This tool is for informational purposes only and not a substitute for professional medical advice.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if user_question := st.chat_input("Ask a question about your report..."):
        if st.session_state.conversation_chain is None:
            st.warning("Please upload and process a document first.")
        else:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Get bot response
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.conversation_chain.invoke({"query": user_question})
                    response = result["result"]
                    # Add bot response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                except Exception as e:
                    error_message = f"Sorry, an error occurred: {e}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                    with st.chat_message("assistant"):
                        st.markdown(error_message)

if __name__ == "__main__":
    main()
