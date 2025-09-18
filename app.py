import streamlit as st
import os
from rag_logic import create_vector_store, get_conversational_chain

def main():
    st.set_page_config(page_title="DocTalk", page_icon="🩺", layout="wide")

    # --- Custom Styling ---
    st.markdown("""
        <style>
        .stApp { background-color: #f0f2f6; }
        h1, h2, h3 { color: #1E3A8A; text-align: center; }
        .persona-badge {
            background-color: #1E3A8A;
            color: white;
            padding: 0.4em 1em;
            border-radius: 20px;
            display: inline-block;
            font-size: 1em;
            margin-top: 10px;
        }
        .chat-bubble-user {
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
        }
        .chat-bubble-assistant {
            background-color: #FFFFFF;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
            border: 1px solid #ccc;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🩺 DocTalk: Your Medical Report Assistant")

    # --- Session State ---
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "persona" not in st.session_state:
        st.session_state.persona = None

    # --- Sidebar Upload & Reset ---
    with st.sidebar:
        st.header("📄 Upload Your Report")
        uploaded_file = st.file_uploader("Choose a PDF or Word document", type=["pdf", "docx"])

        if uploaded_file:
            st.caption(f"**File:** {uploaded_file.name} ({uploaded_file.type})")

            if st.button("🔍 Process Document"):
                with st.spinner("Analyzing your report..."):
                    try:
                        groq_api_key = st.secrets.get("GROQ_API_KEY")
                        if not groq_api_key:
                            st.error("GROQ_API_KEY is missing in Streamlit secrets.")
                            st.stop()

                        vector_store = create_vector_store(uploaded_file)
                        if vector_store is None:
                            st.error("Could not extract text. Try another file.")
                            st.stop()

                        st.session_state.conversation_chain, st.session_state.persona = get_conversational_chain(
                            vector_store, groq_api_key
                        )

                        st.session_state.messages = []
                        st.success("✅ Document processed successfully!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {e}")

        # --- Reset Chat Button ---
        if st.button("🧹 Reset Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

        st.info("Uploaded files are deleted after processing.")
        st.warning("This tool is not a substitute for professional medical advice.")

    # --- Persona Display ---
    if st.session_state.persona:
        st.markdown(f"<div class='persona-badge'>🧑‍⚕️ Role: {st.session_state.persona}</div>", unsafe_allow_html=True)
        st.markdown("#### Ask me anything about your report below 👇")

    # --- Chat Interface ---
    if not st.session_state.conversation_chain:
        st.info("Upload and process a document to start chatting.")

    for message in st.session_state.messages:
        bubble_class = "chat-bubble-user" if message["role"] == "user" else "chat-bubble-assistant"
        st.markdown(f"<div class='{bubble_class}'>{message['content']}</div>", unsafe_allow_html=True)

    if user_question := st.chat_input("Ask a question about your report..."):
        if st.session_state.conversation_chain is None:
            st.warning("Please process a document first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.markdown(f"<div class='chat-bubble-user'>{user_question}</div>", unsafe_allow_html=True)

            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.conversation_chain.invoke({"input": user_question})
                    response = result["answer"]
                    st.markdown(f"<div class='chat-bubble-assistant'>{response}</div>", unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, something went wrong: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
