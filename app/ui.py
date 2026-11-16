
import streamlit as st

from app.answer import GeminiUnavailableError, answer_question
from app.retrieval import HybridRetriever


# Page configuration
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
)

# Application heading
st.title("📄 DocuMind AI")
st.caption("AI-powered document question answering using Hybrid RAG.")

st.divider()


# Load retriever only once
@st.cache_resource
def load_retriever():
    return HybridRetriever()


# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar
with st.sidebar:
    st.header("DocuMind AI")
    st.write("Your intelligent PDF assistant.")

    st.divider()

    st.subheader("Features")
    st.write("• PDF-based question answering")
    st.write("• BM25 keyword search")
    st.write("• Dense vector search")
    st.write("• Hybrid retrieval")
    st.write("• Cross-encoder reranking")
    st.write("• Gemini AI responses")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display source citations
        if message.get("sources"):
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"- **{source['source']}**, page {source['page']}"
                    )


# User input
question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate AI response
    with st.chat_message("assistant"):

        try:
            with st.spinner(
                "Searching documents and generating your answer..."
            ):

                retriever = load_retriever()

                result = answer_question(
                    question.strip(),
                    retriever,
                )

            st.markdown(result["answer"])

            unique_sources = sorted({
                (item["source"], item["page"])
                for item in result["sources"]
            })

            sources = [
                {"source": source, "page": page}
                for source, page in unique_sources
            ]

            if sources:
                with st.expander("View Sources"):
                    for source in sources:
                        st.markdown(
                            f"- **{source['source']}**, "
                            f"page {source['page']}"
                        )

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": sources,
            })

        except GeminiUnavailableError as error:
            st.warning(str(error))

        except Exception as error:
            st.error(f"Something went wrong: {error}")
