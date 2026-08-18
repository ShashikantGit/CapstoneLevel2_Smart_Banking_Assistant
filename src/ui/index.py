import os

import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Banking Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

QUERY_API_URL = os.getenv(
    "QUERY_API_URL",
    "http://localhost:8000/api/v1/query",
)

UPLOAD_API_URL = os.getenv(
    "UPLOAD_API_URL",
    "http://localhost:8000/api/v1/upload",
)

REQUEST_TIMEOUT = 60
UPLOAD_TIMEOUT = 120


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .app-header {
        padding: 1rem 0 1.5rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 1rem;
    }

    /* Chat area */
    .chat-container {
        max-width: 1000px;
        margin: auto;
    }

    /* Source section */
    .source-box {
        background-color: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 0.75rem 1rem;
        margin-top: 0.75rem;
        border-radius: 0.4rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 0.5rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "page" not in st.session_state:
    st.session_state.page = "Chatbot"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clear_chat():
    """Clear the current conversation."""
    st.session_state.messages = []


def call_query_api(
    query: str,
    top_k: int = 5,
):
    """
    Call the FastAPI LangGraph query endpoint.
    """

    payload = {
        "query": query,
        "top_k": top_k,
    }

    response = requests.post(
        QUERY_API_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    return response


def extract_sources(sources):
    """
    Convert different possible source formats into
    human-readable source information.
    """

    if not sources:
        return []

    formatted_sources = []

    for index, source in enumerate(
        sources,
        start=1,
    ):

        if isinstance(source, dict):

            source_name = (
                source.get("source")
                or source.get("file")
                or source.get("document")
                or source.get("filename")
                or source.get("question")
                or f"Source {index}"
            )

            page = source.get(
                "page",
                source.get(
                    "page_number",
                    "N/A",
                ),
            )

            formatted_sources.append(
                {
                    "name": source_name,
                    "page": page,
                }
            )

        else:

            formatted_sources.append(
                {
                    "name": str(source),
                    "page": "N/A",
                }
            )

    return formatted_sources


def display_sources(sources):
    """
    Display source documents returned by the API.
    """

    formatted_sources = extract_sources(
        sources
    )

    if not formatted_sources:
        return

    st.markdown(
        "**📚 Sources**"
    )

    for index, source in enumerate(
        formatted_sources,
        start=1,
    ):

        st.caption(
            f"{index}. {source['name']} "
            f"— Page {source['page']}"
        )


def display_api_error(response):
    """
    Display a user-friendly API error.
    """

    status_code = response.status_code

    try:

        error_data = response.json()

        if isinstance(
            error_data,
            dict,
        ):

            detail = error_data.get(
                "detail"
            )

        else:

            detail = None

    except ValueError:

        detail = None

    if status_code == 400:

        st.error(
            detail
            or "Invalid request. "
            "Please check your question."
        )

    elif status_code == 404:

        st.error(
            "The requested banking API "
            "endpoint was not found."
        )

    elif 400 <= status_code < 500:

        st.error(
            detail
            or "The request could not "
            "be processed."
        )

    elif status_code >= 500:

        st.error(
            "The banking service encountered "
            "an internal error. Please try again."
        )

    else:

        st.error(
            "Unexpected response from "
            "the banking service."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;">
            <h1>🏦</h1>
            <h2>Smart Banking</h2>
            <p>Agentic AI Assistant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        "### Navigation"
    )

    page = st.radio(
        "Select page",
        [
            "Chatbot",
            "File Upload",
        ],
        index=(
            0
            if st.session_state.page == "Chatbot"
            else 1
        ),
        label_visibility="collapsed",
    )

    st.session_state.page = page

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        clear_chat()

        st.rerun()

    st.divider()

    st.markdown(
        "### About"
    )

    st.caption(
        """
        Smart Banking Assistant is an
        Agentic AI application designed
        to answer banking questions using
        retrieval and database tools.
        """
    )

    st.caption(
        """
        **Architecture**

        Streamlit → FastAPI → LangGraph
        → VECTOR_DB / RDBMS
        """
    )

    st.divider()

    st.caption(
        "Phase 11 — Streamlit UI"
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">

        <div class="app-title">
            🏦 Smart Banking Assistant
        </div>

        <div class="app-subtitle">
            Your AI-powered banking assistant
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHATBOT PAGE
# ============================================================

if page == "Chatbot":

    st.markdown(
        '<div class="chat-container">',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Welcome message
    # --------------------------------------------------------

    if not st.session_state.messages:

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                """
                👋 **Welcome to Smart Banking Assistant!**

                I can help you with banking-related
                questions such as:

                - 🏠 Home loans
                - 💳 Banking products
                - 📄 Banking policies
                - 💰 Interest rates
                - 📚 General banking information
                - 🏦 Account-related questions
                """
            )

    # --------------------------------------------------------
    # Display previous conversation
    # --------------------------------------------------------

    for message in st.session_state.messages:

        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        with st.chat_message(role):

            st.markdown(content)

            # Display metadata for assistant messages
            if role == "assistant":

                route = message.get(
                    "route"
                )

                sources = message.get(
                    "sources",
                    [],
                )

                if route:

                    st.caption(
                        f"🔀 Route: {route}"
                    )

                if sources:

                    display_sources(
                        sources
                    )

    # --------------------------------------------------------
    # Chat input
    # --------------------------------------------------------

    query = st.chat_input(
        "Ask your banking question...",
    )

    if query and query.strip():

        query = query.strip()

        # ----------------------------------------------------
        # Display user message
        # ----------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.markdown(query)

        # ----------------------------------------------------
        # Save user message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        # ----------------------------------------------------
        # Assistant response
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    response = call_query_api(
                        query=query,
                        top_k=5,
                    )

                    # ----------------------------------------
                    # API ERROR
                    # ----------------------------------------

                    if response.status_code != 200:

                        display_api_error(
                            response
                        )

                    # ----------------------------------------
                    # SUCCESS
                    # ----------------------------------------

                    else:

                        try:

                            response_data = (
                                response.json()
                            )

                        except ValueError:

                            st.error(
                                "The banking service "
                                "returned an invalid "
                                "response."
                            )

                            response_data = None

                        if response_data is not None:

                            answer = (
                                response_data.get(
                                    "answer"
                                )
                            )

                            route = (
                                response_data.get(
                                    "route"
                                )
                            )

                            sources = (
                                response_data.get(
                                    "sources",
                                    [],
                                )
                            )

                            metadata = (
                                response_data.get(
                                    "metadata",
                                    {},
                                )
                            )

                            # --------------------------------
                            # ANSWER
                            # --------------------------------

                            if not answer:

                                st.error(
                                    "No answer was received "
                                    "from the banking service."
                                )

                            elif (
                                answer.strip().lower()
                                == "not applicable."
                            ):

                                st.warning(
                                    "⚠️ No applicable "
                                    "answer was found "
                                    "for this query."
                                )

                            else:

                                st.markdown(
                                    answer
                                )

                                # --------------------------------
                                # ROUTE
                                # --------------------------------

                                if route:

                                    st.caption(
                                        f"🔀 Route: {route}"
                                    )

                                # --------------------------------
                                # SOURCES
                                # --------------------------------

                                if sources:

                                    display_sources(
                                        sources
                                    )

                                # --------------------------------
                                # SAVE ASSISTANT MESSAGE
                                # --------------------------------

                                st.session_state.messages.append(
                                    {
                                        "role": "assistant",
                                        "content": answer,
                                        "route": route,
                                        "sources": sources,
                                        "metadata": metadata,
                                    }
                                )

                # ------------------------------------------------
                # TIMEOUT
                # ------------------------------------------------

                except requests.exceptions.Timeout:

                    st.error(
                        "⏱️ The request timed out. "
                        "Please try again later."
                    )

                # ------------------------------------------------
                # CONNECTION ERROR
                # ------------------------------------------------

                except (
                    requests.exceptions.ConnectionError
                ):

                    st.error(
                        "🔌 Unable to connect to the "
                        "banking service.\n\n"
                        "Please make sure the FastAPI "
                        "server is running."
                    )

                # ------------------------------------------------
                # REQUEST ERROR
                # ------------------------------------------------

                except (
                    requests.exceptions.RequestException
                ):

                    st.error(
                        "The banking service is "
                        "temporarily unavailable. "
                        "Please try again later."
                    )

                # ------------------------------------------------
                # UNEXPECTED ERROR
                # ------------------------------------------------

                except Exception as exc:

                    st.error(
                        "Something went wrong while "
                        "processing your request."
                    )

                    st.caption(
                        f"Error: {str(exc)}"
                    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# FILE UPLOAD PAGE
# ============================================================

elif page == "File Upload":

    st.markdown(
        "## 📄 Document Upload"
    )

    st.markdown(
        """
        Upload banking documents to the knowledge base.

        Supported file types:

        - PDF
        - TXT
        - DOCX
        """
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=[
            "pdf",
            "txt",
            "docx",
        ],
    )

    if uploaded_file is not None:

        st.info(
            f"Selected file: "
            f"**{uploaded_file.name}**"
        )

        st.caption(
            f"File size: "
            f"{uploaded_file.size / 1024:.2f} KB"
        )

        if st.button(
            "📤 Upload Document",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Uploading and processing document..."
            ):

                try:

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }

                    response = requests.post(
                        UPLOAD_API_URL,
                        files=files,
                        timeout=UPLOAD_TIMEOUT,
                    )

                    # ----------------------------------------
                    # SUCCESS
                    # ----------------------------------------

                    if response.status_code in (
                        200,
                        201,
                    ):

                        try:

                            response_data = (
                                response.json()
                            )

                        except ValueError:

                            response_data = {}

                        st.success(
                            "✅ Document uploaded "
                            "successfully."
                        )

                        if response_data:

                            st.json(
                                response_data
                            )

                    # ----------------------------------------
                    # ERROR
                    # ----------------------------------------

                    else:

                        display_api_error(
                            response
                        )

                except requests.exceptions.Timeout:

                    st.error(
                        "⏱️ Document upload timed out."
                    )

                except (
                    requests.exceptions.ConnectionError
                ):

                    st.error(
                        "🔌 Unable to connect to "
                        "the upload service."
                    )

                except (
                    requests.exceptions.RequestException
                ):

                    st.error(
                        "The upload service is "
                        "temporarily unavailable."
                    )

                except Exception as exc:

                    st.error(
                        "Something went wrong while "
                        "uploading the document."
                    )

                    st.caption(
                        f"Error: {str(exc)}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Smart Banking Assistant •
        Agentic AI •
        Phase 11
    </div>
    """,
    unsafe_allow_html=True,
)