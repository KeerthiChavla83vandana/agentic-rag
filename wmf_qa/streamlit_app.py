import os
import requests
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Vrutti FPO Intelligence Platform",
    page_icon="🌱",
    layout="wide",
)

# Load backend URL from env
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")

# Stylized header
st.markdown(
    """
    <div style="background-color:#1e3d59;padding:20px;border-radius:10px;margin-bottom:20px;">
        <h1 style="color:#ffc13b;margin:0;font-family:'Outfit', sans-serif;">🌱 Vrutti FPO Intelligence Platform</h1>
        <p style="color:#f5f0e1;margin:5px 0 0 0;font-size:16px;">
            Walmart Foundation Initiative • Resilient Livelihoods • Kalaburagi & Chikkaballapur, Karnataka
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Test connection to backend
backend_online = False
try:
    resp = requests.get(BACKEND_URL, timeout=3)
    if resp.status_code == 200:
        backend_online = True
        backend_info = resp.json()
except Exception:
    pass

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ System Control Center")

    # Connection Status Pill
    if backend_online:
        st.success("● Connected to QA Backend")
        st.info(f"Embedding Model: `{backend_info.get('embedding_model')}`")
    else:
        st.error("○ Backend Offline")
        st.warning(f"Unable to connect to `{BACKEND_URL}`. Ensure containers are running.")

    st.markdown("---")

    # Persona Selector
    st.markdown("### 👥 User Persona")
    persona_options = {
        "Program Lead": "🎯 Strategic view, high-level metrics, alerts, funder-grade evidence.",
        "Junior Associate": "📋 Operational details, task lists, locations, action items.",
        "Field Enumerator": "🌾 Simplified ground-level instructions, crop/farmer support tips.",
        "Untailored": "⚖️ Objective, balanced summary of context chunks.",
    }
    selected_persona = st.selectbox(
        "Select User Persona",
        options=list(persona_options.keys()),
        index=0,
        help="Tailors response focus and vocabulary according to your project role.",
    )
    st.caption(f"_{persona_options[selected_persona]}_")

    st.markdown("---")

    # Document uploader
    st.markdown("### 📥 Document & Transcript Indexer")
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, or PPTX report",
        type=["pdf", "docx", "pptx"],
    )

    if uploaded_file is not None:
        doc_id = st.text_input(
            "Define Unique Document ID",
            value=os.path.splitext(uploaded_file.name)[0].replace(" ", "_"),
        )
        folder_tag = st.text_input("Folder Group Tag (Optional)", value="Vrutti_FPO_RAG")

        if st.button("🚀 Ingest & Generate Chunks", use_container_width=True):
            if not backend_online:
                st.error("Cannot upload: Backend is offline.")
            elif not doc_id:
                st.error("Please specify a document ID.")
            else:
                with st.spinner("Processing file through extraction, chunking, and embedding (BAAI/bge-m3)..."):
                    try:
                        # Send file to backend
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                        data = {
                            "document_id": doc_id,
                            "folder_name": folder_tag,
                        }
                        upload_resp = requests.post(
                            f"{BACKEND_URL}/wmf_qa/upload",
                            files=files,
                            data=data,
                        )

                        if upload_resp.status_code == 200:
                            result = upload_resp.json()
                            st.success(
                                f"Successfully indexed **{uploaded_file.name}**!\n\n"
                                f"Generated and stored **{result.get('chunks_extracted')}** chunks in pgvector."
                            )
                        else:
                            st.error(f"Upload failed: {upload_resp.text}")
                    except Exception as e:
                        st.error(f"Error communicating with backend: {e}")

# Main Chat panel
st.markdown("### 💬 Interrogate FPO Intelligence Database")
st.markdown(
    "Ask natural language questions (English, Kannada, or mixed) about FPC membership targets, "
    "farmer meeting records, budget guidelines, field survey feedback, etc."
)

query_input = st.text_input(
    "Enter your question:",
    placeholder="e.g. What are the key issues reported in farmer meetings in Kalaburagi?",
)

if st.button("🔍 Ask Intelligence Layer", type="primary", use_container_width=True) or (query_input and st.session_state.get("last_query") != query_input):
    if not query_input:
        st.warning("Please enter a question.")
    elif not backend_online:
        st.error("Cannot query: Backend is offline.")
    else:
        st.session_state["last_query"] = query_input
        with st.spinner("Executing agentic pipeline..."):
            try:
                # Send query request to backend
                payload = {"query": query_input, "persona": selected_persona}
                resp = requests.post(f"{BACKEND_URL}/wmf_qa/query", json=payload)

                if resp.status_code == 200:
                    data = resp.json()

                    # Render answer in main panel
                    st.markdown("#### 🌟 Synthesized Response")
                    st.info(data.get("response"))

                    # Renders Step-by-Step debugging tracer
                    st.markdown("### 🛠️ Execution Tracer Logs (Step-by-Step Debugger)")

                    # Step 1: Router
                    route = data.get("route")
                    route_icon = "📂" if route == "vector_db" else "🌐"
                    st.markdown(f"**Step 1: Adaptive Router Decision**: `{route.upper()}` {route_icon}")
                    if route == "vector_db":
                        st.caption("Routed to local database because query relates to Vrutti FPO program materials.")
                    else:
                        st.caption("Routed to general knowledge base (no local document contexts needed).")

                    if route == "vector_db":
                        # Corrective action took indicator
                        corrective_taken = data.get("corrective_action_taken")
                        st.markdown(f"**Step 2: Corrective Retrieval Loop Status**")
                        if corrective_taken:
                            st.warning(
                                "⚠️ Low relevance score detected on original query. "
                                f"Triggered query reformulation to: *'{data.get('search_query')}'*"
                            )
                        else:
                            st.success("✔️ Search chunks relevance criteria met immediately. No reformulation needed.")

                        # Expandable retrieved chunks
                        with st.expander("📁 View Graded Retrieval Context Chunks"):
                            retrieved_chunks = data.get("retrieved_chunks", [])
                            if not retrieved_chunks:
                                st.write("No document chunks returned.")
                            else:
                                for idx, chunk in enumerate(retrieved_chunks):
                                    grade_color = "green" if chunk.get("grade") == "YES" else "red"
                                    st.markdown(
                                        f"**Chunk #{idx+1}** (ID: `{chunk.get('chunk_id')}`) "
                                        f"| Relevance Grade: <span style='color:{grade_color};font-weight:bold;'>{chunk.get('grade')}</span>",
                                        unsafe_allow_html=True,
                                    )
                                    st.write(f"Content: _{chunk.get('content')}_")
                                    st.caption(f"RRF Score: `{chunk.get('rrf_score')}`")
                                    st.markdown("---")

                        # Step 3: Verified citations
                        verified_cites = data.get("citations_verified", [])
                        st.markdown("**Step 3: Post-Generation Citation Verification**")
                        if verified_cites:
                            st.success(f"Verified {len(verified_cites)} source document citations in text.")
                            for cite in verified_cites:
                                st.markdown(f"- Cited Chunk ID: `{cite}`")
                        else:
                            st.warning("No document citations could be verified or cited in output.")

                else:
                    st.error(f"Pipeline error: {resp.text}")

            except Exception as e:
                st.error(f"Communication error: {e}")
