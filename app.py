import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain.chains import RetrievalQA

# ---------- CONFIG ----------
load_dotenv()

st.set_page_config(
    page_title="Axiom | Research Intelligence",
    page_icon="📘",
    layout="centered"
)

# ---------- DARK THEME ----------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #020617, #000000);
    color: #E5E7EB;
    font-family: Inter, sans-serif;
}
#MainMenu, footer, header {visibility: hidden;}

.title {
    text-align:center;
    font-size:3rem;
    font-weight:800;
    margin-bottom:0.3rem;
}
.subtitle {
    text-align:center;
    color:#94A3B8;
    margin-bottom:2.5rem;
}
.card {
    background:#020617;
    border:1px solid #1E293B;
    border-radius:16px;
    padding:2rem;
}
.stButton>button {
    width:100%;
    background:linear-gradient(135deg,#6366F1,#22D3EE);
    color:#020617;
    font-weight:700;
    border-radius:12px;
    padding:0.8rem;
}
.alert {
    background:#020617;
    border-left:4px solid #6366F1;
    padding:12px;
    border-radius:10px;
    margin-bottom:1rem;
}
.footer {
    text-align:center;
    margin-top:3rem;
    font-size:0.85rem;
    color:#64748B;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">📘 Axiom</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered research understanding, distilled</div>',
    unsafe_allow_html=True
)

# ---------- CENTER ----------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    pdf_path = "data/paper.pdf"

    if not os.path.exists(pdf_path):
        st.markdown("""
        <div class="alert">
        ⚠️ No research paper detected.<br>
        Add <b>paper.pdf</b> to the <code>data/</code> folder.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.markdown(f"""
    <div class="alert">
    📄 Loaded document: <b>{os.path.basename(pdf_path)}</b>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Generate Research Brief"):
        with st.spinner("Analyzing paper..."):

            # ---------- READ PDF ----------
            reader = PdfReader(pdf_path)
            paper_text = ""

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    paper_text += text

            if not paper_text.strip():
                st.error("❌ This PDF has no readable text (likely scanned).")
                st.stop()

            # ---------- SPLIT ----------
            splitter = CharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=150
            )
            chunks = splitter.split_text(paper_text)

            if not chunks:
                st.error("❌ No text chunks created from the document.")
                st.stop()

            # ---------- LOCAL EMBEDDINGS (NO QUOTA) ----------
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_texts(chunks, embeddings)

            # ---------- LLM ----------
            llm = Ollama(
                model="llama3",
                temperature=0
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )

            query = """
            Provide a clear academic summary covering:
            - Research objective
            - Methodology
            - Key findings
            - Conclusions
            """

            result = qa_chain.invoke(query)

            st.markdown("### 📌 Research Brief")
            st.markdown(f"""
            <div class="card" style="margin-top:1rem;">
            {result["result"]}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footer">Built with LangChain, OpenAI, FAISS, and HuggingFace</div>',
    unsafe_allow_html=True
)
