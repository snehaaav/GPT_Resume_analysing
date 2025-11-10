import streamlit as st
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import docx
import os

st.set_page_config(layout="wide")
st.title("📄 AI Resume–Job Matcher (Free Version)")
st.subheader("🔍 Compare Resume and Job Description using Hugging Face Transformers")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(file):
    """Extract text from PDF"""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX"""
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file):
    """Extract text from TXT"""
    return file.read().decode("utf-8")

def get_resume_text(uploaded_file):
    """Detect file type and extract text"""
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        st.error("❌ Unsupported file type. Please upload PDF, DOCX, or TXT.")
        return ""

def summarize_text(text, max_words=60):
    """Simple extractive summary"""
    sentences = text.split('. ')
    sentences = sorted(sentences, key=len, reverse=True)
    summary = '. '.join(sentences[:3])
    return summary[:max_words * 10] + "..."

def calculate_match(jd, resume):
    """Semantic similarity between JD and resume"""
    jd_embed = model.encode(jd, convert_to_tensor=True)
    resume_embed = model.encode(resume, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(jd_embed, resume_embed)
    return float(similarity.item()) * 100

# Inputs
default_jd = "Business Data Analyst JD: Analyze data, prepare dashboards, work with SQL, Excel, and Power BI. Support data-driven business decisions."
jd_text = st.text_area("💼 Job Description", value=default_jd, height=150)

uploaded_resume = st.file_uploader("📤 Upload Candidate Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if st.button("🚀 Start Analysis"):
    if uploaded_resume is None:
        st.error("Please upload a resume file first.")
    else:
        with st.spinner("Analyzing resume..."):
            resume_text = get_resume_text(uploaded_resume)
            if not resume_text.strip():
                st.error("Could not extract text from the resume file.")
            else:
                score = calculate_match(jd_text, resume_text)
                jd_summary = summarize_text(jd_text)
                resume_summary = summarize_text(resume_text)

                st.success(f"✅ Match Score: {score:.2f}%")
                st.subheader("📊 Summary")
                st.write("**Job Description Summary:**", jd_summary)
                st.write("**Resume Summary:**", resume_summary)

                if score > 70:
                    st.success("Excellent Match! 🎯 The resume fits most job criteria.")
                elif score > 50:
                    st.info("Moderate Match 🙂 Some improvements may help.")
                else:
                    st.warning("Low Match ⚠️ Resume needs better alignment with the JD.")
