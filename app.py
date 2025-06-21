import streamlit as st
import subprocess
import sys
import fitz  # PyMuPDF
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Download model using subprocess (robust and interactive)
try:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
except subprocess.CalledProcessError as e:
    st.error("❌ Failed to download spaCy model.")
    st.stop()

# ✅ Try loading the model after download
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.error("❌ Still couldn't load spaCy model. Check logs.")
    st.stop()


# Safe PDF extraction
def extract_text_from_pdf(uploaded_file):
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                try:
                    text += page.get_text()
                except:
                    continue
            return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    except Exception as e:
        return f"ERROR:: {str(e)}"

def extract_keywords(text):
    try:
        doc = nlp(text.lower())
        return ' '.join([token.text for token in doc if token.is_alpha and not token.is_stop])
    except Exception as e:
        return f"ERROR:: {str(e)}"

def compute_match_score(resume_text, jd_text):
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_text, jd_text])
        score = cosine_similarity(vectors[0:1], vectors[1:2])
        return round(score[0][0] * 100, 2)
    except Exception as e:
        return f"ERROR:: {str(e)}"

def get_top_keywords(text, top_n=10):
    try:
        doc = nlp(text.lower())
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
        freq = {}
        for word in keywords:
            freq[word] = freq.get(word, 0) + 1
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw[0] for kw in sorted_keywords[:top_n]]
    except:
        return []

# ---------------- UI ----------------
st.set_page_config(page_title="Resume Screener")
st.title("📄 Resume Screener using NLP")
st.write("Upload your resume and job description to check your match score.")

uploaded_file = st.file_uploader("📎 Upload Resume (PDF)", type="pdf")

default_jd = """
We are looking for a candidate with strong skills in Python, Machine Learning, Natural Language Processing (NLP),
TensorFlow or PyTorch, and REST API development. Understanding of data preprocessing and model deployment is a plus.
"""
job_description = st.text_area("📝 Paste Job Description", value=default_jd, height=200)

if uploaded_file and job_description:
    with st.spinner("Extracting and analyzing resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        if resume_text.startswith("ERROR::"):
            st.error(f"❌ Resume reading failed: {resume_text}")
            st.stop()

        resume_clean = extract_keywords(resume_text)
        if resume_clean.startswith("ERROR::"):
            st.error(f"❌ NLP processing failed: {resume_clean}")
            st.stop()

        jd_clean = extract_keywords(job_description)
        score = compute_match_score(resume_clean, jd_clean)
        if isinstance(score, str) and score.startswith("ERROR::"):
            st.error(f"❌ Scoring failed: {score}")
            st.stop()

        st.success(f"✅ Resume Match Score: {score}%")

        st.markdown("### 📌 Explanation")
        if score >= 75:
            st.info("Excellent match. Resume strongly aligns with the JD.")
        elif score >= 50:
            st.info("Moderate match. Covers many areas, but can be improved.")
        else:
            st.warning("Low match. Consider revising your resume for relevance.")

        st.markdown("### 🔍 Resume Keywords")
        st.write(", ".join(get_top_keywords(resume_text)))

        st.markdown("### 📋 JD Keywords")
        st.write(", ".join(get_top_keywords(job_description)))
