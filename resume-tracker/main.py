import streamlit as st
import spacy
import PyPDF2
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

# Extract text from the PDF
def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF."""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Preprocess text
def preprocess_text(text):
    """Clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation and special characters
    text = re.sub(r'[\*#]', '', text)     # Remove markdown symbols
    return ' '.join(text.split())

# Extract name using NER
def extract_name_from_text(text):
    """Extract name from resume text using Named Entity Recognition (NER)."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text  # Return the first identified person entity
    return "Unknown Applicant"

# Extract potential skills
def extract_skills_from_text(text):
    """Extract potential skills from text using NLP."""
    doc = nlp(text)
    skills = set()
    for token in doc:
        if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN']:
            skills.add(token.text.lower())
    return skills

# Analyze resume against job description
def analyze_resume(resume_text, job_desc):
    """Analyze resume against job description."""
    processed_resume = preprocess_text(resume_text)
    processed_job = preprocess_text(job_desc)
    
    # Extract skills
    resume_skills = extract_skills_from_text(resume_text)
    job_skills = extract_skills_from_text(job_desc)
    
    # Calculate matches
    matching_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills - resume_skills
    
    # Calculate match percentage
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([processed_resume, processed_job])
    match_percentage = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
    
    return {
        'match_percentage': match_percentage,
        'matching_skills': matching_skills,
        'missing_skills': missing_skills
    }

# Streamlit UI
st.set_page_config(page_title="Smart ATS Resume Analyzer")
st.header("Resume Analyzer")

job_description = st.text_area("Job Description:", height=200)
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

if uploaded_file is not None:
    st.success("PDF Uploaded Successfully")

if st.button("Analyze Resume"):
    if uploaded_file is not None and job_description:
        try:
            # Extract text and analyze
            resume_text = extract_text_from_pdf(uploaded_file.read())
            applicant_name = extract_name_from_text(resume_text)
            results = analyze_resume(resume_text, job_description)
            
            # Display Applicant Name
            st.subheader(f"Applicant Name: {applicant_name}")
            
            # Display ATS Score
            st.subheader("ATS Score")
            score = results['match_percentage']
            st.metric("Match Percentage (Out of 100)", f"{score:.1f}", help="Higher is better.")
            
            # Skills Analysis
            st.subheader("Skills Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("✅ **Skills You Have**")
                st.markdown(
                    '<div style="display: flex; flex-wrap: wrap;">' +
                    ''.join(f'<span style="background-color: #90EE90; color: #000; border-radius: 5px; padding: 5px 10px; margin: 5px;">{skill}</span>'
                            for skill in sorted(results['matching_skills'])) +
                    '</div>', unsafe_allow_html=True
                )
            
            with col2:
                st.markdown("❌ **Missing Skills**")
                st.markdown(
                    '<div style="display: flex; flex-wrap: wrap;">' +
                    ''.join(f'<span style="background-color: #FFB6C1; color: #000; border-radius: 5px; padding: 5px 10px; margin: 5px;">{skill}</span>'
                            for skill in sorted(results['missing_skills'])) +
                    '</div>', unsafe_allow_html=True
                )
            
            # Suggestions for improvement
            if score < 85:
                st.subheader("Suggestions to Improve Your ATS Score")
                st.write("1. Add the missing skills to your resume (if you possess them).")
                st.write("2. Use clear section headings like *Experience*, *Education*, and *Skills*.")
                st.write("3. Avoid using images, tables, or complex formatting.")
                st.write("4. Use keywords from the job description naturally in your resume.")
                st.write("5. Quantify your achievements with numbers or metrics.")
            
            # Additional dynamic suggestions
            if score < 50:
                st.subheader("Suggested Videos to Improve Your Resume or Skills")
                st.write("[How to Write a Winning Resume](https://youtu.be/y8YH0Qbu5h4)")
                st.write("[Tips for ATS-Friendly Resumes](https://youtu.be/J-4Fv8nq1iA)")
                st.write("[Improve Your Skills with Free Courses](https://youtu.be/yp693O87GmM)")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please make sure the PDF is readable and try again.")
    else:
        st.warning("Please upload a resume and provide a job description.")
