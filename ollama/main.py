from agno.agent import Agent , RunResponse
from agno.models.ollama import Ollama
from rich.pretty import pprint
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.models.ollama.tools import OllamaTools
import fitz  # PyMuPDF

ats_agent = Agent(
description="An AI-powered ATS (Applicant Tracking System) expert that analyzes resumes against job descriptions using the latest parsing algorithms. Provides data-driven scoring and prioritized optimization tips to maximize interview chances.",
    model=Ollama(id="deepseek-r1" , structured_outputs=True),
    role="You are an ATS Compliance Auditor specializing in modern resume screening technologies",
 instructions=[
    "You are an expert in Applicant Tracking System (ATS) technologies specializing in PDF resume analysis.",
    "Your primary objectives are to:",
    "1. Evaluate PDF resumes against the latest ATS standards using job descriptions as your scoring benchmark",
    "2. Provide specific technical feedback to optimize PDF resumes for ATS parsing",
    "To achieve these objectives:",
    "1. **PDF-Specific ATS Evaluation Protocol:**",
    "   - For every PDF resume + job description pair:",
    "     - **Keyword Analysis (50% weight):**",
    "       - Extract mandatory keywords from the job description",
    "       - Check for:",
    "         - Keyword presence in both Skills and Experience sections ensuring that relevant keywords are present and contextually appropriate.",
    "         - Contextual placement (LSI keywords) in readable text layers",
    "         - Natural keyword density (1-2% per critical term)",
    "     - **PDF Formatting Audit (30% weight):**",
    "       - Flag:",
    "         - Text embedded in images (unreadable by ATS)",
    "         - Complex layouts that may parse incorrectly",
    "         - Non-standard headers (use 'Work Experience' not 'Career Path')",
    "       - Verify:",
    "         - All text is selectable (not image-based)",
    "         - Simple bullet point structures",
    "         - Standard fonts (Arial, Calibri, Times New Roman)",
    "     - **Experience Alignment (20% weight):**",
    "       - Confirm:",
    "         - Reverse-chronological order",
    "         - Quantified achievements (e.g., 'Increased sales by 30%')",
    "         - Years of experience match job requirements",
    "2. **PDF-Specific Feedback Delivery:**",
    "   - Generate outputs with this structure:",
    "     - **ATS Score:** [0-100] with weighting breakdown",
    "     - **✅ Strengths:** 3-5 compliant elements",
    "Offer specific recommendations to enhance the resume's ATS compatibility and overall effectiveness, such as:",
    "Suggestions for improving keyword usage and placement",
    "Advice on formatting adjustments to ensure proper ATS parsing",
    "Recommendations for emphasizing relevant skills and experiences",
    "     - **⚠️ PDF-Specific Weaknesses:**",
    "       - 'Text appears to be image-based (non-selectable)'",
    "       - 'Complex layout may cause parsing errors in [ATS platform]'",
    # "     - **🔧 Top 3 PDF Fixes:**",
    # "       - 'Convert all text to selectable content (not images)'",
    # "       - 'Simplify layout to single-column structure'",
    # "       - 'Replace header [X] with standard ATS-readable format'",
    # "3. **PDF Parsing Limitations:**",
    # "   - Always note: 'Some ATS platforms parse PDFs less accurately than native formats'",
    # "   - Identify text layer issues: 'Found [X]% of text that may be image-based'",
    # "   - Recommend: 'For critical applications, verify with the [ATS platform]'s PDF parser'",
    # "Key prohibitions:",
    "- Never suggest converting to other formats - focus only on PDF optimization",
    "- Never assume perfect PDF parsing - always account for platform-specific variances",
    "- Never overlook text layer issues in PDFs"
],
tools=[
OllamaTools(id="DuckDuckGo")
],
markdown=True,
show_tool_calls=True,
# response_model=ATSAgent,
# expected_output="A detailed report on the ATS score of the candidate"
) 

def extract_text_with_pymupdf(pdf_file) -> str:
    """Extract text from a PDF using PyMuPDF (fitz).
    
    Args:
        pdf_file: Uploaded file object from Streamlit.
    
    Returns:
        str: Combined text from all pages.
    """
    text_content = []
    
    # Open the PDF file from bytes
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text = page.get_text()
            if text.strip():  # Skip empty pages
                text_content.append(text)
    
    return "\n".join(text_content)

st.set_page_config(page_title="ATS CV Analyzer 📊")


st.title("ATS CV Analyzer")

user_input = st.file_uploader("Upload your CV/Resume", type=['pdf'])
job_description = st.text_area("Enter the job description", height=200)


if st.button("Analyze"):
    with st.spinner("Loading..."):
        resume = extract_text_with_pymupdf(user_input)
        response :  RunResponse = ats_agent.run(f"The Candidate Resume is :{resume}\n and the Job Description is: {job_description}")
        st.text(response.content)
