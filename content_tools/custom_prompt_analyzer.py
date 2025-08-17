import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import os
from datetime import datetime

# Try to import optional dependencies
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# --- Configuration ---
st.set_page_config(
    page_title="Custom Prompt Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background-color: #fafafa;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        border-radius: 0 0 20px 20px;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 300;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
        margin-bottom: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .content-card {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 15px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
    }
    
    .prompt-examples {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .result-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- File Processing ---
class FileProcessor:
    """Handle different file types and extract text content"""
    
    @staticmethod
    def extract_text_from_file(uploaded_file):
        """Extract text from various file types"""
        if uploaded_file is None:
            return None
        
        file_type = uploaded_file.type
        file_name = uploaded_file.name.lower()
        
        try:
            if file_type == "text/plain" or file_name.endswith('.txt'):
                return FileProcessor._extract_from_txt(uploaded_file)
            elif file_name.endswith('.md'):
                return FileProcessor._extract_from_txt(uploaded_file)
            elif file_type == "application/pdf" or file_name.endswith('.pdf'):
                return FileProcessor._extract_from_pdf(uploaded_file)
            elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                              "application/msword"] or file_name.endswith(('.docx', '.doc')):
                return FileProcessor._extract_from_docx(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_type}")
                return None
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
    
    @staticmethod
    def _extract_from_txt(uploaded_file):
        """Extract text from txt/md files"""
        try:
            return str(uploaded_file.read(), "utf-8")
        except:
            return uploaded_file.read().decode("utf-8")
    
    @staticmethod
    def _extract_from_pdf(uploaded_file):
        """Extract text from PDF files"""
        if not PDF_AVAILABLE:
            st.error("PDF support not available. Please install PyPDF2: pip install PyPDF2")
            return None
        
        try:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None
    
    @staticmethod
    def _extract_from_docx(uploaded_file):
        """Extract text from Word documents"""
        if not DOCX_AVAILABLE:
            st.error("Word document support not available. Please install python-docx: pip install python-docx")
            return None
        
        try:
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading Word document: {str(e)}")
            return None
    
    @staticmethod
    def get_supported_formats():
        """Get list of supported file formats"""
        formats = ["txt", "md"]
        if PDF_AVAILABLE:
            formats.append("pdf")
        if DOCX_AVAILABLE:
            formats.extend(["docx", "doc"])
        return formats

# --- AI Analysis Class ---
class CustomPromptAnalyzer:
    """Analyze documents with custom user prompts"""
    
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key
    
    def analyze_with_prompt(self, text, prompt):
        """Analyze text using custom prompt"""
        if self.openai_api_key and OPENAI_AVAILABLE:
            return self._ai_analysis(text, prompt)
        else:
            return self._basic_analysis(text, prompt)
    
    def _ai_analysis(self, text, prompt):
        """Use OpenAI for custom analysis"""
        try:
            # Limit text length for API
            text_limited = text[:4000] if len(text) > 4000 else text
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful document analyst. Provide detailed, structured analysis based on the user's prompt."},
                    {"role": "user", "content": f"Document content:\n{text_limited}\n\nAnalysis request: {prompt}\n\nPlease provide a detailed analysis."}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return {
                'analysis': response.choices[0].message.content,
                'method': 'AI-powered',
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        except Exception as e:
            st.error(f"AI analysis failed: {str(e)}")
            return self._basic_analysis(text, prompt)
    
    def _basic_analysis(self, text, prompt):
        """Basic analysis without AI"""
        words = text.lower().split()
        sentences = re.split(r'[.!?]+', text)
        
        # Basic keyword extraction based on prompt
        prompt_keywords = re.findall(r'\b\w{3,}\b', prompt.lower())
        found_keywords = []
        
        for keyword in prompt_keywords:
            if keyword in text.lower():
                count = text.lower().count(keyword)
                found_keywords.append(f"{keyword} ({count} times)")
        
        # Basic sentiment
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'wonderful', 'amazing']
        negative_words = ['bad', 'terrible', 'awful', 'negative', 'failure', 'horrible', 'disappointing']
        
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        sentiment = "Neutral"
        if pos_count > neg_count:
            sentiment = "Positive"
        elif neg_count > pos_count:
            sentiment = "Negative"
        
        # Generate basic analysis
        analysis = f"""
**Basic Analysis for: "{prompt}"**

**Document Overview:**
- Word Count: {len(words)}
- Sentence Count: {len([s for s in sentences if s.strip()])}
- Character Count: {len(text)}

**Content Analysis:**
- Overall Sentiment: {sentiment}
- Positive indicators: {pos_count}
- Negative indicators: {neg_count}

**Keyword Relevance:**
{chr(10).join(found_keywords) if found_keywords else 'No specific keywords from your prompt found in the document.'}

**Summary:**
This document appears to be {sentiment.lower()} in tone. Based on your prompt "{prompt}", I've identified the above patterns. For more detailed analysis, consider providing an OpenAI API key for AI-powered insights.
        """
        
        return {
            'analysis': analysis.strip(),
            'method': 'Basic pattern matching',
            'word_count': len(words),
            'char_count': len(text),
            'sentiment': sentiment,
            'keywords_found': found_keywords
        }

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🤖 Custom Prompt Analyzer</h1>
    <p>Upload any document and ask specific questions or request custom analysis</p>
</div>
""", unsafe_allow_html=True)

# --- API Key Configuration ---
openai_api_key = os.environ.get("OPENAI_API_KEY")
try:
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

if not openai_api_key:
    with st.sidebar:
        st.header("⚙️ Configuration")
        openai_api_key = st.text_input("OpenAI API Key (optional)", type="password", 
                                      help="For AI-powered analysis. Leave empty for basic analysis.")

if openai_api_key and OPENAI_AVAILABLE:
    openai.api_key = openai_api_key

# --- Main Interface ---
st.markdown("""
<div class="content-card">
    <h3>📁 Upload Your Document</h3>
    <p>Support for PDF, Word documents, text files, and Markdown files</p>
</div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Choose a file to analyze",
    type=FileProcessor.get_supported_formats(),
    help=f"Supported formats: {', '.join(FileProcessor.get_supported_formats())}"
)

# Extract text from uploaded file
document_text = ""
if uploaded_file is not None:
    with st.spinner("📖 Processing document..."):
        document_text = FileProcessor.extract_text_from_file(uploaded_file)
    
    if document_text:
        st.success(f"✅ Successfully processed {uploaded_file.name}")
        
        # Show document preview
        with st.expander("📄 Document Preview", expanded=False):
            st.text_area("Content preview:", document_text[:1000] + "..." if len(document_text) > 1000 else document_text, height=200)
    else:
        st.error("❌ Failed to process the document")

# --- Custom Prompt Section ---
st.markdown("""
<div class="content-card">
    <h3>💬 What would you like to know?</h3>
    <p>Ask any question or request specific analysis about your document</p>
</div>
""", unsafe_allow_html=True)

# Prompt examples
st.markdown("""
<div class="prompt-examples">
    <h4>💡 Example Prompts:</h4>
    <ul>
        <li><strong>Summarization:</strong> "Summarize the main points of this document"</li>
        <li><strong>Sentiment:</strong> "What is the overall sentiment and tone?"</li>
        <li><strong>Key Insights:</strong> "What are the most important insights or conclusions?"</li>
        <li><strong>Specific Questions:</strong> "What does this document say about market trends?"</li>
        <li><strong>Analysis:</strong> "Analyze the strengths and weaknesses mentioned"</li>
        <li><strong>Recommendations:</strong> "What recommendations or action items are suggested?"</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Custom prompt input
user_prompt = st.text_area(
    "Enter your analysis prompt:",
    placeholder="e.g., 'Summarize the key findings and recommendations from this report'",
    height=100
)

# Analysis section
if document_text and user_prompt and st.button("🔍 Analyze Document", type="primary"):
    analyzer = CustomPromptAnalyzer(openai_api_key)
    
    with st.spinner("🤖 Analyzing document with your prompt..."):
        results = analyzer.analyze_with_prompt(document_text, user_prompt)
    
    # Display results
    st.markdown("""
    <div class="result-section">
        <h2>📊 Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Main analysis
    st.subheader(f"Analysis for: \"{user_prompt}\"")
    st.markdown(results['analysis'])
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Words", results['word_count'])
    with col2:
        st.metric("Characters", results['char_count'])
    with col3:
        st.metric("Method", results['method'])
    
    # Additional insights if available
    if 'sentiment' in results:
        st.subheader("📈 Additional Insights")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Detected Sentiment:** {results['sentiment']}")
        
        with col2:
            if results.get('keywords_found'):
                st.write("**Keywords Found:**")
                for keyword in results['keywords_found']:
                    st.write(f"• {keyword}")
    
    # Word cloud if available
    if WORDCLOUD_AVAILABLE and len(document_text) > 50:
        st.subheader("☁️ Document Word Cloud")
        try:
            # Clean text for word cloud
            clean_text = re.sub(r'[^\w\s]', '', document_text.lower())
            words = clean_text.split()
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
            
            if filtered_words:
                wordcloud_text = ' '.join(filtered_words)
                wc = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
        except Exception as e:
            st.info(f"Word cloud generation not available: {e}")
    
    # Export option
    st.subheader("📤 Export Analysis")
    
    export_text = f"""
CUSTOM PROMPT ANALYSIS
======================

Document: {uploaded_file.name if uploaded_file else 'Text Input'}
Prompt: {user_prompt}
Analysis Method: {results['method']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS RESULTS:
{results['analysis']}

DOCUMENT STATISTICS:
- Word Count: {results['word_count']}
- Character Count: {results['char_count']}
    """
    
    st.download_button(
        label="💾 Download Analysis",
        data=export_text,
        file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# --- Instructions if no content ---
if not document_text and not user_prompt:
    st.markdown("""
    <div class="content-card">
        <h3>🚀 How to Use</h3>
        <ol>
            <li><strong>Upload a document</strong> - PDF, Word, text, or Markdown file</li>
            <li><strong>Write your prompt</strong> - Ask any question or request specific analysis</li>
            <li><strong>Get instant insights</strong> - AI-powered or basic analysis based on your setup</li>
        </ol>
        
        <h4>✨ Features:</h4>
        <ul>
            <li>🤖 <strong>AI Analysis</strong> - With OpenAI API key for detailed insights</li>
            <li>🔍 <strong>Basic Analysis</strong> - Works without API key for pattern detection</li>
            <li>📊 <strong>Visual Results</strong> - Word clouds and statistics</li>
            <li>📤 <strong>Export Options</strong> - Download your analysis results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div style='margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; color: white; text-align: center;'>
    <h4>🤖 Custom Prompt Analyzer</h4>
    <p>Transform any document into actionable insights with custom prompts</p>
    <small>Upload • Prompt • Analyze • Export</small>
</div>
""", unsafe_allow_html=True)
