import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
import re
import os
from datetime import datetime
import json
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import io
import sqlite3
import hashlib

# Import feedback system
try:
    from feedback_system import render_enhanced_feedback_section, render_feedback_analytics
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False
    def render_enhanced_feedback_section(*args, **kwargs):
        st.warning("Feedback system not available")
    def render_feedback_analytics():
        st.warning("Feedback analytics not available")

# Import persona testing system
try:
    from persona_testing_system import render_persona_testing_interface
    PERSONA_TESTING_AVAILABLE = True
except ImportError:
    PERSONA_TESTING_AVAILABLE = False
    def render_persona_testing_interface():
        st.warning("Persona testing system not available")

# Optional imports with error handling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass

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

# --- Configuration ---
st.set_page_config(
    page_title="Content Analysis Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS inspired by Simmons & Schmid ---
st.markdown("""
<style>
    /* Import Google Fonts - similar to S&S clean typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background-color: #fafafa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Ultra-modern header with glassmorphism */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 30%, #06b6d4 70%, #10b981 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: 0 0 3rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3),
                    0 10px 20px rgba(0, 0, 0, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.15) 0%, transparent 50%, rgba(255,255,255,0.05) 100%);
        pointer-events: none;
    }
    
    .main-header::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
        pointer-events: none;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3),
                     0 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, rgba(255,255,255,1) 0%, rgba(255,255,255,0.8) 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
        margin-bottom: 0.5rem;
    }
    
    .main-header .features {
        font-size: 0.9rem;
        opacity: 0.8;
        font-style: italic;
    }
    
    /* Enhanced sidebar with glassmorphism */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(226,232,240,0.8);
    }
    
    .css-1lcbmhc {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    /* Ultra-sleek buttons with advanced effects */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25), 
                    0 1px 3px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        text-transform: none;
        letter-spacing: 0.025em;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4), 
                    0 4px 12px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #0891b2 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    /* Metric cards styling */
    .css-1xarl3l {
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    /* Modern tab styling with glassmorphism */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        background: rgba(248, 250, 252, 0.8);
        backdrop-filter: blur(10px);
        padding: 0.75rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        color: #64748b;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3),
                    0 2px 8px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: rgba(99, 102, 241, 0.1);
        color: #6366f1;
        transform: translateY(-1px);
    }
    
    /* Badge system for analysis modes */
    .new-badge::after {
        content: 'NEW!';
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        animation: pulse 2s infinite;
    }
    
    .simple-badge::after {
        content: 'SIMPLE';
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: linear-gradient(135deg, #06b6d4, #0891b2);
        color: white;
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        padding: 0.75rem;
        font-weight: 500;
    }
    
    /* Text area and input styling */
    .stTextArea textarea, .stTextInput input, .stSelectbox select {
        border-radius: 10px;
        border: 1px solid #e9ecef;
        padding: 0.75rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Success/info/error styling */
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        color: #155724;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        color: #0c5460;
    }
    
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        color: #721c24;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Section headers */
    .section-header {
        color: #2c3e50;
        font-weight: 600;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    /* Premium content cards with advanced styling */
    .content-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.1),
                    0 4px 15px rgba(0, 0, 0, 0.05),
                    inset 0 1px 0 rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.6);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .content-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
        border-radius: 20px 20px 0 0;
    }
    
    .content-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 20px 40px rgba(99, 102, 241, 0.15),
                    0 10px 25px rgba(0, 0, 0, 0.1);
        border-color: rgba(99, 102, 241, 0.3);
    }
    
    /* Enhanced analysis mode cards */
    .analysis-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.08),
                    0 3px 10px rgba(0, 0, 0, 0.05);
        border: 2px solid rgba(226, 232, 240, 0.8);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .analysis-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        transform: translateX(-100%);
        transition: transform 0.5s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 50px rgba(99, 102, 241, 0.2),
                    0 10px 20px rgba(0, 0, 0, 0.1);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    .analysis-card:hover::after {
        transform: translateX(0);
    }
    
    .analysis-card.featured {
        border-color: rgba(99, 102, 241, 0.6);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.03) 0%, rgba(139, 92, 246, 0.03) 100%);
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.15);
    }
    
    .analysis-card.featured::after {
        transform: translateX(0);
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
        height: 6px;
    }
    
    /* Subtle animations */
    .element-container {
        transition: all 0.3s ease;
    }
    
    .element-container:hover {
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
**🔬 Content Analysis Tool**
- **Content Scoring**: Analyze readability, sentiment, engagement, SEO, and quality
- **Basic Summarization**: Simple text analysis without heavy models
- **File Support**: Upload PDF, Word, Text, and Markdown files
- **Reliable Operation**: Optimized for consistent performance
""")

# --- API Key Setup ---
openai_api_key = os.environ.get("OPENAI_API_KEY")
try:
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass
if not openai_api_key:
    openai_api_key = st.text_input("Enter your OpenAI API Key (optional)", type="password")

# Try to import OpenAI (optional dependency)
try:
    import openai
    OPENAI_AVAILABLE = True
    if openai_api_key:
        openai.api_key = openai_api_key
except ImportError:
    OPENAI_AVAILABLE = False
    # OpenAI is optional - the tool works fine without it

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    AI_MODELS_AVAILABLE = True
except ImportError:
    AI_MODELS_AVAILABLE = False

# --- Header with S&S inspired styling ---
st.markdown("""
<div class="main-header">
    <h1>🔬 Content Analysis Tool</h1>
</div>
""", unsafe_allow_html=True)

# --- File Processing ---
class FileProcessor:
    """Handle different file types and extract text content"""
    
    @staticmethod
    def extract_text_from_file(uploaded_file):
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'txt':
                return FileProcessor._extract_from_txt(uploaded_file)
            elif file_extension == 'pdf':
                return FileProcessor._extract_from_pdf(uploaded_file)
            elif file_extension in ['docx', 'doc']:
                return FileProcessor._extract_from_docx(uploaded_file)
            elif file_extension == 'md':
                return FileProcessor._extract_from_markdown(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return None
                
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return None
    
    @staticmethod
    def _extract_from_txt(uploaded_file):
        return uploaded_file.read().decode("utf-8")
    
    @staticmethod
    def _extract_from_pdf(uploaded_file):
        if not PDF_AVAILABLE:
            st.error("PDF support not available. Please install PyPDF2")
            return None
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None
    
    @staticmethod
    def _extract_from_docx(uploaded_file):
        if not DOCX_AVAILABLE:
            st.error("Word document support not available")
            return None
        try:
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return None
    
    @staticmethod
    def _extract_from_markdown(uploaded_file):
        content = uploaded_file.read().decode("utf-8")
        # Remove markdown formatting
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
        content = re.sub(r'\*(.*?)\*', r'\1', content)
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        return content
    
    @staticmethod
    def get_supported_formats():
        formats = ["txt", "md"]
        if PDF_AVAILABLE:
            formats.append("pdf")
        if DOCX_AVAILABLE:
            formats.extend(["docx", "doc"])
        return formats

# --- Content Analysis Classes ---
class ContentScorer:
    """Professional content scoring for marketing analysis"""
    
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key
    
    def readability_score(self, text):
        """Calculate readability metrics"""
        try:
            flesch_ease = flesch_reading_ease(text)
            fk_grade = flesch_kincaid_grade(text)
            ari = automated_readability_index(text)
            readability_score = max(0, min(100, flesch_ease))
            
            return {
                'flesch_ease': flesch_ease,
                'fk_grade': fk_grade,
                'ari': ari,
                'readability_score': readability_score,
                'readability_level': self._get_readability_level(flesch_ease)
            }
        except:
            return {'flesch_ease': 0, 'fk_grade': 0, 'ari': 0, 'readability_score': 50, 'readability_level': 'Unknown'}
    
    def _get_readability_level(self, flesch_score):
        if flesch_score >= 90: return "Very Easy"
        elif flesch_score >= 80: return "Easy"
        elif flesch_score >= 70: return "Fairly Easy"
        elif flesch_score >= 60: return "Standard"
        elif flesch_score >= 50: return "Fairly Difficult"
        elif flesch_score >= 30: return "Difficult"
        else: return "Very Difficult"
    
    def sentiment_score(self, text, use_llm=False):
        """Enhanced sentiment analysis with optional LLM support"""
        if use_llm and self.openai_api_key and OPENAI_AVAILABLE:
            return self._llm_sentiment_analysis(text)
        else:
            return self._basic_sentiment_analysis(text)
    
    def _basic_sentiment_analysis(self, text):
        """Basic sentiment analysis using word lists"""
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like',
            'best', 'awesome', 'perfect', 'brilliant', 'outstanding', 'superb', 'magnificent',
            'happy', 'pleased', 'satisfied', 'delighted', 'thrilled', 'excited', 'positive',
            'successful', 'effective', 'powerful', 'incredible', 'marvelous', 'exceptional'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike', 'poor',
            'disappointing', 'frustrating', 'annoying', 'sad', 'angry', 'upset', 'negative',
            'problem', 'issue', 'difficult', 'hard', 'challenging', 'failed', 'wrong',
            'broken', 'useless', 'worthless', 'disaster', 'nightmare', 'catastrophe'
        ]
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {'sentiment': 'NEUTRAL', 'confidence': 0.5, 'sentiment_score': 50, 'positive_words': 0, 'negative_words': 0, 'method': 'Traditional Word Lists'}
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        if positive_ratio > negative_ratio and positive_ratio > 0:
            sentiment = 'POSITIVE'
            confidence = min(0.8, positive_ratio * 10)
            sentiment_score = 50 + (confidence * 50)
        elif negative_ratio > positive_ratio and negative_ratio > 0:
            sentiment = 'NEGATIVE'
            confidence = min(0.8, negative_ratio * 10)
            sentiment_score = 50 - (confidence * 50)
        else:
            sentiment = 'NEUTRAL'
            confidence = 0.5
            sentiment_score = 50
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'method': 'Traditional Word Lists'
        }
    
    def _llm_sentiment_analysis(self, text):
        """LLM-powered sentiment analysis"""
        try:
            import openai
            
            # Set up OpenAI client
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Limit text for API
            text_limited = text[:2000] if len(text) > 2000 else text
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Analyze the sentiment of the given text and provide a score from 0-100 (0=very negative, 50=neutral, 100=very positive). Also identify the dominant sentiment (POSITIVE, NEGATIVE, or NEUTRAL) and confidence level (0-1). Focus on overall tone, emotional language, and implicit sentiment."},
                    {"role": "user", "content": f"Analyze the sentiment of this content:\n\n{text_limited}\n\nProvide your analysis in this format:\nSentiment: [POSITIVE/NEGATIVE/NEUTRAL]\nScore: [0-100]\nConfidence: [0-1]\nReasoning: [brief explanation]"}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse the AI response
            import re
            sentiment_match = re.search(r'Sentiment:\s*(POSITIVE|NEGATIVE|NEUTRAL)', ai_response, re.IGNORECASE)
            score_match = re.search(r'Score:\s*(\d+)', ai_response)
            confidence_match = re.search(r'Confidence:\s*([\d.]+)', ai_response)
            
            sentiment = sentiment_match.group(1).upper() if sentiment_match else 'NEUTRAL'
            sentiment_score = float(score_match.group(1)) if score_match else 50
            confidence = float(confidence_match.group(1)) if confidence_match else 0.7
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'sentiment_score': sentiment_score,
                'positive_words': 'AI Analysis',
                'negative_words': 'AI Analysis',
                'method': 'LLM-Powered Analysis',
                'ai_reasoning': ai_response
            }
            
        except Exception as e:
            st.warning(f"LLM sentiment analysis failed: {e}. Using traditional analysis.")
            return self._basic_sentiment_analysis(text)
    
    def engagement_score(self, text, use_llm=False):
        """Enhanced engagement scoring with optional LLM support"""
        if use_llm and self.openai_api_key and OPENAI_AVAILABLE:
            return self._llm_engagement_analysis(text)
        else:
            return self._basic_engagement_analysis(text)
    
    def _basic_engagement_analysis(self, text):
        """Calculate engagement potential using traditional metrics"""
        words = len(text.split())
        word_score = min(100, (words / 300) * 100) if words <= 300 else max(0, 100 - ((words - 300) / 50) * 10)
        
        questions = text.count('?')
        question_score = min(100, questions * 20)
        
        exclamations = text.count('!')
        excitement_score = min(100, exclamations * 15)
        
        cta_words = ['click', 'buy', 'subscribe', 'join', 'download', 'learn', 'discover', 'try', 'get', 'start', 'register', 'contact']
        cta_count = sum([text.lower().count(word) for word in cta_words])
        cta_score = min(100, cta_count * 25)
        
        power_words = ['amazing', 'incredible', 'exclusive', 'proven', 'guaranteed', 'instant', 'ultimate', 'secret', 'powerful', 'revolutionary']
        power_count = sum([text.lower().count(word) for word in power_words])
        power_score = min(100, power_count * 20)
        
        personal_pronouns = ['you', 'your', 'we', 'us', 'our']
        pronoun_count = sum([text.lower().count(word) for word in personal_pronouns])
        pronoun_score = min(100, pronoun_count * 10)
        
        engagement_score = (word_score * 0.25 + question_score * 0.15 + excitement_score * 0.1 + 
                          cta_score * 0.2 + power_score * 0.15 + pronoun_score * 0.15)
        
        return {
            'engagement_score': engagement_score,
            'word_count': words,
            'questions': questions,
            'exclamations': exclamations,
            'cta_count': cta_count,
            'power_words': power_count,
            'personal_pronouns': pronoun_count,
            'method': 'Traditional Metrics'
        }
    
    def _llm_engagement_analysis(self, text):
        """LLM-powered engagement analysis"""
        try:
            import openai
            
            # Set up OpenAI client
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Limit text for API
            text_limited = text[:2000] if len(text) > 2000 else text
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an engagement and persuasion expert. Analyze how engaging and persuasive this content is on a scale of 0-100. Consider factors like: compelling hooks, emotional appeal, call-to-action effectiveness, storytelling elements, audience connection, clarity of value proposition, and overall persuasiveness."},
                    {"role": "user", "content": f"Analyze the engagement potential of this content:\n\n{text_limited}\n\nProvide analysis in this format:\nEngagement Score: [0-100]\nKey Strengths: [list main engaging elements]\nAreas for Improvement: [suggestions]\nOverall Assessment: [brief summary]"}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse the AI response
            import re
            score_match = re.search(r'Engagement Score:\s*(\d+)', ai_response)
            engagement_score = float(score_match.group(1)) if score_match else 75
            
            # Also get basic metrics for comparison
            basic_results = self._basic_engagement_analysis(text)
            
            return {
                'engagement_score': engagement_score,
                'word_count': basic_results['word_count'],
                'questions': basic_results['questions'],
                'exclamations': basic_results['exclamations'],
                'cta_count': basic_results['cta_count'],
                'power_words': basic_results['power_words'],
                'personal_pronouns': basic_results['personal_pronouns'],
                'method': 'LLM-Powered Analysis',
                'ai_reasoning': ai_response
            }
            
        except Exception as e:
            st.warning(f"LLM engagement analysis failed: {e}. Using traditional analysis.")
            return self._basic_engagement_analysis(text)
    
    def seo_score(self, text, target_keywords=None):
        """Calculate SEO metrics"""
        words = text.split()
        
        if target_keywords:
            keyword_density = {}
            for keyword in target_keywords:
                count = text.lower().count(keyword.lower())
                density = (count / len(words)) * 100 if words else 0
                keyword_density[keyword] = density
            avg_density = np.mean(list(keyword_density.values()))
            keyword_score = min(100, avg_density * 20)
        else:
            keyword_density = {}
            keyword_score = 50
        
        # Optimal length scoring
        if len(words) < 100:
            length_score = len(words) * 0.5
        elif len(words) <= 300:
            length_score = 50 + ((len(words) - 100) / 200) * 50
        elif len(words) <= 2000:
            length_score = 100
        else:
            length_score = max(50, 100 - ((len(words) - 2000) / 100) * 5)
        
        headings = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
        heading_score = min(100, headings * 25)
        
        bullets = len(re.findall(r'^\s*[•\-\*]\s', text, re.MULTILINE))
        list_score = min(100, bullets * 10)
        
        seo_score = (keyword_score * 0.3 + length_score * 0.3 + heading_score * 0.2 + list_score * 0.2)
        
        return {
            'seo_score': seo_score,
            'keyword_density': keyword_density,
            'word_count': len(words),
            'headings_count': headings,
            'bullet_points': bullets
        }
    
    def ai_quality_score(self, text):
        """AI-powered content quality assessment"""
        if not self.openai_api_key or not OPENAI_AVAILABLE:
            return {'quality_score': 75, 'feedback': 'OpenAI API key not provided - using basic quality assessment', 'method': 'Basic Assessment'}
        
        try:
            import openai
            
            # Set up OpenAI client
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Limit text for API
            text_limited = text[:1500] if len(text) > 1500 else text
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a content quality expert. Rate the following content on a scale of 0-100 based on clarity, coherence, value, and overall quality. Consider factors like: logical flow, grammar, usefulness to reader, depth of information, and professional presentation."},
                    {"role": "user", "content": f"Please rate this content for overall quality:\n\n{text_limited}\n\nProvide analysis in this format:\nQuality Score: [0-100]\nStrengths: [key positive aspects]\nWeaknesses: [areas for improvement]\nOverall Assessment: [brief summary]"}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse the AI response
            import re
            score_match = re.search(r'Quality Score:\s*(\d+)', ai_response)
            quality_score = float(score_match.group(1)) if score_match else 75
            
            return {
                'quality_score': quality_score,
                'feedback': ai_response,
                'method': 'AI-Powered Analysis'
            }
            
        except Exception as e:
            # Basic quality assessment based on text characteristics
            basic_score = self._basic_quality_assessment(text)
            return {
                'quality_score': basic_score,
                'feedback': f'AI analysis failed ({e}). Basic quality assessment used.',
                'method': 'Basic Assessment (AI failed)'
            }
    
    def _basic_quality_assessment(self, text):
        """Basic quality assessment based on text characteristics"""
        words = text.split()
        sentences = text.split('.')
        
        # Length assessment
        word_count = len(words)
        word_score = 100 if 50 <= word_count <= 800 else max(20, 100 - abs(word_count - 400) / 10)
        
        # Sentence length assessment
        avg_sentence_length = word_count / max(len(sentences), 1)
        sentence_score = 100 if 10 <= avg_sentence_length <= 25 else max(20, 100 - abs(avg_sentence_length - 17) * 3)
        
        # Basic repetition check
        unique_words = len(set(words))
        repetition_score = min(100, (unique_words / max(word_count, 1)) * 100 * 1.5)
        
        return (word_score * 0.4 + sentence_score * 0.3 + repetition_score * 0.3)
    
    def comprehensive_score(self, text, target_keywords=None, use_llm=False):
        """Generate comprehensive content score with optional LLM enhancements"""
        readability = self.readability_score(text)
        sentiment = self.sentiment_score(text, use_llm=use_llm)
        engagement = self.engagement_score(text, use_llm=use_llm)
        seo = self.seo_score(text, target_keywords)
        ai_quality = self.ai_quality_score(text) if use_llm else {'quality_score': 75, 'feedback': 'LLM analysis not enabled', 'method': 'Basic Assessment'}
        
        # Adjust weights based on whether AI analysis is used
        if use_llm and self.openai_api_key:
            overall_score = (
                readability['readability_score'] * 0.2 +
                sentiment['sentiment_score'] * 0.15 +
                engagement['engagement_score'] * 0.25 +
                seo['seo_score'] * 0.2 +
                ai_quality['quality_score'] * 0.2
            )
        else:
            overall_score = (
                readability['readability_score'] * 0.25 +
                sentiment['sentiment_score'] * 0.2 +
                engagement['engagement_score'] * 0.3 +
                seo['seo_score'] * 0.25
            )
        
        return {
            'overall_score': overall_score,
            'readability': readability,
            'sentiment': sentiment,
            'engagement': engagement,
            'seo': seo,
            'ai_quality': ai_quality,
            'timestamp': datetime.now().isoformat()
        }

# --- Model Loading Functions ---
@st.cache_resource
def load_summarization_model(model_name="facebook/bart-large-cnn"):
    """Load summarization model with caching"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        summarizer = pipeline("summarization", model=model_name, device=-1)  # Use CPU
        return summarizer
    except Exception as e:
        st.error(f"Failed to load {model_name}: {e}")
        return None

@st.cache_resource
def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Load sentence embedding model with caching"""
    if not AI_MODELS_AVAILABLE:
        return None
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model {model_name}: {e}")
        return None

class MessagingFrameworkAnalyzer:
    """Analyze B2B messaging frameworks and positioning"""
    
    def __init__(self):
        self.messaging_pillars = {
            'value_propositions': ['value', 'benefit', 'advantage', 'solution', 'results', 'outcome', 'roi', 'return'],
            'differentiators': ['unique', 'only', 'exclusive', 'different', 'unlike', 'competitive', 'advantage', 'edge'],
            'proof_points': ['proven', 'track record', 'success', 'case study', 'testimonial', 'award', 'certified', 'trusted'],
            'target_audience': ['customers', 'clients', 'businesses', 'companies', 'enterprises', 'organizations', 'industry'],
            'pain_points': ['challenge', 'problem', 'pain', 'difficulty', 'struggle', 'issue', 'concern', 'frustration'],
            'capabilities': ['expertise', 'capability', 'service', 'solution', 'technology', 'platform', 'tool', 'feature']
        }
    
    def analyze_messaging_framework(self, text):
        """Extract key messaging components from text"""
        text_lower = text.lower()
        
        framework = {}
        
        # Analyze each pillar
        for pillar, keywords in self.messaging_pillars.items():
            matches = []
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for keyword in keywords:
                    if keyword in sentence_lower:
                        matches.append(sentence.strip())
                        break
            
            framework[pillar] = {
                'count': len(matches),
                'examples': matches[:3],  # Top 3 examples
                'strength': self._calculate_strength(len(matches), len(sentences))
            }
        
        return framework
    
    def _calculate_strength(self, matches, total_sentences):
        """Calculate messaging strength score"""
        if total_sentences == 0:
            return 0
        percentage = (matches / total_sentences) * 100
        
        if percentage >= 15:
            return "Strong"
        elif percentage >= 8:
            return "Moderate" 
        elif percentage >= 3:
            return "Weak"
        else:
            return "Missing"
    
    def generate_messaging_house(self, text):
        """Generate a messaging house structure"""
        framework = self.analyze_messaging_framework(text)
        
        house = {
            'brand_promise': self._extract_brand_promise(text, framework),
            'value_pillars': self._extract_value_pillars(framework),
            'supporting_messages': self._extract_supporting_messages(framework),
            'proof_points': framework['proof_points']['examples'][:3],
            'target_insights': self._extract_target_insights(framework)
        }
        
        return house
    
    def _extract_brand_promise(self, text, framework):
        """Extract or suggest brand promise"""
        # Look for sentences with strong value propositions
        value_examples = framework['value_propositions']['examples']
        diff_examples = framework['differentiators']['examples']
        
        if value_examples:
            return value_examples[0]  # Best value prop sentence
        elif diff_examples:
            return diff_examples[0]   # Best differentiator
        else:
            return "Brand promise not clearly identified - consider strengthening value messaging"
    
    def _extract_value_pillars(self, framework):
        """Extract 3-4 key value pillars"""
        pillars = []
        
        # Combine capabilities and differentiators
        capabilities = framework['capabilities']['examples'][:2]
        differentiators = framework['differentiators']['examples'][:2]
        
        pillars.extend(capabilities)
        pillars.extend(differentiators)
        
        return pillars[:4] if pillars else ["Value pillars need to be more clearly defined"]
    
    def _extract_supporting_messages(self, framework):
        """Extract supporting messages for each pillar"""
        messages = []
        
        for pillar_name, pillar_data in framework.items():
            if pillar_data['examples'] and pillar_name not in ['proof_points']:
                messages.extend(pillar_data['examples'][:1])  # One example per pillar
        
        return messages[:6]  # Limit to 6 supporting messages
    
    def _extract_target_insights(self, framework):
        """Extract target audience insights"""
        audience_examples = framework['target_audience']['examples']
        pain_examples = framework['pain_points']['examples']
        
        insights = {
            'audience_clarity': "Clear" if audience_examples else "Needs definition",
            'pain_awareness': "Strong" if pain_examples else "Missing",
            'examples': audience_examples[:2] + pain_examples[:2]
        }
        
        return insights
    
    def score_messaging_maturity(self, framework):
        """Score overall messaging maturity"""
        scores = {}
        total_score = 0
        
        for pillar, data in framework.items():
            strength = data['strength']
            if strength == "Strong":
                score = 4
            elif strength == "Moderate":
                score = 3
            elif strength == "Weak":
                score = 2
            else:
                score = 1
            
            scores[pillar] = score
            total_score += score
        
        max_score = len(framework) * 4
        percentage = (total_score / max_score) * 100
        
        if percentage >= 80:
            maturity = "Advanced"
        elif percentage >= 60:
            maturity = "Developing"
        elif percentage >= 40:
            maturity = "Basic"
        else:
            maturity = "Foundational"
        
        return {
            'overall_score': round(percentage, 1),
            'maturity_level': maturity,
            'pillar_scores': scores,
            'recommendations': self._generate_recommendations(scores)
        }
    
    def _generate_recommendations(self, scores):
        """Generate improvement recommendations"""
        recommendations = []
        
        weak_areas = [pillar for pillar, score in scores.items() if score <= 2]
        
        for area in weak_areas:
            if area == 'value_propositions':
                recommendations.append("Strengthen value propositions - clearly articulate customer benefits")
            elif area == 'differentiators':
                recommendations.append("Enhance differentiation - what makes you uniquely valuable?")
            elif area == 'proof_points':
                recommendations.append("Add credible proof points - case studies, testimonials, data")
            elif area == 'target_audience':
                recommendations.append("Clarify target audience - be more specific about who you serve")
            elif area == 'pain_points':
                recommendations.append("Address customer pain points - show understanding of their challenges")
            elif area == 'capabilities':
                recommendations.append("Better articulate capabilities - what can you deliver?")
        
        return recommendations[:3]  # Top 3 recommendations

class AICompetitiveAnalyzer:
    """AI-powered competitive messaging analysis with semantic understanding"""
    
    def __init__(self):
        self.brand_archetypes = {
            'innovator': ['innovative', 'cutting-edge', 'revolutionary', 'breakthrough', 'pioneering', 'advanced', 'next-generation', 'disruptive'],
            'trusted_partner': ['trusted', 'reliable', 'proven', 'experienced', 'established', 'dependable', 'secure', 'stable'],
            'challenger': ['challenge', 'disrupt', 'transform', 'change', 'revolution', 'different', 'bold', 'fearless'],
            'expert': ['expertise', 'specialized', 'professional', 'technical', 'knowledge', 'skilled', 'certified', 'authority'],
            'human_friendly': ['personal', 'friendly', 'approachable', 'caring', 'understanding', 'supportive', 'helpful', 'empathetic'],
            'premium': ['premium', 'luxury', 'exclusive', 'elite', 'superior', 'high-end', 'quality', 'excellence']
        }
        
        self.messaging_themes = {
            'value': ['value', 'benefit', 'advantage', 'worth', 'return', 'investment', 'cost-effective', 'savings'],
            'innovation': ['innovation', 'technology', 'advanced', 'modern', 'future', 'digital', 'smart', 'intelligent'],
            'trust': ['trust', 'security', 'safety', 'protection', 'reliability', 'confidence', 'assurance', 'guarantee'],
            'growth': ['growth', 'scale', 'expand', 'increase', 'improve', 'optimize', 'enhance', 'accelerate'],
            'efficiency': ['efficiency', 'streamline', 'automate', 'simplify', 'fast', 'quick', 'easy', 'convenient'],
            'partnership': ['partnership', 'collaboration', 'together', 'support', 'service', 'relationship', 'team', 'community']
        }
    
    def get_semantic_embeddings(self, texts):
        """Get semantic embeddings for texts"""
        embedding_model = load_embedding_model()
        if embedding_model is None:
            return None
        
        try:
            embeddings = embedding_model.encode(texts)
            return embeddings
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return None
    
    def analyze_brand_archetype(self, text):
        """Analyze brand archetype using keyword matching and semantic similarity"""
        text_lower = text.lower()
        archetype_scores = {}
        
        # Keyword-based scoring
        for archetype, keywords in self.brand_archetypes.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize by text length
            archetype_scores[archetype] = score / max(len(text.split()), 1) * 100
        
        # Get top archetypes
        sorted_archetypes = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'primary_archetype': sorted_archetypes[0][0] if sorted_archetypes[0][1] > 0 else 'undefined',
            'archetype_scores': dict(sorted_archetypes),
            'archetype_strength': sorted_archetypes[0][1] if sorted_archetypes else 0
        }
    
    def analyze_messaging_themes(self, text):
        """Analyze messaging themes"""
        text_lower = text.lower()
        theme_scores = {}
        
        for theme, keywords in self.messaging_themes.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            theme_scores[theme] = score / max(len(text.split()), 1) * 100
        
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'theme_scores': dict(sorted_themes),
            'dominant_themes': [theme for theme, score in sorted_themes[:3] if score > 0],
            'theme_diversity': len([score for score in theme_scores.values() if score > 0])
        }
    
    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        embeddings = self.get_semantic_embeddings([text1, text2])
        if embeddings is None:
            return 0.0
        
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def find_unique_messaging(self, client_text, competitor_texts):
        """Find messaging elements unique to client vs competitors"""
        client_words = set(client_text.lower().split())
        competitor_words = set()
        
        for comp_text in competitor_texts:
            competitor_words.update(comp_text.lower().split())
        
        # Find unique words (simple approach)
        unique_to_client = client_words - competitor_words
        unique_to_competitors = competitor_words - client_words
        common_words = client_words & competitor_words
        
        return {
            'unique_to_client': list(unique_to_client)[:10],  # Top 10
            'unique_to_competitors': list(unique_to_competitors)[:10],
            'common_words': list(common_words)[:10],
            'differentiation_score': len(unique_to_client) / max(len(client_words), 1) * 100
        }
    
    def generate_competitive_insights(self, client_name, client_text, competitors):
        """Generate AI-powered competitive insights"""
        if not OPENAI_AVAILABLE or not openai_api_key:
            return self._generate_basic_insights(client_name, client_text, competitors)
        
        try:
            # Prepare competitor summary
            comp_summary = "\n".join([f"- {comp['name']}: {comp['text'][:200]}..." for comp in competitors])
            
            prompt = f"""
            Analyze the messaging comparison between {client_name} and their competitors.
            
            CLIENT ({client_name}):
            {client_text[:500]}...
            
            COMPETITORS:
            {comp_summary}
            
            Provide a strategic analysis focusing on:
            1. Key messaging differentiators for {client_name}
            2. Gaps in the competitive landscape
            3. Opportunities for positioning
            4. Tone and brand personality differences
            
            Keep the analysis concise and actionable for marketing strategy.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a marketing strategist analyzing competitive messaging for B2B brands."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return {
                'insights': response.choices[0].message.content,
                'method': 'AI-Generated (OpenAI)',
                'confidence': 'High'
            }
            
        except Exception as e:
            st.error(f"AI analysis failed: {e}")
            return self._generate_basic_insights(client_name, client_text, competitors)
    
    def _generate_basic_insights(self, client_name, client_text, competitors):
        """Generate basic competitive insights without AI"""
        client_archetype = self.analyze_brand_archetype(client_text)
        client_themes = self.analyze_messaging_themes(client_text)
        
        insights = []
        insights.append(f"{client_name} primary brand archetype: {client_archetype['primary_archetype'].title()}")
        insights.append(f"Dominant messaging themes: {', '.join(client_themes['dominant_themes'])}")
        
        # Compare with competitors
        competitor_archetypes = []
        for comp in competitors:
            comp_archetype = self.analyze_brand_archetype(comp['text'])
            competitor_archetypes.append(comp_archetype['primary_archetype'])
        
        unique_archetype = client_archetype['primary_archetype'] not in competitor_archetypes
        if unique_archetype:
            insights.append(f"✅ {client_name} has a unique brand archetype in this competitive set")
        else:
            insights.append(f"⚠️ {client_name} shares archetype similarities with competitors")
        
        return {
            'insights': '\n'.join(insights),
            'method': 'Rule-based Analysis',
            'confidence': 'Moderate'
        }
    
    def comprehensive_competitive_analysis(self, client_name, client_text, competitors, openai_api_key=None):
        """Run comprehensive competitive analysis"""
        results = {
            'client_analysis': {
                'name': client_name,
                'archetype': self.analyze_brand_archetype(client_text),
                'themes': self.analyze_messaging_themes(client_text),
                'word_count': len(client_text.split())
            },
            'competitor_analyses': [],
            'comparative_insights': {},
            'recommendations': []
        }
        
        # Analyze each competitor
        for comp in competitors:
            comp_analysis = {
                'name': comp['name'],
                'archetype': self.analyze_brand_archetype(comp['text']),
                'themes': self.analyze_messaging_themes(comp['text']),
                'similarity_to_client': self.calculate_semantic_similarity(client_text, comp['text']),
                'word_count': len(comp['text'].split())
            }
            results['competitor_analyses'].append(comp_analysis)
        
        # Generate comparative insights
        competitor_texts = [comp['text'] for comp in competitors]
        unique_messaging = self.find_unique_messaging(client_text, competitor_texts)
        ai_insights = self.generate_competitive_insights(client_name, client_text, competitors)
        
        results['comparative_insights'] = {
            'unique_messaging': unique_messaging,
            'ai_insights': ai_insights,
            'semantic_similarities': [comp['similarity_to_client'] for comp in results['competitor_analyses']]
        }
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _generate_recommendations(self, analysis_results):
        """Generate strategic recommendations based on analysis"""
        recommendations = []
        
        client = analysis_results['client_analysis']
        competitors = analysis_results['competitor_analyses']
        insights = analysis_results['comparative_insights']
        
        # Archetype recommendations
        client_archetype = client['archetype']['primary_archetype']
        competitor_archetypes = [comp['archetype']['primary_archetype'] for comp in competitors]
        
        if client_archetype in competitor_archetypes:
            recommendations.append(f"🎯 Consider strengthening your '{client_archetype}' positioning or exploring adjacent archetypes for differentiation")
        else:
            recommendations.append(f"✅ Your '{client_archetype}' archetype is unique in this competitive set - leverage this advantage")
        
        # Similarity recommendations
        high_similarity_comps = [comp['name'] for comp in competitors if comp['similarity_to_client'] > 0.7]
        if high_similarity_comps:
            recommendations.append(f"⚠️ High messaging similarity with {', '.join(high_similarity_comps)} - focus on unique value propositions")
        
        # Differentiation recommendations
        diff_score = insights['unique_messaging']['differentiation_score']
        if diff_score < 20:
            recommendations.append("🔄 Low differentiation score - consider developing more distinctive messaging")
        elif diff_score > 50:
            recommendations.append("🌟 Strong differentiation - maintain and amplify unique messaging elements")
        
        return recommendations[:5]  # Top 5 recommendations

class EnhancedSummarizer:
    """Enhanced text summarization with multiple model options"""
    
    def __init__(self):
        self.available_models = self._get_available_models()
    
    def _get_available_models(self):
        """Get list of available summarization models"""
        models = ["Basic (Rule-based)"]
        
        if TRANSFORMERS_AVAILABLE:
            models.extend([
                "BART (Hugging Face)",
                "T5-Small (Hugging Face)", 
                "DistilBART (Faster)"
            ])
        
        if OPENAI_AVAILABLE:
            models.append("GPT-3.5 (OpenAI)")
        
        return models
    
    def summarize_with_model(self, text, model_choice, max_length=150, openai_api_key=None):
        """Summarize text using selected model"""
        
        if model_choice == "Basic (Rule-based)":
            return self._basic_summary(text, max_length)
        
        elif model_choice == "BART (Hugging Face)":
            return self._huggingface_summary(text, "facebook/bart-large-cnn", max_length)
        
        elif model_choice == "T5-Small (Hugging Face)":
            return self._huggingface_summary(text, "t5-small", max_length)
        
        elif model_choice == "DistilBART (Faster)":
            return self._huggingface_summary(text, "sshleifer/distilbart-cnn-12-6", max_length)
        
        elif model_choice == "GPT-3.5 (OpenAI)" and openai_api_key:
            return self._openai_summary(text, max_length, openai_api_key)
        
        else:
            return self._basic_summary(text, max_length)
    
    def _basic_summary(self, text, max_length):
        """Rule-based summarization"""
        sentences = self.extract_key_sentences(text, min(5, max_length // 30))
        summary = ' '.join(sentences[:3])
        
        return {
            'summary': summary,
            'method': 'Rule-based extraction',
            'confidence': 'Basic',
            'word_count': len(summary.split())
        }
    
    def _huggingface_summary(self, text, model_name, max_length):
        """Hugging Face model summarization"""
        if not TRANSFORMERS_AVAILABLE:
            return self._basic_summary(text, max_length)
        
        try:
            with st.spinner(f"Loading {model_name} model..."):
                summarizer = load_summarization_model(model_name)
            
            if summarizer is None:
                return self._basic_summary(text, max_length)
            
            # Limit input text length for model
            input_text = text[:1024] if len(text) > 1024 else text
            
            with st.spinner("Generating summary..."):
                result = summarizer(
                    input_text,
                    max_length=max_length,
                    min_length=max_length // 3,
                    do_sample=False
                )
            
            summary = result[0]['summary_text']
            
            return {
                'summary': summary,
                'method': f'Hugging Face ({model_name})',
                'confidence': 'High',
                'word_count': len(summary.split())
            }
            
        except Exception as e:
            st.error(f"Hugging Face summarization failed: {e}")
            return self._basic_summary(text, max_length)
    
    def _openai_summary(self, text, max_length, api_key):
        """OpenAI GPT summarization"""
        if not OPENAI_AVAILABLE or not api_key:
            return self._basic_summary(text, max_length)
        
        try:
            # Limit text for API
            input_text = text[:3000] if len(text) > 3000 else text
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"Summarize the following text in approximately {max_length} words. Focus on key points and maintain clarity."},
                    {"role": "user", "content": input_text}
                ],
                max_tokens=max_length * 2,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content
            
            return {
                'summary': summary,
                'method': 'OpenAI GPT-3.5',
                'confidence': 'Very High',
                'word_count': len(summary.split())
            }
            
        except Exception as e:
            st.error(f"OpenAI summarization failed: {e}")
            return self._basic_summary(text, max_length)
    
    def extract_key_sentences(self, text, num_sentences=3):
        """Extract key sentences based on word frequency"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= num_sentences:
            return sentences
        
        # Count word frequency
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Score sentences based on word frequency
        sentence_scores = []
        for sentence in sentences:
            sentence_words = re.findall(r'\b\w+\b', sentence.lower())
            score = sum(word_freq[word] for word in sentence_words)
            sentence_scores.append((score, sentence))
        
        # Get top sentences
        sentence_scores.sort(reverse=True)
        top_sentences = [sent for _, sent in sentence_scores[:num_sentences]]
        
        return top_sentences
    
    def get_word_frequency_summary(self, text):
        """Get summary based on word frequency analysis"""
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        filtered_freq = {word: freq for word, freq in word_freq.items() if word not in stop_words and len(word) > 2}
        
        # Get top words
        top_words = dict(Counter(filtered_freq).most_common(10))
        
        return {
            'top_words': top_words,
            'total_words': len(words),
            'unique_words': len(set(words))
        }
    
    def extract_themes(self, text):
        """Extract main themes from text using keyword clustering"""
        # Clean and tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'said', 'say', 'says', 'get', 'got', 'one', 'two', 'first', 'last', 'also', 'just', 'now', 'well', 'way', 'new', 'make', 'made', 'take', 'come', 'go', 'see', 'know', 'think', 'time', 'year', 'day', 'back', 'use', 'used', 'using', 'work', 'working', 'works', 'people', 'person'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_freq = Counter(filtered_words)
        
        # Get top themes (most frequent meaningful words)
        theme_words = dict(word_freq.most_common(8))
        
        # Create theme categories based on common patterns
        business_words = ['business', 'company', 'market', 'customer', 'product', 'service', 'sales', 'revenue', 'profit', 'strategy', 'growth', 'management', 'team', 'organization', 'industry', 'competitive', 'client', 'partnership']
        tech_words = ['technology', 'software', 'system', 'platform', 'digital', 'data', 'analytics', 'automation', 'innovation', 'development', 'solution', 'technical', 'programming', 'algorithm', 'database', 'network', 'security', 'cloud']
        process_words = ['process', 'workflow', 'procedure', 'method', 'approach', 'framework', 'implementation', 'execution', 'planning', 'project', 'task', 'activity', 'operation', 'function', 'system', 'structure']
        people_words = ['leadership', 'communication', 'collaboration', 'relationship', 'culture', 'employee', 'staff', 'talent', 'skills', 'training', 'development', 'performance', 'motivation', 'engagement', 'feedback']
        
        # Define comprehensive theme categories with keywords
        theme_categories = {
            'Technology & Innovation': ['technology', 'digital', 'innovation', 'software', 'platform', 'automation', 'artificial', 'intelligence', 'data', 'analytics', 'tech', 'system', 'solution', 'development', 'cloud', 'mobile', 'app', 'online', 'algorithm', 'programming', 'database', 'network', 'security'],
            'Business & Strategy': ['business', 'strategy', 'growth', 'revenue', 'profit', 'market', 'competitive', 'customer', 'client', 'sales', 'marketing', 'commercial', 'enterprise', 'company', 'organization', 'management', 'leadership', 'strategic', 'operations', 'industry', 'partnership'],
            'Customer Experience': ['customer', 'user', 'experience', 'service', 'support', 'satisfaction', 'feedback', 'engagement', 'relationship', 'communication', 'interaction', 'journey', 'touchpoint', 'personalization', 'quality', 'client'],
            'Process & Efficiency': ['process', 'efficiency', 'workflow', 'productivity', 'optimization', 'streamline', 'automation', 'improvement', 'performance', 'operations', 'procedures', 'methodology', 'framework', 'structure', 'implementation', 'execution'],
            'Communication & Collaboration': ['communication', 'collaboration', 'team', 'meeting', 'discussion', 'feedback', 'sharing', 'coordination', 'partnership', 'stakeholder', 'engagement', 'dialogue', 'interaction', 'teamwork'],
            'Challenges & Pain Points': ['challenge', 'problem', 'issue', 'difficulty', 'concern', 'barrier', 'obstacle', 'limitation', 'constraint', 'struggle', 'frustration', 'complexity', 'risk', 'threat', 'bottleneck'],
            'Solutions & Benefits': ['solution', 'benefit', 'advantage', 'improvement', 'value', 'outcome', 'result', 'success', 'achievement', 'opportunity', 'potential', 'capability', 'feature', 'enhancement'],
            'Time & Resources': ['time', 'resource', 'budget', 'cost', 'investment', 'effort', 'capacity', 'allocation', 'planning', 'schedule', 'deadline', 'timeline', 'priority', 'funding'],
            'People & Skills': ['people', 'team', 'staff', 'employee', 'talent', 'skill', 'expertise', 'knowledge', 'training', 'development', 'capability', 'competency', 'experience', 'leadership', 'culture'],
            'Future & Goals': ['future', 'goal', 'objective', 'vision', 'plan', 'roadmap', 'strategy', 'direction', 'ambition', 'aspiration', 'target', 'milestone', 'next', 'growth']
        }
        
        # Analyze text for theme presence
        text_lower = text.lower()
        theme_scores = {}
        theme_evidence = {}
        
        for theme_name, keywords in theme_categories.items():
            score = 0
            evidence = []
            
            # Count keyword occurrences and collect evidence
            for keyword in keywords:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                if count > 0:
                    score += count
                    evidence.append(f"{keyword}({count})")
            
            if score > 0:
                theme_scores[theme_name] = score
                theme_evidence[theme_name] = evidence[:5]  # Top 5 evidence points
        
        # Sort themes by relevance
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create structured theme output
        structured_themes = {}
        enhanced_theme_words = {}
        
        for theme_name, score in sorted_themes:
            if score >= 2:  # Minimum threshold for theme relevance
                structured_themes[theme_name] = {
                    'score': score,
                    'evidence': theme_evidence[theme_name],
                    'strength': 'High' if score >= 10 else 'Medium' if score >= 5 else 'Low'
                }
                # For compatibility, also add theme name to top_theme_words
                enhanced_theme_words[theme_name] = score
        
        # If no structured themes found, fall back to original word frequency approach
        if not structured_themes:
            enhanced_theme_words = theme_words
            structured_themes = {
            'Business & Strategy': [word for word in theme_words if word in business_words],
            'Technology & Innovation': [word for word in theme_words if word in tech_words],
            'Process & Operations': [word for word in theme_words if word in process_words],
            'People & Culture': [word for word in theme_words if word in people_words],
            'General Themes': [word for word in theme_words if word not in business_words + tech_words + process_words + people_words]
        }
        # Remove empty categories
            structured_themes = {category: words for category, words in structured_themes.items() if words}
        
        return {
            'theme_categories': structured_themes,
            'top_theme_words': enhanced_theme_words,
            'theme_summary': self._generate_enhanced_theme_summary(structured_themes)
        }
    
    def _generate_theme_summary(self, themes):
        """Generate a summary of the main themes"""
        if not themes:
            return "No clear themes identified in the content."
        
        summary_parts = []
        for category, words in themes.items():
            if words:
                word_list = ', '.join(words[:3])  # Top 3 words per category
                summary_parts.append(f"{category}: {word_list}")
        
        return "Main themes identified: " + "; ".join(summary_parts) + "."
    
    def _generate_enhanced_theme_summary(self, themes):
        """Generate enhanced thematic summary with strength indicators"""
        if not themes:
            return "No clear themes identified in the content."
        
        summary_parts = []
        for category, details in themes.items():
            if isinstance(details, dict) and 'strength' in details:
                strength = details['strength']
                evidence_count = len(details.get('evidence', []))
                summary_parts.append(f"{category} ({strength} - {evidence_count} indicators)")
            elif isinstance(details, list) and details:
                summary_parts.append(f"{category} ({len(details)} terms)")
        
        if summary_parts:
            return "Key themes identified: " + "; ".join(summary_parts) + "."
        else:
            return "Content analyzed but no significant themes detected."
    
    def extract_conceptual_themes(self, text, use_ai=False, openai_api_key=None):
        """Extract actual conceptual themes from interview content using pattern recognition + optional AI"""
        try:
            # First, try AI-powered analysis if enabled and available
            if use_ai and openai_api_key and OPENAI_AVAILABLE:
                ai_themes = self._ai_theme_discovery(text, openai_api_key)
                if ai_themes:
                    return ai_themes
            
            # Split text into sentences for thematic analysis
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 10]
            
            # Extract actual themes using pattern recognition
            discovered_themes = self._discover_conceptual_themes(sentences)
            
            # If we have meaningful themes, use them
            if discovered_themes:
                return {
                    'theme_categories': discovered_themes,
                    'top_theme_words': {theme: details['weight'] for theme, details in discovered_themes.items()},
                    'theme_summary': self._generate_conceptual_theme_summary(discovered_themes)
                }
            
            # Fallback to enhanced keyword analysis
            return self.extract_themes(text)
            
        except Exception as e:
            # Final fallback to existing method
            return self.extract_themes(text)
    
    def _ai_theme_discovery(self, text, openai_api_key):
        """Use AI to discover themes in interview content"""
        try:
            import openai
            
            # Set up OpenAI client
            client = openai.OpenAI(api_key=openai_api_key)
            
            # Craft a sophisticated prompt for theme discovery
            prompt = f"""Analyze this interview transcript and identify the main themes. For each theme, provide:
1. Theme name (be specific and descriptive)
2. Strength level (High/Medium/Low based on prominence)
3. 2-3 supporting quote excerpts from the text

Focus on identifying:
- Pain points and challenges mentioned
- Process issues or inefficiencies  
- Communication problems
- Technology or tool limitations
- Time/resource constraints
- User experience concerns
- Team collaboration issues
- Training or knowledge gaps
- Opportunities for improvement
- Specific topics being discussed

Return your analysis in this exact JSON format:
{{
  "themes": {{
    "Theme Name": {{
      "strength": "High/Medium/Low",
      "evidence": ["quote 1...", "quote 2...", "quote 3..."],
      "weight": 5
    }}
  }},
  "summary": "Brief summary of main themes identified"
}}

Interview text:
{text[:3000]}"""  # Limit text length for API

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert qualitative researcher specializing in interview analysis and thematic coding."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse the AI response
            ai_response = response.choices[0].message.content
            
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_result = json.loads(json_str)
                
                if 'themes' in parsed_result:
                    return {
                        'theme_categories': parsed_result['themes'],
                        'top_theme_words': {theme: details.get('weight', 1) for theme, details in parsed_result['themes'].items()},
                        'theme_summary': parsed_result.get('summary', 'AI-generated thematic analysis completed')
                    }
            
            return None
            
        except Exception as e:
            st.warning(f"AI theme analysis failed: {str(e)}. Falling back to pattern recognition.")
            return None
    
    def _discover_conceptual_themes(self, sentences):
        """Discover actual conceptual themes from sentence content"""
        themes = {}
        
        # Define theme patterns that capture actual concepts from interviews
        theme_patterns = {
            'Communication Challenges': [
                r'hard to communicate|difficult to explain|don\'t understand|miscommunication|unclear|confusing',
                r'lost in translation|mixed messages|poor communication|communication breakdown|not getting through'
            ],
            'Time & Pressure Issues': [
                r'not enough time|time pressure|rushed|deadline|quickly|urgent|immediate',
                r'time consuming|takes too long|slow process|delayed|behind schedule'
            ],
            'User Experience Problems': [
                r'user[s]? (complain|struggle|find it hard|have trouble|are frustrated)|poor experience',
                r'difficult to use|not intuitive|confusing interface|user feedback|usability issues'
            ],
            'Process Inefficiencies': [
                r'inefficient|waste of time|redundant|unnecessary steps|could be automated',
                r'manual process|repetitive|streamline|optimize|improve workflow|bottleneck'
            ],
            'Technology Limitations': [
                r'system doesn\'t|software can\'t|technical limitation|technology constraint',
                r'outdated system|legacy|compatibility issues|technical debt|system crashes'
            ],
            'Team Collaboration Issues': [
                r'silos|disconnect between teams|lack of alignment|communication between departments',
                r'coordination problems|team conflict|working in isolation|not sharing'
            ],
            'Customer Feedback & Requests': [
                r'customers say|client feedback|user complaints|customer requests|feedback shows',
                r'customer satisfaction|client concerns|user testimonials|market research'
            ],
            'Resource Constraints': [
                r'lack of resources|limited budget|not enough staff|resource allocation',
                r'need more people|understaffed|budget constraints|cost concerns|funding issues'
            ],
            'Innovation & Opportunities': [
                r'new idea|innovative approach|creative solution|think outside|opportunity to improve',
                r'potential for|could enhance|next generation|future state|breakthrough'
            ],
            'Training & Knowledge Gaps': [
                r'need training|don\'t know how|lack of knowledge|learning curve|need to understand',
                r'knowledge transfer|skill development|competency gap|expertise needed|onboarding'
            ]
        }
        
        # Analyze sentences for conceptual themes
        for theme_name, patterns in theme_patterns.items():
            theme_evidence = []
            theme_weight = 0
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                for pattern in patterns:
                    matches = re.findall(pattern, sentence_lower)
                    if matches:
                        # Store the actual sentence as evidence
                        evidence_text = sentence[:150] + "..." if len(sentence) > 150 else sentence
                        if evidence_text not in theme_evidence:
                            theme_evidence.append(evidence_text)
                        theme_weight += len(matches)
            
            if theme_evidence:
                themes[theme_name] = {
                    'evidence': theme_evidence[:3],  # Top 3 example sentences
                    'weight': theme_weight,
                    'strength': 'High' if theme_weight >= 3 else 'Medium' if theme_weight >= 2 else 'Low'
                }
        
        # Also extract explicit topic mentions
        topic_themes = self._extract_explicit_topics(sentences)
        themes.update(topic_themes)
        
        return themes
    
    def _extract_explicit_topics(self, sentences):
        """Extract topics that are explicitly mentioned in interviews"""
        topics = {}
        
        # Look for explicit topic indicators
        topic_indicators = [
            r'talking about (.+?)(?:\.|,|;|and|but|$)',
            r'discuss(?:ing|ed)? (.+?)(?:\.|,|;|and|but|$)',
            r'focus(?:ing|ed)? on (.+?)(?:\.|,|;|and|but|$)',
            r'main (?:issue|concern|topic|theme|problem) (?:is|was) (.+?)(?:\.|,|;|and|but|$)',
            r'problem with (.+?)(?:\.|,|;|and|but|$)',
            r'challenge with (.+?)(?:\.|,|;|and|but|$)',
            r'(?:about|regarding|concerning) (.+?)(?:\.|,|;|and|but|$)'
        ]
        
        for sentence in sentences:
            for pattern in topic_indicators:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    topic = match.strip()
                    # Clean up the topic
                    topic = re.sub(r'\b(the|a|an|is|was|are|were|that|this|those|these)\b', '', topic, flags=re.IGNORECASE)
                    topic = topic.strip()
                    
                    if len(topic) > 5 and len(topic) < 60:  # Reasonable topic length
                        topic_name = f"Discussion: {topic.title()}"
                        if topic_name not in topics:
                            topics[topic_name] = {
                                'evidence': [],
                                'weight': 0,
                                'strength': 'Medium'
                            }
                        evidence_text = sentence[:150] + "..." if len(sentence) > 150 else sentence
                        if evidence_text not in topics[topic_name]['evidence']:
                            topics[topic_name]['evidence'].append(evidence_text)
                        topics[topic_name]['weight'] += 1
        
        return topics
    
    def _generate_conceptual_theme_summary(self, themes):
        """Generate summary for conceptually discovered themes"""
        if not themes:
            return "No significant themes identified in the interview content."
        
        # Separate conceptual themes from discussion topics
        conceptual_themes = [name for name in themes.keys() if not name.startswith('Discussion:')]
        discussion_topics = [name for name in themes.keys() if name.startswith('Discussion:')]
        
        summary_parts = []
        
        if conceptual_themes:
            high_impact = [name for name in conceptual_themes if themes[name]['strength'] == 'High']
            medium_impact = [name for name in conceptual_themes if themes[name]['strength'] == 'Medium']
            
            if high_impact:
                summary_parts.append(f"Key concerns: {', '.join(high_impact)}")
            if medium_impact:
                summary_parts.append(f"Secondary themes: {', '.join(medium_impact[:2])}")
        
        if discussion_topics:
            topic_names = [name.replace('Discussion: ', '') for name in discussion_topics[:3]]
            summary_parts.append(f"Main topics: {', '.join(topic_names)}")
        
        return "; ".join(summary_parts) if summary_parts else "Interview analyzed with mixed themes."
    
    def create_theme_wordcloud_data(self, text, exclude_terms=None):
        """Create data for theme-based word cloud"""
        if exclude_terms is None:
            exclude_terms = set()
        
        # Clean text for word cloud
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        words = text.split()
        
        # Remove excluded terms and stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        filtered_words = [w for w in words if w not in stop_words and w not in exclude_terms and len(w) > 2]
        
        return ' '.join(filtered_words)
    
    def compare_themes_across_texts(self, texts, text_names):
        """Compare themes across multiple texts"""
        all_themes = []
        
        for i, text in enumerate(texts):
            themes = self.extract_themes(text)
            themes['text_name'] = text_names[i]
            all_themes.append(themes)
        
        # Find common theme words across all texts
        all_theme_words = []
        for theme_data in all_themes:
            all_theme_words.extend(theme_data['top_theme_words'].keys())
        
        common_words = Counter(all_theme_words)
        common_themes = {word: count for word, count in common_words.items() if count >= len(texts) // 2}
        
        return {
            'individual_themes': all_themes,
            'common_themes': common_themes,
            'comparison_summary': self._generate_comparison_summary(all_themes, common_themes)
        }
    
    def _generate_comparison_summary(self, all_themes, common_themes):
        """Generate a summary comparing themes across texts"""
        if not common_themes:
            return "No common themes found across all documents."
        
        common_words_list = list(common_themes.keys())[:5]
        return f"Common themes across documents: {', '.join(common_words_list)}. Each document shows unique focus areas while sharing these core concepts."

# --- Sidebar Configuration ---
st.sidebar.header("🔧 Analysis Configuration")

analysis_mode = st.sidebar.selectbox(
    "Select Analysis Mode",
    ["🏠 Overview", "📋 Interview Analysis", "🎭 AI Persona Testing", "Content Scoring", "Thematic Summary", "Messaging Framework", "🥊 AI Competitive Analysis", "Custom Prompt Analysis"]
)

target_keywords = st.sidebar.text_input(
    "Target Keywords (comma-separated)",
    placeholder="keyword1, keyword2, keyword3"
)

if target_keywords:
    target_keywords = [k.strip() for k in target_keywords.split(',') if k.strip()]

# --- Feedback Analytics Section ---
st.sidebar.markdown("---")
st.sidebar.header("📊 Analytics")

if FEEDBACK_AVAILABLE:
    if st.sidebar.button("📈 View Client Feedback", help="View feedback analytics dashboard"):
        # Create a new section for analytics
        st.markdown("---")
        render_feedback_analytics()
        st.markdown("---")
else:
    st.sidebar.info("Feedback analytics unavailable")

# Set target_keywords to None if not provided
if not target_keywords:
    target_keywords = None

# --- Main Content Area ---
if analysis_mode == "🏠 Overview":
    st.markdown("## Welcome to Content Analysis Tool")
    
    st.markdown("""
    <div class="content-card">
        <p>Choose from our comprehensive suite of analysis tools designed for marketing professionals, content creators, and researchers. Each mode offers specialized insights to help you understand, optimize, and compare your content effectively.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis Modes Overview
    st.markdown("### 🔧 Available Analysis Modes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Interview Analysis - TOP PRIORITY
        st.markdown("""
        <div class='analysis-card featured simple-badge'>
            <h4 style='margin: 0 0 1rem 0; color: #1f2937; font-size: 1.25rem; font-weight: 700;'>📋 Interview Analysis</h4>
            <p style='margin: 0 0 1rem 0; color: #4b5563; font-size: 1rem; line-height: 1.6;'>
                Upload multiple interviews and get clear thematic comparisons. Simple, focused, and powerful for research insights.
            </p>
            <div style='display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;'>
                <span style='background: rgba(99, 102, 241, 0.1); color: #6366f1; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Multi-interview upload</span>
                <span style='background: rgba(6, 182, 212, 0.1); color: #06b6d4; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Thematic comparison</span>
                <span style='background: rgba(16, 185, 129, 0.1); color: #10b981; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Clear insights</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Persona Testing - NEW PREMIUM FEATURE
        st.markdown("""
        <div class='analysis-card featured new-badge'>
            <h4 style='margin: 0 0 1rem 0; color: #1f2937; font-size: 1.25rem; font-weight: 700;'>🎭 AI Persona Testing</h4>
            <p style='margin: 0 0 1rem 0; color: #4b5563; font-size: 1rem; line-height: 1.6;'>
                Test your messaging against AI decision-maker personas. Get instant feedback from CIOs, CFOs, CMOs and other key stakeholders.
            </p>
            <div style='display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;'>
                <span style='background: rgba(168, 85, 247, 0.1); color: #a855f7; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Decision-maker AI agents</span>
                <span style='background: rgba(236, 72, 153, 0.1); color: #ec4899; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Instant feedback</span>
                <span style='background: rgba(245, 101, 101, 0.1); color: #f56565; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Resonance mapping</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Content Scoring
        st.markdown("""
        <div class='analysis-card'>
            <h4 style='margin: 0 0 1rem 0; color: #1f2937; font-size: 1.25rem; font-weight: 700;'>📊 Content Scoring</h4>
            <p style='margin: 0 0 1rem 0; color: #4b5563; font-size: 1rem; line-height: 1.6;'>
                Advanced content quality assessment with AI-powered analysis, readability, engagement, and SEO scoring.
            </p>
            <div style='display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;'>
                <span style='background: rgba(139, 92, 246, 0.1); color: #8b5cf6; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>AI Quality Analysis</span>
                <span style='background: rgba(99, 102, 241, 0.1); color: #6366f1; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Readability metrics</span>
                <span style='background: rgba(6, 182, 212, 0.1); color: #06b6d4; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>SEO analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Thematic Summary  
        st.markdown("""
        <div class='analysis-card'>
            <h4 style='margin: 0 0 1rem 0; color: #1f2937; font-size: 1.25rem; font-weight: 700;'>🎯 Thematic Summary</h4>
            <p style='margin: 0 0 1rem 0; color: #4b5563; font-size: 1rem; line-height: 1.6;'>
                AI-powered summarization with multiple model options. Extract key themes, generate word clouds, and create executive summaries.
            </p>
            <div style='display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;'>
                <span style='background: rgba(139, 92, 246, 0.1); color: #8b5cf6; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Multiple AI models</span>
                <span style='background: rgba(99, 102, 241, 0.1); color: #6366f1; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Theme extraction</span>
                <span style='background: rgba(16, 185, 129, 0.1); color: #10b981; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Word clouds</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Custom Prompt Analysis
        st.markdown("""
        <div class='analysis-card'>
            <h4 style='margin: 0 0 1rem 0; color: #1f2937; font-size: 1.25rem; font-weight: 700;'>🤖 Custom Prompt Analysis</h4>
            <p style='margin: 0 0 1rem 0; color: #4b5563; font-size: 1rem; line-height: 1.6;'>
                Ask specific questions about your content using AI. Perfect for targeted analysis and custom insights.
            </p>
            <div style='display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;'>
                <span style='background: rgba(139, 92, 246, 0.1); color: #8b5cf6; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>AI-powered Q&A</span>
                <span style='background: rgba(99, 102, 241, 0.1); color: #6366f1; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Custom insights</span>
                <span style='background: rgba(6, 182, 212, 0.1); color: #06b6d4; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Flexible analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:

        
        # Messaging Framework - NEW!
        st.markdown("""
        <div class='analysis-card featured new-badge'>
            <h4 style='margin: 0 0 1rem 0; color: #1f2937; font-size: 1.25rem; font-weight: 700;'>🏗️ Messaging Framework</h4>
            <p style='margin: 0 0 1rem 0; color: #4b5563; font-size: 1rem; line-height: 1.6;'>
                Analyze B2B messaging frameworks. Generate messaging houses, assess maturity, and get strategic recommendations.
            </p>
            <div style='display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;'>
                <span style='background: rgba(245, 158, 11, 0.1); color: #f59e0b; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Messaging houses</span>
                <span style='background: rgba(99, 102, 241, 0.1); color: #6366f1; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Maturity scoring</span>
                <span style='background: rgba(139, 92, 246, 0.1); color: #8b5cf6; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Strategic insights</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Competitive Analysis - NEW!
        st.markdown("""
        <div class='analysis-card featured new-badge'>
            <h4 style='margin: 0 0 1rem 0; color: #1f2937; font-size: 1.25rem; font-weight: 700;'>🥊 AI Competitive Analysis</h4>
            <p style='margin: 0 0 1rem 0; color: #4b5563; font-size: 1rem; line-height: 1.6;'>
                AI-powered competitive messaging analysis with semantic similarity, brand archetypes, and strategic gap identification.
            </p>
            <div style='display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem;'>
                <span style='background: rgba(139, 92, 246, 0.1); color: #8b5cf6; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Semantic AI analysis</span>
                <span style='background: rgba(99, 102, 241, 0.1); color: #6366f1; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Brand archetypes</span>
                <span style='background: rgba(6, 182, 212, 0.1); color: #06b6d4; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>Competitive gaps</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        

    
    # Quick Start Section
    st.markdown("### 🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📤 Upload Content**
        
        Support for multiple file formats:
        - PDF documents
        - Word files (.docx, .doc)  
        - Text files (.txt, .md)
        - Direct text paste
        """)
    
    with col2:
        st.markdown("""
        **🔍 Choose Analysis**
        
        Select the analysis mode that fits your needs:
        - Content optimization
        - Competitive analysis
        - Messaging frameworks
        - Custom insights
        """)
    
    with col3:
        st.markdown("""
        **📊 Get Insights**
        
        Professional reports and visualizations:
        - Interactive charts
        - Downloadable reports
        - Actionable recommendations
        - Export capabilities
        """)
    
    # Getting Started CTA
    st.markdown("---")
    
    st.markdown("### 👆 **Ready to get started?**")
    st.info("**Select an analysis mode from the sidebar** to begin analyzing your content. Each mode offers specialized tools designed for different use cases and professional needs.")
    
    # Feature Highlights
    with st.expander("🌟 Key Features"):
        st.markdown("""
        **🔬 Advanced Analysis**
        - Multiple AI models (Hugging Face, OpenAI)
        - Professional scoring algorithms
        - B2B messaging frameworks
        
        **📊 Rich Visualizations**
        - Interactive charts and graphs
        - Word clouds and thematic maps
        - Comparative analysis views
        
        **💼 Professional Tools**
        - Messaging maturity assessment
        - Competitive gap analysis
        - Export-ready reports
        
        **⚡ Flexible Input**
        - Multiple file format support
        - Direct text input
        - Batch document processing
        """)

elif analysis_mode == "🎭 AI Persona Testing":
    if PERSONA_TESTING_AVAILABLE:
        render_persona_testing_interface()
    else:
        st.error("AI Persona Testing system not available. Please check your installation.")

elif analysis_mode == "📋 Interview Analysis":
    st.markdown('<h2 class="section-header">📋 Interview Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card">
        <p>Upload multiple interviews and get clear thematic comparisons. Simple, focused analysis that reveals patterns across your research.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple upload interface
    st.markdown("### 📁 Upload Your Interviews")
    
    uploaded_files = st.file_uploader(
        "Choose interview files",
        type=['txt', 'md', 'pdf', 'docx', 'doc'],
        accept_multiple_files=True,
        help="Upload 2-10 interview transcripts for comparison"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} interviews uploaded")
        
        # Process files
        interviews = []
        for file in uploaded_files:
            content = FileProcessor.extract_text_from_file(file)
            if content:
                interviews.append({
                    'name': file.name.replace('.txt', '').replace('.pdf', '').replace('.docx', ''),
                    'content': content
                })
        
        if len(interviews) >= 2:
            st.info("💡 Upload 2-10 interview files for thematic comparison analysis")
            
            # AI Analysis Options
            st.markdown("### 🧠 Analysis Options")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                use_ai_analysis = st.checkbox("🤖 Enable AI-Powered Theme Discovery", 
                                            help="Uses advanced AI to identify nuanced themes and insights")
                if use_ai_analysis:
                    st.session_state['use_ai_analysis'] = True
                else:
                    st.session_state['use_ai_analysis'] = False
            
            with col2:
                if use_ai_analysis:
                    openai_api_key = st.text_input("OpenAI API Key", 
                                                 type="password", 
                                                 help="Required for AI analysis",
                                                 value=st.session_state.get('openai_api_key', ''))
                    if openai_api_key:
                        st.session_state['openai_api_key'] = openai_api_key
                else:
                    openai_api_key = None
            
            if use_ai_analysis and not openai_api_key:
                st.warning("⚠️ OpenAI API key required for AI analysis. Will use pattern recognition as fallback.")
        
        if len(interviews) >= 2 and st.button("🔍 Analyze Interviews", type="primary"):
            
            # Use AI analysis if enabled and available
            ai_enabled = st.session_state.get('use_ai_analysis', False)
            api_key = st.session_state.get('openai_api_key', None)
            
            analysis_mode = "🧠 AI-Powered" if (ai_enabled and api_key) else "🔍 Pattern Recognition"
            
            with st.spinner(f"Analyzing interview themes using {analysis_mode} analysis..."):
                analyzer = EnhancedSummarizer()
                
                # Extract conceptual themes from each interview (with optional AI)
                interview_analyses = []
                
                for interview in interviews:
                    themes = analyzer.extract_conceptual_themes(
                        interview['content'], 
                        use_ai=ai_enabled, 
                        openai_api_key=api_key
                    )
                    word_freq = analyzer.get_word_frequency_summary(interview['content'])
                    
                    interview_analyses.append({
                        'name': interview['name'],
                        'themes': themes,
                        'word_freq': word_freq,
                        'word_count': len(interview['content'].split())
                    })
                
                # Display Results
                st.markdown("## 📊 Interview Analysis Results")
                
                # Quick overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Interviews Analyzed", len(interviews))
                with col2:
                    total_words = sum(analysis['word_count'] for analysis in interview_analyses)
                    st.metric("Total Words", f"{total_words:,}")
                with col3:
                    total_themes = 0
                    for analysis in interview_analyses:
                        theme_words = analysis['themes']['top_theme_words']
                        if isinstance(theme_words, dict):
                            total_themes += len(theme_words)
                        else:
                            total_themes += len(theme_words) if theme_words else 0
                    avg_themes = total_themes / len(interview_analyses)
                    st.metric("Avg Themes per Interview", f"{avg_themes:.1f}")
                
                # Main analysis tabs
                tab1, tab2, tab3 = st.tabs(["🎯 Common Themes", "📊 Interview Comparison", "📋 Individual Summaries"])
                
                with tab1:
                    st.markdown("### 🎯 Themes Across All Interviews")
                    
                    # Find common themes
                    all_themes = []
                    for analysis in interview_analyses:
                        theme_words = analysis['themes']['top_theme_words']
                        if isinstance(theme_words, dict):
                            all_themes.extend(list(theme_words.keys()))
                        else:
                            all_themes.extend(theme_words if theme_words else [])
                    
                    # Count theme frequency across interviews
                    from collections import Counter
                    theme_counts = Counter(all_themes)
                    common_themes = theme_counts.most_common(10)
                    
                    if common_themes:
                        # Create bar chart of common themes
                        theme_names = [theme[0] for theme in common_themes]
                        theme_frequencies = [theme[1] for theme in common_themes]
                        
                        fig = go.Figure(data=[
                            go.Bar(x=theme_names, y=theme_frequencies, 
                                  marker_color='#4ecdc4')
                        ])
                        fig.update_layout(
                            title="Most Common Themes Across Interviews",
                            xaxis_title="Themes",
                            yaxis_title="Mentions Across Interviews",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Theme insights
                        st.markdown("#### 💡 Key Insights")
                        for i, (theme, count) in enumerate(common_themes[:5], 1):
                            percentage = (count / len(interviews)) * 100
                            st.write(f"**{i}. {theme.title()}** - Mentioned in {count}/{len(interviews)} interviews ({percentage:.0f}%)")
                
                with tab2:
                    st.markdown("### 📊 Interview-by-Interview Comparison")
                    
                    # Create comparison matrix
                    comparison_data = []
                    
                    for analysis in interview_analyses:
                        # Get theme words (handle both list and dict formats)
                        theme_words = analysis['themes']['top_theme_words']
                        if isinstance(theme_words, dict):
                            theme_list = list(theme_words.keys())[:10]  # Take top 10 keys
                        else:
                            theme_list = theme_words if theme_words else []
                        
                        row = {
                            'Interview': analysis['name'],
                            'Word Count': analysis['word_count'],
                            'Unique Themes': len(theme_list),
                            'Top Theme': theme_list[0] if theme_list else 'None'
                        }
                        
                        # Add top 3 themes as separate columns
                        for i in range(3):
                            if i < len(theme_list):
                                row[f'Theme {i+1}'] = theme_list[i]
                            else:
                                row[f'Theme {i+1}'] = '-'
                        
                        comparison_data.append(row)
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Visual comparison of word counts
                    fig = go.Figure(data=[
                        go.Bar(x=[analysis['name'] for analysis in interview_analyses],
                              y=[analysis['word_count'] for analysis in interview_analyses],
                              marker_color='#ff6b6b')
                    ])
                    fig.update_layout(
                        title="Interview Length Comparison",
                        xaxis_title="Interviews",
                        yaxis_title="Word Count",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.markdown("### 📋 Individual Interview Summaries")
                    
                    for analysis in interview_analyses:
                        with st.expander(f"📄 {analysis['name']}"):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown("**Key Themes:**")
                                # Handle both list and dict formats for themes
                                theme_words = analysis['themes']['top_theme_words']
                                if isinstance(theme_words, dict):
                                    theme_list = list(theme_words.keys())[:5]
                                else:
                                    theme_list = theme_words[:5] if theme_words else []
                                
                                for theme in theme_list:
                                    # Check if we have evidence for this theme
                                    theme_details = analysis['themes']['theme_categories'].get(theme, {})
                                    if isinstance(theme_details, dict) and 'evidence' in theme_details:
                                        with st.expander(f"🔍 {theme} ({theme_details.get('strength', 'Low')})", expanded=False):
                                            st.write("**Evidence:**")
                                            for evidence in theme_details['evidence']:
                                                st.write(f"• {evidence}")
                                    else:
                                        st.write(f"• {theme}")
                                
                                if analysis['themes']['theme_summary']:
                                    st.markdown("**Summary:**")
                                    st.info(analysis['themes']['theme_summary'])
                            
                            with col2:
                                st.metric("Word Count", analysis['word_count'])
                                st.metric("Unique Words", analysis['word_freq']['unique_words'])
                                st.metric("Theme Count", len(theme_list))
                
                # Export option
                st.markdown("---")
                st.markdown("### 📄 Export Analysis")
                
                # Prepare export data
                export_lines = []
                export_lines.append("INTERVIEW THEMATIC ANALYSIS REPORT")
                export_lines.append("=" * 50)
                export_lines.append(f"Number of Interviews: {len(interviews)}")
                export_lines.append(f"Total Words Analyzed: {total_words:,}")
                export_lines.append("")
                
                export_lines.append("COMMON THEMES ACROSS INTERVIEWS:")
                for theme, count in common_themes[:10]:
                    percentage = (count / len(interviews)) * 100
                    export_lines.append(f"- {theme.title()}: {count}/{len(interviews)} interviews ({percentage:.0f}%)")
                export_lines.append("")
                
                export_lines.append("INDIVIDUAL INTERVIEW SUMMARIES:")
                for analysis in interview_analyses:
                    export_lines.append(f"\n{analysis['name'].upper()}:")
                    export_lines.append(f"Word Count: {analysis['word_count']}")
                    
                    # Handle themes safely
                    theme_words = analysis['themes']['top_theme_words']
                    if isinstance(theme_words, dict):
                        theme_list = list(theme_words.keys())[:5]
                    else:
                        theme_list = theme_words[:5] if theme_words else []
                    
                    export_lines.append("Key Themes: " + ", ".join(theme_list))
                    if analysis['themes']['theme_summary']:
                        export_lines.append(f"Summary: {analysis['themes']['theme_summary']}")
                
                export_text = "\n".join(export_lines)
                
                if st.download_button(
                    "📥 Download Interview Analysis Report",
                    data=export_text,
                    file_name=f"interview_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                ):
                    st.success("Interview analysis report downloaded!")
        
        elif len(interviews) == 1:
            st.info("📝 Upload at least 2 interviews to enable comparison analysis")
        elif len(interviews) == 0:
            st.info("📁 Please upload interview files to begin analysis")

elif analysis_mode == "Content Scoring":
    st.markdown('<h2 class="section-header">📊 Content Scoring Analysis</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content-card">
        <p>Analyze your content for readability, sentiment, engagement, SEO factors, and overall quality. 
        Get comprehensive insights with professional-grade scoring metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload File"], horizontal=True)
    
    content_input = ""
    
    if input_method == "Type/Paste Text":
        content_input = st.text_area("Enter content to analyze", height=300, placeholder="Paste your content here...")
    else:
        supported_formats = FileProcessor.get_supported_formats()
        st.info(f"📁 Supported formats: {', '.join(supported_formats).upper()}")
        
        uploaded_file = st.file_uploader("Upload your content file", type=supported_formats)
        
        if uploaded_file is not None:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                content_input = FileProcessor.extract_text_from_file(uploaded_file)
                
            if content_input:
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                with st.expander("📄 Preview extracted text"):
                    preview_text = content_input[:500] + "..." if len(content_input) > 500 else content_input
                    st.text_area("Extracted content preview:", preview_text, height=150, disabled=True)
    
    # LLM Options for Enhanced Analysis
    st.markdown("### 🤖 Analysis Options")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        use_llm_analysis = st.checkbox("🧠 Enable LLM-Powered Analysis", 
                                     help="Uses AI for enhanced sentiment, engagement, and quality scoring")
    with col2:
        if use_llm_analysis:
            openai_api_key = st.text_input("OpenAI API Key", 
                                         type="password", 
                                         help="Required for AI-enhanced scoring",
                                         value=st.session_state.get('scoring_api_key', ''))
            if openai_api_key:
                st.session_state['scoring_api_key'] = openai_api_key
        else:
            openai_api_key = None
    
    if use_llm_analysis and not openai_api_key:
        st.warning("⚠️ OpenAI API key required for LLM analysis. Will use traditional scoring as fallback.")
    
    if content_input and st.button("🔍 Analyze Content", type="primary"):
        scorer = ContentScorer(openai_api_key=openai_api_key)
        analysis_mode_text = "🧠 AI-Enhanced" if (use_llm_analysis and openai_api_key) else "📊 Traditional"
        
        with st.spinner(f"Analyzing content using {analysis_mode_text} analysis..."):
            results = scorer.comprehensive_score(content_input, target_keywords, use_llm=(use_llm_analysis and openai_api_key is not None))
        
        # Display results
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.metric("Overall Content Score", f"{results['overall_score']:.1f}/100")
        with col2:
            grade = "A+" if results['overall_score'] >= 90 else "A" if results['overall_score'] >= 80 else "B" if results['overall_score'] >= 70 else "C" if results['overall_score'] >= 60 else "D"
            st.metric("Grade", grade)
        with col3:
            word_count = len(content_input.split())
            st.metric("Word Count", word_count)
        
        # Score breakdown
        st.subheader(f"📊 Detailed Scoring Breakdown ({analysis_mode_text})")
        
        # Show analysis methods used
        if use_llm_analysis and openai_api_key:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🧠 **Sentiment Analysis**: {results['sentiment'].get('method', 'Traditional')}")
                st.info(f"🎯 **Engagement Analysis**: {results['engagement'].get('method', 'Traditional')}")
            with col2:
                st.info(f"🤖 **AI Quality Assessment**: {results['ai_quality'].get('method', 'Traditional')}")
                st.info(f"📚 **Readability & SEO**: Traditional Metrics")
        
        # Display scores
        if use_llm_analysis and openai_api_key:
            score_cols = st.columns(5)
            scores = [
                ("Readability", results['readability']['readability_score'], "📚"),
                ("Sentiment", results['sentiment']['sentiment_score'], "😊"),
                ("Engagement", results['engagement']['engagement_score'], "🎯"),
                ("SEO", results['seo']['seo_score'], "🔍"),
                ("AI Quality", results['ai_quality']['quality_score'], "🤖")
            ]
        else:
            score_cols = st.columns(4)
            scores = [
                ("Readability", results['readability']['readability_score'], "📚"),
                ("Sentiment", results['sentiment']['sentiment_score'], "😊"),
                ("Engagement", results['engagement']['engagement_score'], "🎯"),
                ("SEO", results['seo']['seo_score'], "🔍")
            ]
        
        for i, (name, score, icon) in enumerate(scores):
            with score_cols[i]:
                st.metric(f"{icon} {name}", f"{score:.1f}")
        
        # Radar chart
        st.subheader("📈 Score Visualization")
        if use_llm_analysis and openai_api_key:
            categories = ['Readability', 'Sentiment', 'Engagement', 'SEO', 'AI Quality']
            values = [
                results['readability']['readability_score'],
                results['sentiment']['sentiment_score'],
                results['engagement']['engagement_score'],
                results['seo']['seo_score'],
                results['ai_quality']['quality_score']
            ]
        else:
            categories = ['Readability', 'Sentiment', 'Engagement', 'SEO']
            values = [
                results['readability']['readability_score'],
                results['sentiment']['sentiment_score'],
                results['engagement']['engagement_score'],
                results['seo']['seo_score']
            ]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Content Score'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Content Scoring Radar Chart")
        st.plotly_chart(fig)
        
        # Enhanced AI Insights & Feedback Section
        if FEEDBACK_AVAILABLE:
            render_enhanced_feedback_section(results, content_input, use_llm_analysis, openai_api_key)
        else:
            # Fallback to original AI insights if feedback system not available
            if use_llm_analysis and openai_api_key:
                st.subheader("🧠 AI-Powered Insights")
                
                tab1, tab2, tab3 = st.tabs(["🎭 Sentiment Analysis", "🎯 Engagement Analysis", "🤖 Quality Assessment"])
                
                with tab1:
                    if 'ai_reasoning' in results['sentiment']:
                        st.markdown("### Detailed Sentiment Analysis")
                        st.write(results['sentiment']['ai_reasoning'])
                    else:
                        st.info("Using traditional sentiment analysis")
                
                with tab2:
                    if 'ai_reasoning' in results['engagement']:
                        st.markdown("### Engagement & Persuasiveness Analysis")
                        st.write(results['engagement']['ai_reasoning'])
                    else:
                        st.info("Using traditional engagement analysis")
                
                with tab3:
                    if results['ai_quality']['method'] == 'AI-Powered Analysis':
                        st.markdown("### Content Quality Assessment")
                        st.write(results['ai_quality']['feedback'])
                    else:
                        st.info("AI quality assessment not available")
        
        # Detailed insights
        st.subheader("🔍 Detailed Insights")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📚 Readability", "😊 Sentiment", "🎯 Engagement", "🔍 SEO"])
        
        with tab1:
            st.write(f"**Reading Level:** {results['readability']['readability_level']}")
            st.write(f"**Flesch Reading Ease:** {results['readability']['flesch_ease']:.1f}")
            st.write(f"**Grade Level:** {results['readability']['fk_grade']:.1f}")
            
        with tab2:
            st.write(f"**Overall Sentiment:** {results['sentiment']['sentiment']}")
            st.write(f"**Confidence:** {results['sentiment']['confidence']:.2f}")
            st.write(f"**Positive Words Found:** {results['sentiment']['positive_words']}")
            st.write(f"**Negative Words Found:** {results['sentiment']['negative_words']}")
            
        with tab3:
            eng = results['engagement']
            st.write(f"**Questions:** {eng['questions']}")
            st.write(f"**Exclamation Points:** {eng['exclamations']}")
            st.write(f"**Call-to-Action Words:** {eng['cta_count']}")
            st.write(f"**Power Words:** {eng['power_words']}")
            st.write(f"**Personal Pronouns:** {eng['personal_pronouns']}")
            
        with tab4:
            if target_keywords:
                st.write("**Keyword Density:**")
                for keyword, density in results['seo']['keyword_density'].items():
                    st.write(f"- {keyword}: {density:.2f}%")
            st.write(f"**Headings Found:** {results['seo']['headings_count']}")
            st.write(f"**Bullet Points:** {results['seo']['bullet_points']}")

elif analysis_mode == "Basic Text Analysis":
    st.markdown('<h2 class="section-header">📝 Basic Text Analysis</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content-card">
        <p>Simple yet powerful text analysis and summarization without heavy machine learning models. 
        Perfect for quick insights and thematic exploration.</p>
    </div>
    """, unsafe_allow_html=True)
    
    input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload File"], horizontal=True, key="basic_input")
    
    content_input = ""
    
    if input_method == "Type/Paste Text":
        content_input = st.text_area("Enter text to analyze", height=300, placeholder="Paste your text here...")
    else:
        supported_formats = FileProcessor.get_supported_formats()
        uploaded_file = st.file_uploader("Upload your content file", type=supported_formats, key="basic_upload")
        
        if uploaded_file is not None:
            content_input = FileProcessor.extract_text_from_file(uploaded_file)
            if content_input:
                st.success(f"✅ Processed {uploaded_file.name}")
    
    if content_input and st.button("📝 Analyze Text", type="primary"):
        summarizer = EnhancedSummarizer()
        
        with st.spinner("Analyzing text..."):
            key_sentences = summarizer.extract_key_sentences(content_input, 5)
            word_analysis = summarizer.get_word_frequency_summary(content_input)
            themes = summarizer.extract_themes(content_input)
        
        # Create tabs for organized display
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Summary", "🎯 Themes", "☁️ Word Clouds", "📊 Statistics"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📄 Key Sentences")
                for i, sentence in enumerate(key_sentences, 1):
                    st.write(f"{i}. {sentence}")
            
            with col2:
                st.subheader("📊 Quick Stats")
                words = len(content_input.split())
                sentences = len(re.split(r'[.!?]+', content_input))
                paragraphs = len([p for p in content_input.split('\n\n') if p.strip()])
                
                st.metric("Words", words)
                st.metric("Sentences", sentences)
                st.metric("Paragraphs", paragraphs)
                st.metric("Unique Words", word_analysis['unique_words'])
        
        with tab2:
            st.subheader("🎯 Thematic Analysis")
            st.write(f"**Summary:** {themes['theme_summary']}")
            
            if themes['theme_categories']:
                st.subheader("📋 Theme Categories")
                for category, words in themes['theme_categories'].items():
                    if words:
                        st.write(f"**{category}:** {', '.join(words)}")
            
            # Theme word frequency chart
            if themes['top_theme_words']:
                st.subheader("📈 Theme Word Frequency")
                theme_df = pd.DataFrame(list(themes['top_theme_words'].items()), columns=['Theme Word', 'Frequency'])
                fig = px.bar(theme_df, x='Theme Word', y='Frequency', title="Most Important Theme Words")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("☁️ General Word Cloud")
                try:
                    clean_text = summarizer.create_theme_wordcloud_data(content_input)
                    if clean_text.strip():
                        wc = WordCloud(width=400, height=300, background_color='white').generate(clean_text)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("Not enough content for word cloud")
                except:
                    st.info("Word cloud generation not available")
            
            with col2:
                st.subheader("🎯 Theme Word Cloud")
                try:
                    if themes['top_theme_words']:
                        # Create text from theme words with frequency weighting
                        theme_text = ' '.join([word + ' ' * freq for word, freq in themes['top_theme_words'].items()])
                        wc_theme = WordCloud(width=400, height=300, background_color='white').generate(theme_text)
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.imshow(wc_theme, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("No themes identified for word cloud")
                except:
                    st.info("Theme word cloud generation not available")
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔤 Most Frequent Words")
                top_words_df = pd.DataFrame(list(word_analysis['top_words'].items()), columns=['Word', 'Frequency'])
                fig = px.bar(top_words_df, x='Word', y='Frequency', title="Word Frequency Analysis")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("📈 Text Composition")
                words = len(content_input.split())
                sentences = len(re.split(r'[.!?]+', content_input))
                
                # Create a simple composition chart
                composition_data = {
                    'Metric': ['Total Words', 'Unique Words', 'Sentences', 'Avg Words/Sentence'],
                    'Value': [words, word_analysis['unique_words'], sentences, round(words/sentences if sentences > 0 else 0, 1)]
                }
                comp_df = pd.DataFrame(composition_data)
                fig = px.bar(comp_df, x='Metric', y='Value', title="Text Composition Analysis")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig)

elif analysis_mode == "Thematic Summary":
    st.markdown('<h2 class="section-header">🎯 Thematic Summary & Analysis</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content-card">
        <p>This mode focuses on extracting and visualizing the main themes from your content, similar to interview transcript analysis. 
        Perfect for understanding the core concepts and creating beautiful word maps from any text content.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input options
    input_option = st.radio(
        "Choose input method:",
        ["📝 Paste Text", "📁 Upload File"],
        horizontal=True
    )
    
    content_input = ""
    
    if input_option == "📝 Paste Text":
        content_input = st.text_area(
            "Enter your content for thematic analysis:",
            height=200,
            placeholder="Paste your text, interview transcript, meeting notes, or any content you want to analyze for themes..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a file for thematic analysis",
            type=FileProcessor.get_supported_formats(),
            help=f"Supported formats: {', '.join(FileProcessor.get_supported_formats())}"
        )
        
        if uploaded_file is not None:
            content_input = FileProcessor.extract_text_from_file(uploaded_file)
            if content_input:
                st.success(f"✅ Processed {uploaded_file.name}")
    
    # Analysis options
    col1, col2, col3 = st.columns(3)
    with col1:
        exclude_terms = st.text_input(
            "Exclude terms from analysis (comma-separated)",
            placeholder="interviewer, um, uh, like"
        )
    with col2:
        summary_length = st.selectbox(
            "Summary detail level",
            ["Brief", "Standard", "Detailed"],
            index=1
        )
    with col3:
        # Model selection for summarization
        summarizer = EnhancedSummarizer()
        model_choice = st.selectbox(
            "Summarization Model",
            summarizer.available_models,
            index=1 if len(summarizer.available_models) > 1 else 0,
            help="Choose the model for generating summaries"
        )
    
    if content_input and st.button("🎯 Generate Thematic Summary", type="primary"):
        summarizer = EnhancedSummarizer()
        
        # Parse exclude terms
        exclude_set = set()
        if exclude_terms:
            exclude_set = set([term.strip().lower() for term in exclude_terms.split(',') if term.strip()])
        
        with st.spinner("Analyzing themes and generating summary..."):
            # Enhanced summarization with model choice
            max_length = 100 if summary_length == "Brief" else 200 if summary_length == "Standard" else 350
            
            summary_result = summarizer.summarize_with_model(
                content_input, 
                model_choice, 
                max_length=max_length,
                openai_api_key=openai_api_key
            )
            
            themes = summarizer.extract_themes(content_input)
            key_sentences = summarizer.extract_key_sentences(
                content_input, 
                3 if summary_length == "Brief" else 5 if summary_length == "Standard" else 8
            )
            word_analysis = summarizer.get_word_frequency_summary(content_input)
        
        # Main Results Display
        st.subheader("📋 Executive Summary")
        
        # Enhanced summary display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📄 AI-Generated Summary")
            st.markdown(f"**Method:** {summary_result['method']} | **Confidence:** {summary_result['confidence']}")
            st.write(summary_result['summary'])
        
        with col2:
            st.markdown("### 🔍 Key Metrics")
            st.metric("Total Words", len(content_input.split()))
            st.metric("Summary Words", summary_result['word_count'])
            st.metric("Main Themes", len(themes['top_theme_words']))
            st.metric("Categories", len(themes['theme_categories']))
        
        # Theme Summary
        st.markdown("### 🎯 Thematic Overview")
        st.info(themes['theme_summary'])
        
        # Create main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Key Excerpts", "🎯 Theme Analysis", "☁️ Visual Maps", "📊 Deep Dive"])
        
        with tab1:
            st.subheader("📄 Most Important Excerpts")
            st.markdown("*The following sentences best represent the core content:*")
            
            for i, sentence in enumerate(key_sentences, 1):
                with st.expander(f"Key Point {i}", expanded=i <= 3):
                    st.write(sentence)
        
        with tab2:
            st.subheader("🎯 Detailed Theme Breakdown")
            
            # Theme categories
            if themes['theme_categories']:
                for category, words in themes['theme_categories'].items():
                    if words:
                        st.markdown(f"**{category}**")
                        # Create a nice display for theme words
                        theme_cols = st.columns(min(len(words), 4))
                        for i, word in enumerate(words):
                            with theme_cols[i % 4]:
                                st.metric(word.title(), themes['top_theme_words'].get(word, 0), 
                                         delta=None, help=f"Frequency: {themes['top_theme_words'].get(word, 0)}")
                        st.markdown("---")
            
            # Theme word frequency visualization
            if themes['top_theme_words']:
                st.subheader("📈 Theme Word Frequency")
                theme_df = pd.DataFrame(list(themes['top_theme_words'].items()), columns=['Theme Word', 'Frequency'])
                
                # Create a horizontal bar chart for better readability
                fig = px.bar(theme_df, y='Theme Word', x='Frequency', orientation='h',
                            title="Theme Words by Frequency", 
                            color='Frequency', color_continuous_scale='viridis')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("☁️ Theme Visualization Maps")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**General Word Cloud**")
                try:
                    clean_text = summarizer.create_theme_wordcloud_data(content_input, exclude_set)
                    if clean_text.strip():
                        wc = WordCloud(width=500, height=300, background_color='white', 
                                     colormap='viridis', max_words=50).generate(clean_text)
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("Not enough content for word cloud")
                except:
                    st.info("Word cloud generation not available")
            
            with col2:
                st.markdown("**Theme-Focused Cloud**")
                try:
                    if themes['top_theme_words']:
                        # Weight theme words by frequency for better visualization
                        theme_text = ' '.join([word + ' ' * min(freq, 10) for word, freq in themes['top_theme_words'].items()])
                        wc_theme = WordCloud(width=500, height=300, background_color='white',
                                          colormap='plasma', max_words=30).generate(theme_text)
                        fig, ax = plt.subplots(figsize=(7, 4))
                        ax.imshow(wc_theme, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("No themes identified")
                except:
                    st.info("Theme word cloud not available")
        
        with tab4:
            st.subheader("📊 Statistical Deep Dive")
            
            # Text statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Content Statistics**")
                words = len(content_input.split())
                unique_words = word_analysis['unique_words']
                sentences = len(re.split(r'[.!?]+', content_input))
                
                stats_data = {
                    'Metric': ['Total Words', 'Unique Words', 'Sentences', 'Avg Words/Sentence', 'Vocabulary Richness'],
                    'Value': [words, unique_words, sentences, 
                             round(words/sentences if sentences > 0 else 0, 1),
                             round(unique_words/words * 100 if words > 0 else 0, 1)]
                }
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
            
            with col2:
                st.markdown("**Word Frequency Distribution**")
                if word_analysis['top_words']:
                    freq_df = pd.DataFrame(list(word_analysis['top_words'].items()), columns=['Word', 'Count'])
                    fig = px.pie(freq_df.head(8), values='Count', names='Word', 
                               title="Top Words Distribution")
                    st.plotly_chart(fig)
        
        # Export options
        st.markdown("---")
        st.subheader("📤 Export Results")
        
        # Prepare export data
        export_data = {
            'summary': themes['theme_summary'],
            'key_sentences': key_sentences,
            'theme_categories': themes['theme_categories'],
            'top_theme_words': themes['top_theme_words'],
            'statistics': {
                'total_words': len(content_input.split()),
                'unique_words': word_analysis['unique_words'],
                'sentences': len(re.split(r'[.!?]+', content_input))
            }
        }
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Copy Summary to Clipboard"):
                summary_text = f"""
THEMATIC ANALYSIS SUMMARY

{themes['theme_summary']}

KEY THEMES:
{chr(10).join([f"• {category}: {', '.join(words)}" for category, words in themes['theme_categories'].items() if words])}

KEY EXCERPTS:
{chr(10).join([f"{i+1}. {sentence}" for i, sentence in enumerate(key_sentences)])}
                """.strip()
                st.code(summary_text, language='text')
        
        with col2:
            st.download_button(
                label="💾 Download Full Analysis",
                data=str(export_data),
                file_name="thematic_analysis.txt",
                mime="text/plain"
            )

elif analysis_mode == "Custom Prompt Analysis":
    st.markdown('<h2 class="section-header">🤖 Custom Prompt Analysis</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content-card">
        <p>Upload any document and ask specific questions or request custom analysis. 
        Perfect for targeted insights, Q&A, and specific analysis requests.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input options
    input_option = st.radio(
        "Choose input method:",
        ["📝 Paste Text", "📁 Upload File"],
        horizontal=True,
        key="custom_prompt_input"
    )
    
    content_input = ""
    
    if input_option == "📝 Paste Text":
        content_input = st.text_area(
            "Enter your content for custom analysis:",
            height=200,
            placeholder="Paste your text, document content, or any text you want to analyze with custom prompts..."
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a file for custom prompt analysis",
            type=FileProcessor.get_supported_formats(),
            help=f"Supported formats: {', '.join(FileProcessor.get_supported_formats())}",
            key="custom_prompt_upload"
        )
        
        if uploaded_file is not None:
            content_input = FileProcessor.extract_text_from_file(uploaded_file)
            if content_input:
                st.success(f"✅ Processed {uploaded_file.name}")
    
    # Show document preview if available
    if content_input:
        with st.expander("📄 Document Preview", expanded=False):
            preview_text = content_input[:1000] + "..." if len(content_input) > 1000 else content_input
            st.text_area("Content preview:", preview_text, height=150, key="preview")
    
    # Prompt examples section
    st.markdown("### 💡 Example Prompts")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **📊 Analysis & Insights:**
        - "Summarize the main points of this document"
        - "What are the key findings and conclusions?"
        - "Analyze the strengths and weaknesses mentioned"
        - "What recommendations are provided?"
        """)
    
    with col2:
        st.markdown("""
        **🎯 Specific Questions:**
        - "What does this say about market trends?"
        - "What are the main risks identified?"
        - "What action items or next steps are mentioned?"
        - "How does this relate to [specific topic]?"
        """)
    
    # Custom prompt input
    user_prompt = st.text_area(
        "🤖 Enter your analysis prompt:",
        placeholder="e.g., 'Summarize the key findings and recommendations from this report'",
        height=100
    )
    
    # Analysis button and processing
    if content_input and user_prompt and st.button("🔍 Analyze with Custom Prompt", type="primary"):
        
        with st.spinner("🤖 Analyzing content with your custom prompt..."):
            # Custom analysis logic
            if OPENAI_AVAILABLE and openai_api_key:
                # AI-powered analysis
                try:
                    # Limit text length for API
                    text_limited = content_input[:4000] if len(content_input) > 4000 else content_input
                    
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful document analyst. Provide detailed, structured analysis based on the user's prompt."},
                            {"role": "user", "content": f"Document content:\n{text_limited}\n\nAnalysis request: {user_prompt}\n\nPlease provide a detailed analysis."}
                        ],
                        max_tokens=1000,
                        temperature=0.3
                    )
                    
                    ai_analysis = response.choices[0].message.content
                    analysis_method = "AI-powered (OpenAI)"
                    
                except Exception as e:
                    st.error(f"AI analysis failed: {str(e)}")
                    ai_analysis = None
                    analysis_method = "Basic analysis (AI failed)"
            else:
                ai_analysis = None
                analysis_method = "Basic analysis"
            
            # If AI analysis failed or not available, use basic analysis
            if not ai_analysis:
                words = content_input.lower().split()
                sentences = re.split(r'[.!?]+', content_input)
                
                # Basic keyword extraction based on prompt
                prompt_keywords = re.findall(r'\b\w{3,}\b', user_prompt.lower())
                found_keywords = []
                
                for keyword in prompt_keywords:
                    if keyword in content_input.lower():
                        count = content_input.lower().count(keyword)
                        found_keywords.append(f"{keyword} ({count} times)")
                
                # Basic sentiment
                positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'wonderful', 'amazing', 'effective', 'strong', 'beneficial']
                negative_words = ['bad', 'terrible', 'awful', 'negative', 'failure', 'horrible', 'disappointing', 'weak', 'poor', 'problematic']
                
                pos_count = sum(1 for word in words if word in positive_words)
                neg_count = sum(1 for word in words if word in negative_words)
                
                sentiment = "Neutral"
                if pos_count > neg_count:
                    sentiment = "Positive"
                elif neg_count > pos_count:
                    sentiment = "Negative"
                
                # Generate basic analysis
                ai_analysis = f"""
**Analysis for: "{user_prompt}"**

**Document Overview:**
- Word Count: {len(words)}
- Sentence Count: {len([s for s in sentences if s.strip()])}
- Character Count: {len(content_input)}

**Content Analysis:**
- Overall Sentiment: {sentiment}
- Positive indicators: {pos_count}
- Negative indicators: {neg_count}

**Keyword Relevance:**
{chr(10).join(['• ' + kw for kw in found_keywords]) if found_keywords else '• No specific keywords from your prompt found in the document.'}

**Summary:**
This document appears to be {sentiment.lower()} in tone. Based on your prompt "{user_prompt}", I've identified the above patterns. For more detailed analysis, consider providing an OpenAI API key for AI-powered insights.
                """
        
        # Display results in organized tabs
        tab1, tab2, tab3 = st.tabs(["🤖 Analysis Results", "📊 Document Stats", "💾 Export"])
        
        with tab1:
            st.subheader(f"Analysis for: \"{user_prompt}\"")
            st.markdown(ai_analysis)
            
            # Analysis metadata
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Analysis Method", analysis_method)
            with col2:
                st.metric("Content Length", f"{len(content_input.split())} words")
        
        with tab2:
            st.subheader("📈 Document Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            words = len(content_input.split())
            sentences = len(re.split(r'[.!?]+', content_input))
            paragraphs = len([p for p in content_input.split('\n\n') if p.strip()])
            
            with col1:
                st.metric("Words", words)
            with col2:
                st.metric("Sentences", sentences)
            with col3:
                st.metric("Paragraphs", paragraphs)
            with col4:
                avg_words_per_sentence = round(words/sentences if sentences > 0 else 0, 1)
                st.metric("Avg Words/Sentence", avg_words_per_sentence)
            
            # Word frequency analysis
            word_freq = Counter(content_input.lower().split())
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            filtered_freq = {word: freq for word, freq in word_freq.items() if word not in stop_words and len(word) > 2}
            top_words = dict(Counter(filtered_freq).most_common(10))
            
            if top_words:
                st.subheader("🔤 Most Frequent Words")
                words_df = pd.DataFrame(list(top_words.items()), columns=['Word', 'Frequency'])
                fig = px.bar(words_df, x='Word', y='Frequency', title="Top Words in Document")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig)
        
        with tab3:
            st.subheader("📤 Export Analysis")
            
            export_text = f"""
CUSTOM PROMPT ANALYSIS
======================

Prompt: {user_prompt}
Analysis Method: {analysis_method}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS RESULTS:
{ai_analysis}

DOCUMENT STATISTICS:
- Word Count: {len(content_input.split())}
- Character Count: {len(content_input)}
- Sentences: {len(re.split(r'[.!?]+', content_input))}
            """
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="💾 Download Analysis",
                    data=export_text,
                    file_name=f"custom_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                if st.button("📋 Copy to Clipboard"):
                    st.code(export_text, language='text')

elif analysis_mode == "Messaging Framework":
    st.markdown('<h2 class="section-header">🏗️ Messaging Framework Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card">
        <p>Analyze B2B messaging frameworks to identify key pillars, assess messaging maturity, and generate strategic recommendations for positioning and differentiation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input methods
    input_method = st.radio("How would you like to input your messaging content?", 
                           ["📝 Paste Text", "📁 Upload File"], horizontal=True)
    
    content_input = ""
    
    if input_method == "📝 Paste Text":
        content_input = st.text_area(
            "Paste your messaging content (website copy, marketing materials, etc.)",
            height=200,
            placeholder="Paste your company's messaging, value propositions, website copy, or marketing materials here..."
        )
    else:
        uploaded_file = st.file_uploader("Upload messaging document", 
                                       type=['txt', 'md', 'pdf', 'docx', 'doc'])
        if uploaded_file is not None:
            content_input = FileProcessor.extract_text_from_file(uploaded_file)
            if content_input:
                st.success(f"✅ Processed {uploaded_file.name}")
    
    # Framework Type Selection
    framework_type = st.selectbox(
        "Messaging Framework Focus",
        ["Complete Analysis", "B2B Positioning", "Value Proposition", "Competitive Differentiation"],
        help="Choose the focus area for your messaging analysis"
    )
    
    if content_input and st.button("🏗️ Analyze Messaging Framework", type="primary"):
        messaging_analyzer = MessagingFrameworkAnalyzer()
        
        with st.spinner("Analyzing messaging framework..."):
            # Core analysis
            framework = messaging_analyzer.analyze_messaging_framework(content_input)
            messaging_house = messaging_analyzer.generate_messaging_house(content_input)
            maturity_score = messaging_analyzer.score_messaging_maturity(framework)
        
        # Results Display
        st.subheader("📊 Messaging Maturity Assessment")
        
        # High-level metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Score", f"{maturity_score['overall_score']}%")
        with col2:
            st.metric("Maturity Level", maturity_score['maturity_level'])
        with col3:
            st.metric("Total Words", len(content_input.split()))
        with col4:
            strong_pillars = sum(1 for score in maturity_score['pillar_scores'].values() if score >= 3)
            st.metric("Strong Pillars", f"{strong_pillars}/6")
        
        # Framework Analysis Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🏠 Messaging House", "📊 Pillar Analysis", "💡 Recommendations", "📄 Export"])
        
        with tab1:
            st.markdown("### 🏠 Generated Messaging House")
            
            # Brand Promise
            st.markdown("#### 🎯 Brand Promise")
            st.info(messaging_house['brand_promise'])
            
            # Value Pillars
            st.markdown("#### 🏛️ Value Pillars")
            for i, pillar in enumerate(messaging_house['value_pillars'], 1):
                st.write(f"**{i}.** {pillar}")
            
            # Supporting Messages
            if messaging_house['supporting_messages']:
                st.markdown("#### 💬 Supporting Messages")
                for i, message in enumerate(messaging_house['supporting_messages'], 1):
                    st.write(f"**{i}.** {message}")
            
            # Proof Points
            if messaging_house['proof_points']:
                st.markdown("#### ✅ Proof Points")
                for i, proof in enumerate(messaging_house['proof_points'], 1):
                    st.write(f"**{i}.** {proof}")
            
            # Target Insights
            st.markdown("#### 🎯 Target Audience Insights")
            insights = messaging_house['target_insights']
            st.write(f"**Audience Clarity:** {insights['audience_clarity']}")
            st.write(f"**Pain Awareness:** {insights['pain_awareness']}")
            
            if insights['examples']:
                st.write("**Key Insights:**")
                for example in insights['examples'][:3]:
                    st.write(f"• {example}")
        
        with tab2:
            st.markdown("### 📊 Messaging Pillar Analysis")
            
            # Create pillar strength visualization
            pillar_names = []
            pillar_scores = []
            pillar_strengths = []
            
            for pillar, data in framework.items():
                pillar_names.append(pillar.replace('_', ' ').title())
                pillar_scores.append(data['count'])
                pillar_strengths.append(data['strength'])
            
            # Plotting
            fig = go.Figure()
            
            colors = {
                'Strong': '#2E8B57',
                'Moderate': '#FFA500', 
                'Weak': '#FF6347',
                'Missing': '#DC143C'
            }
            
            fig.add_trace(go.Bar(
                x=pillar_names,
                y=pillar_scores,
                text=pillar_strengths,
                textposition='auto',
                marker_color=[colors.get(strength, '#888888') for strength in pillar_strengths],
                name='Pillar Strength'
            ))
            
            fig.update_layout(
                title="Messaging Pillar Strength Analysis",
                xaxis_title="Messaging Pillars",
                yaxis_title="Mention Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown
            st.markdown("#### 📋 Detailed Pillar Breakdown")
            
            for pillar, data in framework.items():
                with st.expander(f"{pillar.replace('_', ' ').title()} - {data['strength']}"):
                    st.write(f"**Mentions:** {data['count']}")
                    st.write(f"**Strength:** {data['strength']}")
                    
                    if data['examples']:
                        st.write("**Examples found in your content:**")
                        for example in data['examples']:
                            st.write(f"• {example}")
                    else:
                        st.write("*No clear examples found - consider strengthening this pillar*")
        
        with tab3:
            st.markdown("### 💡 Strategic Recommendations")
            
            if maturity_score['recommendations']:
                st.markdown("#### 🎯 Priority Improvements")
                for i, rec in enumerate(maturity_score['recommendations'], 1):
                    st.warning(f"**{i}.** {rec}")
            else:
                st.success("🎉 Your messaging framework is well-developed across all key pillars!")
            
            # Additional insights based on maturity level
            st.markdown("#### 📈 Next Steps by Maturity Level")
            
            level = maturity_score['maturity_level']
            
            if level == "Foundational":
                st.markdown("""
                **Foundational Level Focus:**
                - Define clear value propositions
                - Identify target audience pain points
                - Establish core differentiators
                - Gather initial proof points
                """)
            elif level == "Basic":
                st.markdown("""
                **Basic Level Focus:**
                - Strengthen weak messaging pillars
                - Add more proof points and case studies
                - Refine target audience definitions
                - Enhance competitive differentiation
                """)
            elif level == "Developing":
                st.markdown("""
                **Developing Level Focus:**
                - Fine-tune messaging consistency
                - Add emotional resonance
                - Strengthen proof points
                - Test messaging effectiveness
                """)
            else:  # Advanced
                st.markdown("""
                **Advanced Level Focus:**
                - Optimize messaging for different channels
                - A/B test message variations
                - Develop industry-specific messaging
                - Train teams on messaging consistency
                """)
        
        with tab4:
            st.markdown("### 📄 Export Analysis")
            
            # Export data preparation
            export_content = []
            export_content.append("MESSAGING FRAMEWORK ANALYSIS REPORT")
            export_content.append("="*50)
            export_content.append(f"Overall Score: {maturity_score['overall_score']}%")
            export_content.append(f"Maturity Level: {maturity_score['maturity_level']}")
            export_content.append("")
            
            export_content.append("MESSAGING HOUSE:")
            export_content.append(f"Brand Promise: {messaging_house['brand_promise']}")
            export_content.append("")
            export_content.append("Value Pillars:")
            for i, pillar in enumerate(messaging_house['value_pillars'], 1):
                export_content.append(f"{i}. {pillar}")
            export_content.append("")
            
            export_content.append("RECOMMENDATIONS:")
            for i, rec in enumerate(maturity_score['recommendations'], 1):
                export_content.append(f"{i}. {rec}")
            
            st.text_area("Report Preview:", value="\n".join(export_content[:20]) + "\n...", height=300)
            
            if st.download_button(
                "📥 Download Messaging Framework Report",
                data="\n".join(export_content),
                file_name=f"messaging_framework_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            ):
                st.success("Messaging framework report downloaded!")

elif analysis_mode == "🥊 AI Competitive Analysis":
    st.markdown('<h2 class="section-header">🥊 AI Competitive Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card">
        <p>AI-powered competitive messaging analysis using semantic embeddings, brand archetypes, and strategic gap identification. Compare your messaging against competitors to find differentiation opportunities and positioning gaps.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Client Information
    st.markdown("### 🏢 Client Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        client_name = st.text_input("Client/Brand Name", placeholder="e.g., Your Company Name")
    
    with col2:
        analysis_depth = st.selectbox("Analysis Depth", ["Standard", "Deep AI", "Quick"], index=1)
    
    # Client Content Input
    st.markdown("#### 📄 Client Messaging Content")
    client_input_method = st.radio("Input Method", ["📝 Paste Text", "📁 Upload File"], horizontal=True, key="client_input")
    
    client_content = ""
    if client_input_method == "📝 Paste Text":
        client_content = st.text_area(
            "Client messaging content (website, marketing materials, etc.)",
            height=150,
            placeholder="Paste your client's messaging, website copy, value propositions, etc..."
        )
    else:
        client_file = st.file_uploader("Upload client document", type=['txt', 'md', 'pdf', 'docx', 'doc'], key="client_file")
        if client_file:
            client_content = FileProcessor.extract_text_from_file(client_file)
            if client_content:
                st.success(f"✅ Processed {client_file.name}")
    
    # Competitor Information
    st.markdown("### 🏆 Competitor Analysis")
    
    num_competitors = st.slider("Number of Competitors", 1, 5, 2)
    
    competitors = []
    for i in range(num_competitors):
        st.markdown(f"#### Competitor {i+1}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            comp_name = st.text_input(f"Competitor Name", key=f"comp_name_{i}", placeholder=f"Competitor {i+1}")
        
        with col2:
            comp_input_method = st.radio("Input Method", ["📝 Paste", "📁 Upload"], horizontal=True, key=f"comp_method_{i}")
        
        comp_content = ""
        if comp_input_method == "📝 Paste":
            comp_content = st.text_area(
                f"Competitor messaging content",
                height=100,
                key=f"comp_content_{i}",
                placeholder="Paste competitor's messaging..."
            )
        else:
            comp_file = st.file_uploader(f"Upload competitor document", type=['txt', 'md', 'pdf', 'docx', 'doc'], key=f"comp_file_{i}")
            if comp_file:
                comp_content = FileProcessor.extract_text_from_file(comp_file)
                if comp_content:
                    st.success(f"✅ Processed {comp_file.name}")
        
        if comp_name and comp_content:
            competitors.append({'name': comp_name, 'text': comp_content})
    
    # Analysis Execution
    if client_name and client_content and competitors and st.button("🚀 Run AI Competitive Analysis", type="primary"):
        
        ai_analyzer = AICompetitiveAnalyzer()
        
        with st.spinner("🤖 Running AI-powered competitive analysis..."):
            try:
                # Run comprehensive analysis
                analysis_results = ai_analyzer.comprehensive_competitive_analysis(
                    client_name, client_content, competitors, openai_api_key
                )
                
                # Display Results
                st.markdown("## 📊 Competitive Analysis Results")
                
                # Executive Summary
                st.markdown("### 🎯 Executive Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                client_analysis = analysis_results['client_analysis']
                
                with col1:
                    st.metric("Client Words", client_analysis['word_count'])
                with col2:
                    st.metric("Primary Archetype", client_analysis['archetype']['primary_archetype'].title())
                with col3:
                    avg_similarity = np.mean(analysis_results['comparative_insights']['semantic_similarities'])
                    st.metric("Avg Similarity", f"{avg_similarity:.1%}")
                with col4:
                    diff_score = analysis_results['comparative_insights']['unique_messaging']['differentiation_score']
                    st.metric("Differentiation", f"{diff_score:.1f}%")
                
                # Main Analysis Tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧠 AI Insights", "🎭 Brand Archetypes", "🔍 Competitive Gaps", "📊 Detailed Analysis", "📄 Export"])
                
                with tab1:
                    st.markdown("### 🤖 AI-Generated Strategic Insights")
                    
                    ai_insights = analysis_results['comparative_insights']['ai_insights']
                    
                    st.markdown(f"**Analysis Method:** {ai_insights['method']}")
                    st.markdown(f"**Confidence Level:** {ai_insights['confidence']}")
                    
                    st.markdown("#### Strategic Analysis")
                    st.write(ai_insights['insights'])
                    
                    st.markdown("#### 🎯 Strategic Recommendations")
                    for i, rec in enumerate(analysis_results['recommendations'], 1):
                        st.info(f"**{i}.** {rec}")
                
                with tab2:
                    st.markdown("### 🎭 Brand Archetype Analysis")
                    
                    # Create archetype comparison chart
                    archetype_data = []
                    
                    # Add client data
                    client_archetypes = client_analysis['archetype']['archetype_scores']
                    archetype_data.append({
                        'Brand': client_name,
                        'Type': 'Client',
                        **client_archetypes
                    })
                    
                    # Add competitor data
                    for comp in analysis_results['competitor_analyses']:
                        comp_archetypes = comp['archetype']['archetype_scores']
                        archetype_data.append({
                            'Brand': comp['name'],
                            'Type': 'Competitor',
                            **comp_archetypes
                        })
                    
                    archetype_df = pd.DataFrame(archetype_data)
                    
                    # Create radar chart for archetypes
                    archetype_columns = list(ai_analyzer.brand_archetypes.keys())
                    
                    fig = go.Figure()
                    
                    # Add client trace
                    client_scores = [client_archetypes.get(arch, 0) for arch in archetype_columns]
                    fig.add_trace(go.Scatterpolar(
                        r=client_scores,
                        theta=[arch.replace('_', ' ').title() for arch in archetype_columns],
                        fill='toself',
                        name=client_name,
                        line_color='#ff6b6b'
                    ))
                    
                    # Add competitor traces
                    colors = ['#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd']
                    for i, comp in enumerate(analysis_results['competitor_analyses']):
                        comp_scores = [comp['archetype']['archetype_scores'].get(arch, 0) for arch in archetype_columns]
                        fig.add_trace(go.Scatterpolar(
                            r=comp_scores,
                            theta=[arch.replace('_', ' ').title() for arch in archetype_columns],
                            fill='toself',
                            name=comp['name'],
                            line_color=colors[i % len(colors)]
                        ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, max(max(client_scores), 5)])),
                        title="Brand Archetype Comparison",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Archetype insights
                    st.markdown("#### 🔍 Archetype Insights")
                    
                    client_primary = client_analysis['archetype']['primary_archetype']
                    competitor_primaries = [comp['archetype']['primary_archetype'] for comp in analysis_results['competitor_analyses']]
                    
                    if client_primary in competitor_primaries:
                        st.warning(f"⚠️ **Archetype Overlap**: {client_name} shares the '{client_primary}' archetype with competitors. Consider differentiating positioning.")
                    else:
                        st.success(f"✅ **Unique Positioning**: {client_name}'s '{client_primary}' archetype is unique in this competitive set.")
                
                with tab3:
                    st.markdown("### 🔍 Competitive Gap Analysis")
                    
                    unique_msg = analysis_results['comparative_insights']['unique_messaging']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🎯 Unique to Client")
                        if unique_msg['unique_to_client']:
                            for word in unique_msg['unique_to_client'][:8]:
                                st.markdown(f"- **{word}**")
                        else:
                            st.info("No unique terms identified - consider developing distinctive messaging")
                    
                    with col2:
                        st.markdown("#### 🏆 Competitor Advantages")
                        if unique_msg['unique_to_competitors']:
                            for word in unique_msg['unique_to_competitors'][:8]:
                                st.markdown(f"- {word}")
                        else:
                            st.success("No major competitor messaging advantages identified")
                    
                    st.markdown("#### 📈 Differentiation Opportunities")
                    
                    # Messaging themes comparison
                    theme_comparison = []
                    
                    client_themes = client_analysis['themes']['theme_scores']
                    theme_comparison.append({
                        'Brand': client_name,
                        'Type': 'Client',
                        **client_themes
                    })
                    
                    for comp in analysis_results['competitor_analyses']:
                        comp_themes = comp['themes']['theme_scores']
                        theme_comparison.append({
                            'Brand': comp['name'],
                            'Type': 'Competitor',
                            **comp_themes
                        })
                    
                    theme_df = pd.DataFrame(theme_comparison)
                    
                    # Create themes heatmap
                    theme_columns = list(ai_analyzer.messaging_themes.keys())
                    theme_matrix = theme_df[theme_columns].values
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=theme_matrix,
                        x=[theme.title() for theme in theme_columns],
                        y=theme_df['Brand'].tolist(),
                        colorscale='RdYlBu_r',
                        showscale=True
                    ))
                    
                    fig.update_layout(
                        title="Messaging Themes Heatmap",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    st.markdown("### 📊 Detailed Competitive Analysis")
                    
                    # Similarity matrix
                    st.markdown("#### 🔗 Semantic Similarity Matrix")
                    
                    similarity_data = []
                    all_brands = [{'name': client_name, 'text': client_content}] + competitors
                    
                    for i, brand1 in enumerate(all_brands):
                        for j, brand2 in enumerate(all_brands):
                            if i != j:
                                similarity = ai_analyzer.calculate_semantic_similarity(brand1['text'], brand2['text'])
                                similarity_data.append({
                                    'Brand 1': brand1['name'],
                                    'Brand 2': brand2['name'],
                                    'Similarity': similarity
                                })
                    
                    if similarity_data:
                        similarity_df = pd.DataFrame(similarity_data)
                        similarity_pivot = similarity_df.pivot(index='Brand 1', columns='Brand 2', values='Similarity')
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=similarity_pivot.values,
                            x=similarity_pivot.columns,
                            y=similarity_pivot.index,
                            colorscale='RdYlBu',
                            zmin=0,
                            zmax=1,
                            showscale=True
                        ))
                        
                        fig.update_layout(
                            title="Brand Messaging Similarity Matrix",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed competitor breakdown
                    st.markdown("#### 📋 Competitor Breakdown")
                    
                    for comp in analysis_results['competitor_analyses']:
                        with st.expander(f"📊 {comp['name']} Analysis"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Word Count", comp['word_count'])
                            with col2:
                                st.metric("Primary Archetype", comp['archetype']['primary_archetype'].title())
                            with col3:
                                st.metric("Similarity to Client", f"{comp['similarity_to_client']:.1%}")
                            
                            st.markdown("**Dominant Themes:**")
                            for theme in comp['themes']['dominant_themes']:
                                st.write(f"- {theme.title()}")
                
                with tab5:
                    st.markdown("### 📄 Export Competitive Analysis")
                    
                    # Prepare export data
                    export_data = []
                    export_data.append("AI COMPETITIVE ANALYSIS REPORT")
                    export_data.append("=" * 50)
                    export_data.append(f"Client: {client_name}")
                    export_data.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                    export_data.append("")
                    
                    export_data.append("EXECUTIVE SUMMARY:")
                    export_data.append(f"Primary Archetype: {client_analysis['archetype']['primary_archetype'].title()}")
                    export_data.append(f"Differentiation Score: {diff_score:.1f}%")
                    export_data.append(f"Average Similarity: {avg_similarity:.1%}")
                    export_data.append("")
                    
                    export_data.append("AI INSIGHTS:")
                    export_data.append(ai_insights['insights'])
                    export_data.append("")
                    
                    export_data.append("STRATEGIC RECOMMENDATIONS:")
                    for i, rec in enumerate(analysis_results['recommendations'], 1):
                        export_data.append(f"{i}. {rec}")
                    export_data.append("")
                    
                    export_data.append("COMPETITIVE ANALYSIS:")
                    for comp in analysis_results['competitor_analyses']:
                        export_data.append(f"- {comp['name']}: {comp['archetype']['primary_archetype'].title()} archetype, {comp['similarity_to_client']:.1%} similarity")
                    
                    st.text_area("Report Preview:", value="\n".join(export_data[:25]) + "\n...", height=400)
                    
                    if st.download_button(
                        "📥 Download Competitive Analysis Report",
                        data="\n".join(export_data),
                        file_name=f"competitive_analysis_{client_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    ):
                        st.success("Competitive analysis report downloaded!")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.info("Try using fewer competitors or shorter text inputs.")

elif analysis_mode == "Document Comparison":
    st.markdown('<h2 class="section-header">⚖️ Document Comparison</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content-card">
        <p>Compare multiple documents side-by-side with comprehensive scoring and thematic analysis. 
        Identify patterns, common themes, and performance differences across your content portfolio.</p>
    </div>
    """, unsafe_allow_html=True)
    
    num_docs = st.number_input("Number of documents to compare", min_value=2, max_value=5, value=2)
    
    docs = []
    doc_names = []
    
    for i in range(num_docs):
        st.subheader(f"📄 Document {i+1}")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            doc_name = st.text_input(f"Document {i+1} Name", value=f"Document {i+1}", key=f"doc_name_{i}")
            doc_names.append(doc_name)
        
        with col2:
            input_method = st.radio(f"Input method for Document {i+1}", ["Type/Paste Text", "Upload File"], horizontal=True, key=f"doc_input_{i}")
        
        if input_method == "Type/Paste Text":
            doc_content = st.text_area(f"Document {i+1} Content", height=100, key=f"doc_text_{i}")
        else:
            supported_formats = FileProcessor.get_supported_formats()
            uploaded_file = st.file_uploader(f"Upload Document {i+1}", type=supported_formats, key=f"doc_file_{i}")
            
            if uploaded_file is not None:
                doc_content = FileProcessor.extract_text_from_file(uploaded_file)
                if doc_content:
                    st.success(f"✅ Processed {uploaded_file.name}")
                    doc_names[i] = uploaded_file.name.split('.')[0]
            else:
                doc_content = ""
        
        docs.append(doc_content)
    
    if all(d.strip() for d in docs) and st.button("⚖️ Compare Documents", type="primary"):
        scorer = ContentScorer(openai_api_key)
        summarizer = EnhancedSummarizer()
        
        with st.spinner("Analyzing all documents..."):
            all_results = []
            progress_bar = st.progress(0)
            total_steps = len(docs) * 2  # scoring + theme analysis
            current_step = 0
            
            # Score all documents
            for i, doc in enumerate(docs):
                st.write(f"📊 Scoring document {i+1}: {doc_names[i]}")
                result = scorer.comprehensive_score(doc, target_keywords)
                result['name'] = doc_names[i]
                result['word_count'] = len(doc.split())
                all_results.append(result)
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            # Theme analysis across documents
            st.write("🎯 Analyzing themes across all documents...")
            theme_comparison = summarizer.compare_themes_across_texts(docs, doc_names)
            
            current_step += len(docs)
            progress_bar.progress(1.0)
            progress_bar.empty()
        
        # Create tabs for organized results
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Scores", "🎯 Themes", "☁️ Word Maps", "📈 Visualizations"])
        
        with tab1:
            st.subheader("📊 Document Scores Comparison")
            
            df_scores = pd.DataFrame([
                {
                    'Document': r['name'],
                    'Overall Score': r['overall_score'],
                    'Readability': r['readability']['readability_score'],
                    'Sentiment': r['sentiment']['sentiment_score'],
                    'Engagement': r['engagement']['engagement_score'],
                    'SEO': r['seo']['seo_score'],
                    'Word Count': r['word_count']
                }
                for r in all_results
            ])
            
            st.dataframe(df_scores, use_container_width=True)
            
            # Best and worst performers
            best_doc = df_scores.loc[df_scores['Overall Score'].idxmax()]
            worst_doc = df_scores.loc[df_scores['Overall Score'].idxmin()]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 **Best Performer:** {best_doc['Document']} ({best_doc['Overall Score']:.1f})")
            with col2:
                st.error(f"⚠️ **Needs Improvement:** {worst_doc['Document']} ({worst_doc['Overall Score']:.1f})")
        
        with tab2:
            st.subheader("🎯 Thematic Analysis Across Documents")
            
            # Common themes summary
            st.write(f"**Cross-Document Analysis:** {theme_comparison['comparison_summary']}")
            
            if theme_comparison['common_themes']:
                st.subheader("🔗 Common Themes")
                common_df = pd.DataFrame(list(theme_comparison['common_themes'].items()), columns=['Theme', 'Frequency'])
                fig = px.bar(common_df, x='Theme', y='Frequency', title="Common Themes Across Documents")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig)
            
            # Individual document themes
            st.subheader("📋 Individual Document Themes")
            for theme_data in theme_comparison['individual_themes']:
                with st.expander(f"📄 {theme_data['text_name']} - Themes"):
                    st.write(f"**Summary:** {theme_data['theme_summary']}")
                    
                    if theme_data['theme_categories']:
                        for category, words in theme_data['theme_categories'].items():
                            if words:
                                st.write(f"**{category}:** {', '.join(words)}")
        
        with tab3:
            st.subheader("☁️ Document Word Clouds")
            
            # Create word clouds for each document
            cols = st.columns(min(len(docs), 3))  # Max 3 columns
            
            for i, (doc, name) in enumerate(zip(docs, doc_names)):
                col_idx = i % 3
                with cols[col_idx]:
                    st.markdown(f"**{name}**")
                    try:
                        clean_text = summarizer.create_theme_wordcloud_data(doc)
                        if clean_text.strip():
                            wc = WordCloud(width=300, height=200, background_color='white').generate(clean_text)
                            fig, ax = plt.subplots(figsize=(4, 3))
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis("off")
                            st.pyplot(fig)
                        else:
                            st.info("Not enough content")
                    except:
                        st.info("Word cloud not available")
            
            # Common themes word cloud
            if theme_comparison['common_themes']:
                st.subheader("🔗 Common Themes Word Cloud")
                try:
                    common_text = ' '.join([word + ' ' * freq for word, freq in theme_comparison['common_themes'].items()])
                    wc_common = WordCloud(width=600, height=300, background_color='white').generate(common_text)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.imshow(wc_common, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except:
                    st.info("Common themes word cloud not available")
        
        with tab4:
            st.subheader("📈 Comparative Visualizations")
            
            # Score comparison chart
            fig = px.bar(
                df_scores.melt(id_vars=['Document'], value_vars=['Overall Score', 'Readability', 'Sentiment', 'Engagement', 'SEO'], 
                              var_name='Metric', value_name='Score'),
                x='Document',
                y='Score',
                color='Metric',
                title="Document Scores Comparison",
                barmode='group'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Radar chart comparison
            if len(docs) <= 3:  # Only show radar for <= 3 docs to avoid clutter
                st.subheader("📡 Multi-Document Radar Comparison")
                
                fig = go.Figure()
                categories = ['Readability', 'Sentiment', 'Engagement', 'SEO']
                
                for result in all_results:
                    values = [
                        result['readability']['readability_score'],
                        result['sentiment']['sentiment_score'],
                        result['engagement']['engagement_score'],
                        result['seo']['seo_score']
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=result['name']
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="Multi-Document Score Comparison"
                )
                st.plotly_chart(fig)


