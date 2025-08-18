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
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
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
    
    .main-header .features {
        font-size: 0.9rem;
        opacity: 0.8;
        font-style: italic;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    .css-1lcbmhc {
        background-color: #f8f9fa;
    }
    
    /* Button styling inspired by S&S */
    .stButton > button {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(44, 62, 80, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(44, 62, 80, 0.3);
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
    }
    
    /* Metric cards styling */
    .css-1xarl3l {
        background-color: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px 10px 0 0;
        border: 1px solid #e9ecef;
        border-bottom: none;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #6c757d;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50;
        color: white;
        border-color: #2c3e50;
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
    
    /* Cards for content sections */
    .content-card {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 15px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
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
**🔬 Stable Content Analysis Tool**
- **Content Scoring**: Analyze readability, sentiment, engagement, SEO, and quality
- **Basic Summarization**: Simple text analysis without heavy models
- **File Support**: Upload PDF, Word, Text, and Markdown files
- **Stable Operation**: Optimized for reliability without model crashes
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

# --- Header with S&S inspired styling ---
st.markdown("""
<div class="main-header">
    <h1>🔬 Content Analysis Tool</h1>
    <p>AI-powered messaging analysis and competitive comparison for marketing professionals</p>
    <div class="features">Messaging comparison • Brand analysis • Competitive insights • Content scoring • Thematic analysis</div>
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
class StableContentScorer:
    """Stable content scoring without heavy models"""
    
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
    
    def sentiment_score(self, text):
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
            return {'sentiment': 'NEUTRAL', 'confidence': 0.5, 'sentiment_score': 50, 'positive_words': 0, 'negative_words': 0}
        
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
            'negative_words': negative_count
        }
    
    def engagement_score(self, text):
        """Calculate engagement potential"""
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
            'personal_pronouns': pronoun_count
        }
    
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
    
    def comprehensive_score(self, text, target_keywords=None):
        """Generate comprehensive content score"""
        readability = self.readability_score(text)
        sentiment = self.sentiment_score(text)
        engagement = self.engagement_score(text)
        seo = self.seo_score(text, target_keywords)
        
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
            'timestamp': datetime.now().isoformat()
        }

class BasicSummarizer:
    """Basic text summarization without heavy models"""
    
    def __init__(self):
        pass
    
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
        
        # Categorize themes
        themes = {
            'Business & Strategy': [word for word in theme_words if word in business_words],
            'Technology & Innovation': [word for word in theme_words if word in tech_words],
            'Process & Operations': [word for word in theme_words if word in process_words],
            'People & Culture': [word for word in theme_words if word in people_words],
            'General Themes': [word for word in theme_words if word not in business_words + tech_words + process_words + people_words]
        }
        
        # Remove empty categories
        themes = {category: words for category, words in themes.items() if words}
        
        return {
            'theme_categories': themes,
            'top_theme_words': theme_words,
            'theme_summary': self._generate_theme_summary(themes)
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
    ["Content Scoring", "Basic Text Analysis", "Thematic Summary", "Custom Prompt Analysis", "Document Comparison"]
)

target_keywords = st.sidebar.text_input(
    "Target Keywords (comma-separated)",
    placeholder="keyword1, keyword2, keyword3"
)

if target_keywords:
    target_keywords = [k.strip() for k in target_keywords.split(',') if k.strip()]
else:
    target_keywords = None

# --- Main Content Area ---
if analysis_mode == "Content Scoring":
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
    
    if content_input and st.button("🔍 Analyze Content", type="primary"):
        scorer = StableContentScorer(openai_api_key)
        
        with st.spinner("Analyzing content..."):
            results = scorer.comprehensive_score(content_input, target_keywords)
        
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
        st.subheader("📊 Detailed Scoring Breakdown")
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
        summarizer = BasicSummarizer()
        
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
    col1, col2 = st.columns(2)
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
    
    if content_input and st.button("🎯 Generate Thematic Summary", type="primary"):
        summarizer = BasicSummarizer()
        
        # Parse exclude terms
        exclude_set = set()
        if exclude_terms:
            exclude_set = set([term.strip().lower() for term in exclude_terms.split(',') if term.strip()])
        
        with st.spinner("Analyzing themes and generating summary..."):
            themes = summarizer.extract_themes(content_input)
            key_sentences = summarizer.extract_key_sentences(
                content_input, 
                3 if summary_length == "Brief" else 5 if summary_length == "Standard" else 8
            )
            word_analysis = summarizer.get_word_frequency_summary(content_input)
        
        # Main Results Display
        st.subheader("📋 Executive Summary")
        
        # Key insights box
        with st.container():
            st.markdown("### 🔍 Key Insights")
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.metric("Total Words", len(content_input.split()))
                st.metric("Main Themes Identified", len(themes['top_theme_words']))
            
            with insight_col2:
                sentences_count = len(re.split(r'[.!?]+', content_input))
                st.metric("Sentences", sentences_count)
                st.metric("Theme Categories", len(themes['theme_categories']))
        
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
        scorer = StableContentScorer(openai_api_key)
        summarizer = BasicSummarizer()
        
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

# --- Footer inspired by S&S ---
st.markdown("""
<div style='margin-top: 4rem; padding: 3rem 0 2rem 0; background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); 
            border-radius: 20px 20px 0 0; color: white; text-align: center;'>
    <div style='font-size: 1.1rem; font-weight: 300; margin-bottom: 1rem;'>
        🔬 <strong style='font-weight: 500;'>Content Analysis Tool</strong>
    </div>
    <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 1.5rem; line-height: 1.6;'>
        With keen attention to detail and a knack for establishing insights that matter,<br>
        we deliver 'aha' analysis moments for marketing messaging and competitive comparison.
    </div>
    <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1.5rem;'>
        <div style='opacity: 0.8;'>📊 Messaging Analysis</div>
        <div style='opacity: 0.8;'>🎯 Brand Insights</div>
        <div style='opacity: 0.8;'>⚖️ Competitive Comparison</div>
        <div style='opacity: 0.8;'>🤖 Custom Analysis</div>
    </div>
    <div style='font-size: 0.8rem; opacity: 0.7; font-style: italic;'>
        AI-powered analysis for marketing professionals • Messaging frameworks • Competitive intelligence
    </div>
</div>
""", unsafe_allow_html=True)
