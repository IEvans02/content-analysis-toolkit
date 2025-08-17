import streamlit as st
import openai
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
    pass  # seaborn is optional for styling

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
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("⚠️ Transformers not available. Summarization features will be limited.")

# --- Configuration ---
st.set_page_config(
    page_title="Unified Content Analysis Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Instructions ---
st.markdown("""
**🔬 Unified Content Analysis Tool**
- **Content Scoring**: Analyze readability, sentiment, engagement, SEO, and quality
- **Text Summarization**: Generate summaries and thematic analysis
- **File Support**: Upload PDF, Word, Text, and Markdown files
- **Comparative Analysis**: Compare multiple documents side-by-side
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

if openai_api_key:
    openai.api_key = openai_api_key

st.title("🔬 Unified Content Analysis Tool")

# --- Load Models (Lazy Loading) ---
@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    except Exception as e:
        st.warning(f"⚠️ Sentiment model failed to load: {e}")
        return None

@st.cache_resource  
def load_summarizer_model():
    """Load summarization model"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    except Exception as e:
        st.warning(f"⚠️ Summarization model failed to load: {e}")
        return None

def get_sentiment_model():
    """Get sentiment model with lazy loading"""
    if 'sentiment_model' not in st.session_state:
        with st.spinner("Loading sentiment analysis model..."):
            st.session_state.sentiment_model = load_sentiment_model()
    return st.session_state.sentiment_model

def get_summarizer_model():
    """Get summarizer model with lazy loading"""
    if 'summarizer_model' not in st.session_state:
        with st.spinner("Loading summarization model..."):
            st.session_state.summarizer_model = load_summarizer_model()
    return st.session_state.summarizer_model

# --- File Processing ---
class FileProcessor:
    """Handle different file types and extract text content"""
    
    @staticmethod
    def extract_text_from_file(uploaded_file):
        """Extract text from uploaded file based on file type"""
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
            st.error("Word document support not available. Please install python-docx")
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
        # Remove markdown formatting for analysis
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
    """Content scoring functionality"""
    
    def __init__(self, sentiment_analyzer=None, openai_api_key=None):
        self.sentiment_analyzer = sentiment_analyzer
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
            return {'flesch_ease': 0, 'fk_grade': 0, 'ari': 0, 'readability_score': 0, 'readability_level': 'Unknown'}
    
    def _get_readability_level(self, flesch_score):
        if flesch_score >= 90: return "Very Easy"
        elif flesch_score >= 80: return "Easy"
        elif flesch_score >= 70: return "Fairly Easy"
        elif flesch_score >= 60: return "Standard"
        elif flesch_score >= 50: return "Fairly Difficult"
        elif flesch_score >= 30: return "Difficult"
        else: return "Very Difficult"
    
    def sentiment_score(self, text):
        """Analyze sentiment"""
        if self.sentiment_analyzer is not None:
            try:
                max_length = 512
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                
                sentiments = []
                for chunk in chunks[:5]:
                    if chunk.strip():
                        result = self.sentiment_analyzer(chunk)
                        sentiments.append(result[0])
                
                if not sentiments:
                    return {'sentiment': 'NEUTRAL', 'confidence': 0.5, 'sentiment_score': 50}
                
                positive_scores = [s['score'] for s in sentiments if s['label'] in ['POSITIVE', 'LABEL_1']]
                negative_scores = [s['score'] for s in sentiments if s['label'] in ['NEGATIVE', 'LABEL_0']]
                
                avg_positive = np.mean(positive_scores) if positive_scores else 0
                avg_negative = np.mean(negative_scores) if negative_scores else 0
                
                if avg_positive > avg_negative:
                    sentiment = 'POSITIVE'
                    confidence = avg_positive
                    sentiment_score = 50 + (avg_positive * 50)
                elif avg_negative > avg_positive:
                    sentiment = 'NEGATIVE'
                    confidence = avg_negative
                    sentiment_score = 50 - (avg_negative * 50)
                else:
                    sentiment = 'NEUTRAL'
                    confidence = max(avg_positive, avg_negative)
                    sentiment_score = 50
                
                return {'sentiment': sentiment, 'confidence': confidence, 'sentiment_score': sentiment_score}
            except:
                return self._basic_sentiment_analysis(text)
        else:
            return self._basic_sentiment_analysis(text)
    
    def _basic_sentiment_analysis(self, text):
        """Basic sentiment analysis using word lists"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'best', 'awesome', 'perfect', 'brilliant', 'outstanding']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike', 'poor', 'disappointing', 'frustrating', 'annoying', 'sad', 'angry']
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {'sentiment': 'NEUTRAL', 'confidence': 0.5, 'sentiment_score': 50}
        
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
        
        return {'sentiment': sentiment, 'confidence': confidence, 'sentiment_score': sentiment_score}
    
    def engagement_score(self, text):
        """Calculate engagement potential"""
        words = len(text.split())
        word_score = min(100, (words / 300) * 100) if words <= 300 else max(0, 100 - ((words - 300) / 50) * 10)
        
        questions = text.count('?')
        question_score = min(100, questions * 20)
        
        exclamations = text.count('!')
        excitement_score = min(100, exclamations * 15)
        
        cta_words = ['click', 'buy', 'subscribe', 'join', 'download', 'learn', 'discover', 'try', 'get', 'start']
        cta_count = sum([text.lower().count(word) for word in cta_words])
        cta_score = min(100, cta_count * 25)
        
        power_words = ['amazing', 'incredible', 'exclusive', 'proven', 'guaranteed', 'instant', 'ultimate', 'secret']
        power_count = sum([text.lower().count(word) for word in power_words])
        power_score = min(100, power_count * 20)
        
        engagement_score = (word_score * 0.3 + question_score * 0.2 + excitement_score * 0.15 + cta_score * 0.2 + power_score * 0.15)
        
        return {
            'engagement_score': engagement_score,
            'word_count': words,
            'questions': questions,
            'exclamations': exclamations,
            'cta_count': cta_count,
            'power_words': power_count
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
        
        length_score = min(100, (len(words) / 300) * 100) if len(words) <= 300 else max(50, 100 - ((len(words) - 300) / 100) * 5)
        headings = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
        heading_score = min(100, headings * 25)
        
        seo_score = (keyword_score * 0.4 + length_score * 0.4 + heading_score * 0.2)
        
        return {
            'seo_score': seo_score,
            'keyword_density': keyword_density,
            'word_count': len(words),
            'headings_count': headings
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

class ContentSummarizer:
    """Content summarization functionality"""
    
    def __init__(self, summarizer=None):
        self.summarizer = summarizer
    
    def summarize_text(self, text, max_length=130, min_length=30):
        """Generate summary of text"""
        if self.summarizer is None:
            return {"summary_text": "Summarization not available - model not loaded"}
        
        try:
            # Handle long texts by chunking
            if len(text.split()) > 1000:
                # Split into chunks and summarize each
                chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
                summaries = []
                for chunk in chunks[:3]:  # Limit to first 3 chunks
                    if chunk.strip():
                        summary = self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                        summaries.append(summary[0]['summary_text'])
                return {"summary_text": " ".join(summaries)}
            else:
                summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
                return summary[0]
        except Exception as e:
            return {"summary_text": f"Summarization failed: {str(e)}"}
    
    def extract_themes(self, text):
        """Extract main themes from text"""
        if self.summarizer is None:
            return {"summary_text": "Theme extraction not available"}
        
        theme_prompt = f"List the main themes or topics in the following text.\n\n{text[:1000]}"
        try:
            theme_summary = self.summarizer(theme_prompt, max_length=60, min_length=20, do_sample=False)
            return theme_summary[0]
        except Exception as e:
            return {"summary_text": f"Theme extraction failed: {str(e)}"}
    
    def compare_texts(self, texts):
        """Compare multiple texts and find commonalities"""
        if self.summarizer is None:
            return {"summary_text": "Text comparison not available"}
        
        # Generate individual summaries
        summaries = []
        for i, text in enumerate(texts):
            summary = self.summarize_text(text, max_length=100, min_length=20)
            summaries.append(f"Text {i+1}: {summary['summary_text']}")
        
        # Generate comparison summary
        comparison_input = "\n".join(summaries)
        try:
            comparison = self.summarizer(f"Compare and contrast these summaries, highlighting commonalities and differences:\n{comparison_input}", 
                                       max_length=200, min_length=50, do_sample=False)
            return comparison[0]
        except Exception as e:
            return {"summary_text": f"Comparison failed: {str(e)}"}

# --- Sidebar Configuration ---
st.sidebar.header("🔧 Analysis Configuration")

# Analysis mode selection
analysis_mode = st.sidebar.selectbox(
    "Select Analysis Mode",
    ["Content Scoring", "Text Summarization", "Combined Analysis", "Document Comparison"]
)

# Target keywords for SEO scoring
target_keywords = st.sidebar.text_input(
    "Target Keywords (comma-separated)",
    placeholder="keyword1, keyword2, keyword3"
)

if target_keywords:
    target_keywords = [k.strip() for k in target_keywords.split(',') if k.strip()]
else:
    target_keywords = None

# Summarization settings
if analysis_mode in ["Text Summarization", "Combined Analysis", "Document Comparison"]:
    st.sidebar.subheader("📝 Summarization Settings")
    summary_length = st.sidebar.selectbox("Summary Length", ["Short (30-80 words)", "Medium (80-130 words)", "Long (130-200 words)"])
    
    if summary_length == "Short (30-80 words)":
        max_len, min_len = 80, 30
    elif summary_length == "Medium (80-130 words)":
        max_len, min_len = 130, 80
    else:
        max_len, min_len = 200, 130

# --- Main Content Area ---
if analysis_mode == "Content Scoring":
    st.header("📊 Content Scoring Analysis")
    
    # Input method selection
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
        scorer = ContentScorer(
            get_sentiment_model(),
            openai_api_key
        )
        
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

elif analysis_mode == "Text Summarization":
    st.header("📝 Text Summarization")
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload File"], horizontal=True, key="sum_input")
    
    content_input = ""
    
    if input_method == "Type/Paste Text":
        content_input = st.text_area("Enter text to summarize", height=300, placeholder="Paste your text here...")
    else:
        supported_formats = FileProcessor.get_supported_formats()
        st.info(f"📁 Supported formats: {', '.join(supported_formats).upper()}")
        
        uploaded_file = st.file_uploader("Upload your content file", type=supported_formats, key="sum_upload")
        
        if uploaded_file is not None:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                content_input = FileProcessor.extract_text_from_file(uploaded_file)
                
            if content_input:
                st.success(f"✅ Successfully processed {uploaded_file.name}")
    
    if content_input and st.button("📝 Generate Summary", type="primary"):
        summarizer = ContentSummarizer(get_summarizer_model())
        
        with st.spinner("Generating summary..."):
            summary = summarizer.summarize_text(content_input, max_len, min_len)
            themes = summarizer.extract_themes(content_input)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📄 Summary")
            st.write(summary['summary_text'])
        
        with col2:
            st.subheader("📊 Text Statistics")
            words = len(content_input.split())
            sentences = len(re.split(r'[.!?]+', content_input))
            st.metric("Word Count", words)
            st.metric("Sentences", sentences)
            st.metric("Avg Words/Sentence", f"{words/sentences:.1f}" if sentences > 0 else "0")
        
        st.subheader("🎯 Main Themes")
        st.write(themes['summary_text'])
        
        # Word cloud
        st.subheader("☁️ Word Cloud")
        try:
            def clean_text_for_cloud(text):
                text = re.sub(r'[^\w\s]', '', text)
                text = text.lower()
                return text
            
            wc = WordCloud(width=800, height=400, background_color='white').generate(clean_text_for_cloud(content_input))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        except:
            st.info("Word cloud generation not available")

elif analysis_mode == "Combined Analysis":
    st.header("🔬 Combined Content Analysis")
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Type/Paste Text", "Upload File"], horizontal=True, key="combined_input")
    
    content_input = ""
    
    if input_method == "Type/Paste Text":
        content_input = st.text_area("Enter content to analyze", height=300, placeholder="Paste your content here...")
    else:
        supported_formats = FileProcessor.get_supported_formats()
        st.info(f"📁 Supported formats: {', '.join(supported_formats).upper()}")
        
        uploaded_file = st.file_uploader("Upload your content file", type=supported_formats, key="combined_upload")
        
        if uploaded_file is not None:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                content_input = FileProcessor.extract_text_from_file(uploaded_file)
                
            if content_input:
                st.success(f"✅ Successfully processed {uploaded_file.name}")
    
    if content_input and st.button("🔬 Full Analysis", type="primary"):
        scorer = ContentScorer(get_sentiment_model(), openai_api_key)
        summarizer = ContentSummarizer(get_summarizer_model())
        
        with st.spinner("Performing comprehensive analysis..."):
            # Content scoring
            score_results = scorer.comprehensive_score(content_input, target_keywords)
            
            # Summarization
            summary = summarizer.summarize_text(content_input, max_len, min_len)
            themes = summarizer.extract_themes(content_input)
        
        # Layout with tabs
        tab1, tab2, tab3 = st.tabs(["📊 Content Scores", "📝 Summary & Themes", "📈 Visualizations"])
        
        with tab1:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.metric("Overall Score", f"{score_results['overall_score']:.1f}/100")
            with col2:
                grade = "A+" if score_results['overall_score'] >= 90 else "A" if score_results['overall_score'] >= 80 else "B" if score_results['overall_score'] >= 70 else "C" if score_results['overall_score'] >= 60 else "D"
                st.metric("Grade", grade)
            with col3:
                word_count = len(content_input.split())
                st.metric("Word Count", word_count)
            
            score_cols = st.columns(4)
            scores = [
                ("Readability", score_results['readability']['readability_score'], "📚"),
                ("Sentiment", score_results['sentiment']['sentiment_score'], "😊"),
                ("Engagement", score_results['engagement']['engagement_score'], "🎯"),
                ("SEO", score_results['seo']['seo_score'], "🔍")
            ]
            
            for i, (name, score, icon) in enumerate(scores):
                with score_cols[i]:
                    st.metric(f"{icon} {name}", f"{score:.1f}")
        
        with tab2:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📄 Summary")
                st.write(summary['summary_text'])
                
                st.subheader("🎯 Main Themes")
                st.write(themes['summary_text'])
            
            with col2:
                st.subheader("📊 Text Statistics")
                words = len(content_input.split())
                sentences = len(re.split(r'[.!?]+', content_input))
                reading_level = score_results['readability']['readability_level']
                sentiment = score_results['sentiment']['sentiment']
                
                st.metric("Words", words)
                st.metric("Sentences", sentences)
                st.metric("Reading Level", reading_level)
                st.metric("Sentiment", sentiment)
        
        with tab3:
            # Radar chart
            st.subheader("📈 Content Scores")
            categories = ['Readability', 'Sentiment', 'Engagement', 'SEO']
            values = [
                score_results['readability']['readability_score'],
                score_results['sentiment']['sentiment_score'],
                score_results['engagement']['engagement_score'],
                score_results['seo']['seo_score']
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name='Content Score'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Content Scoring Radar Chart")
            st.plotly_chart(fig)
            
            # Word cloud
            st.subheader("☁️ Word Cloud")
            try:
                def clean_text_for_cloud(text):
                    text = re.sub(r'[^\w\s]', '', text)
                    text = text.lower()
                    return text
                
                wc = WordCloud(width=800, height=400, background_color='white').generate(clean_text_for_cloud(content_input))
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            except:
                st.info("Word cloud generation not available")

elif analysis_mode == "Document Comparison":
    st.header("⚖️ Document Comparison")
    
    st.write("Upload or paste multiple documents to compare their content, quality, and themes")
    
    # Number of documents to compare
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
            doc_content = st.text_area(f"Document {i+1} Content", height=150, key=f"doc_text_{i}")
        else:
            supported_formats = FileProcessor.get_supported_formats()
            uploaded_file = st.file_uploader(f"Upload Document {i+1}", type=supported_formats, key=f"doc_file_{i}")
            
            if uploaded_file is not None:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    doc_content = FileProcessor.extract_text_from_file(uploaded_file)
                    
                if doc_content:
                    st.success(f"✅ Processed {uploaded_file.name}")
                    doc_names[i] = uploaded_file.name.split('.')[0]  # Use filename as name
            else:
                doc_content = ""
        
        docs.append(doc_content)
    
    if all(d.strip() for d in docs) and st.button("⚖️ Compare Documents", type="primary"):
        # Load models only when needed for comparison
        scorer = ContentScorer(get_sentiment_model(), openai_api_key)
        summarizer = ContentSummarizer(get_summarizer_model())
        
        with st.spinner("Analyzing all documents..."):
            # Score all documents
            all_results = []
            progress_bar = st.progress(0)
            total_steps = len(docs) * 2 + 1  # scoring + summarizing + comparison
            current_step = 0
            
            for i, doc in enumerate(docs):
                st.write(f"📊 Scoring document {i+1}: {doc_names[i]}")
                result = scorer.comprehensive_score(doc, target_keywords)
                result['name'] = doc_names[i]
                result['word_count'] = len(doc.split())
                all_results.append(result)
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            # Generate summaries (limit text length to avoid memory issues)
            summaries = []
            for i, doc in enumerate(docs):
                st.write(f"📝 Summarizing document {i+1}: {doc_names[i]}")
                # Limit doc length for summarization to avoid memory issues
                doc_limited = doc[:2000] if len(doc) > 2000 else doc
                summary = summarizer.summarize_text(doc_limited, max_len, min_len)
                summaries.append(summary['summary_text'])
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            # Compare documents (using limited text)
            st.write("🔍 Generating comparative analysis...")
            docs_limited = [doc[:1000] for doc in docs]  # Further limit for comparison
            comparison = summarizer.compare_texts(docs_limited)
            current_step += 1
            progress_bar.progress(1.0)
            
            progress_bar.empty()
        
        # Results display
        st.subheader("📊 Comparison Results")
        
        # Create comparison DataFrame
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
        
        # Visualization
        fig = px.bar(
            df_scores.melt(id_vars=['Document'], value_vars=['Overall Score', 'Readability', 'Sentiment', 'Engagement', 'SEO'], 
                          var_name='Metric', value_name='Score'),
            x='Document',
            y='Score',
            color='Metric',
            title="Document Scores Comparison",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best and worst performers
        best_doc = df_scores.loc[df_scores['Overall Score'].idxmax()]
        worst_doc = df_scores.loc[df_scores['Overall Score'].idxmin()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🏆 **Best Performer:** {best_doc['Document']} ({best_doc['Overall Score']:.1f})")
        with col2:
            st.error(f"⚠️ **Needs Improvement:** {worst_doc['Document']} ({worst_doc['Overall Score']:.1f})")
        
        # Document summaries
        st.subheader("📝 Document Summaries")
        for i, (name, summary) in enumerate(zip(doc_names, summaries)):
            with st.expander(f"📄 {name} Summary"):
                st.write(summary)
        
        # Comparison analysis
        st.subheader("🔍 Comparative Analysis")
        st.write(comparison['summary_text'])

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🔬 Unified Content Analysis Tool | Content Scoring + Text Summarization</p>
    </div>
    """,
    unsafe_allow_html=True
)
