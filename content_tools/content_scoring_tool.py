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
from transformers import pipeline
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import io

# Optional imports
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass  # seaborn is optional for styling

# Optional imports with error handling
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("⚠️ python-docx not installed. Word document support disabled. Install with: pip install python-docx")

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ PyPDF2 not installed. PDF support disabled. Install with: pip install PyPDF2")

# --- Configuration ---
st.set_page_config(
    page_title="Content Scoring Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Instructions for API Key ---
st.markdown("""
**Internal Content Scoring Tool**
- Set your OpenAI API key in your environment as `OPENAI_API_KEY`, or add it to Streamlit secrets.
- This tool evaluates content across multiple dimensions for quality assessment.
""")

# --- Set your OpenAI API key ---
openai_api_key = os.environ.get("OPENAI_API_KEY")
try:
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass
if not openai_api_key:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

if openai_api_key:
    openai.api_key = openai_api_key

st.title("📊 Internal Content Scoring Tool")

# --- Load NLP Models (cached) ---
@st.cache_resource
def load_models():
    try:
        # Try loading the preferred models first
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        return sentiment_analyzer, summarizer, True
    except Exception as e:
        st.warning(f"Could not load advanced models: {str(e)}. Using basic sentiment analysis.")
        # Return None for models that failed to load
        return None, None, False

def load_models_safely():
    """Load models with error handling and fallbacks"""
    if 'models_loaded' not in st.session_state:
        with st.spinner("Loading AI models..."):
            sentiment_analyzer, summarizer, success = load_models()
            st.session_state.models_loaded = True
            st.session_state.sentiment_analyzer = sentiment_analyzer
            st.session_state.summarizer = summarizer
            st.session_state.models_success = success
            
            if not success:
                st.info("Using lightweight sentiment analysis. For full features, please upgrade PyTorch to version >=2.6.0")

# Load models
load_models_safely()

# --- File Processing Functions ---
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
        """Extract text from TXT file"""
        return uploaded_file.read().decode("utf-8")
    
    @staticmethod
    def _extract_from_pdf(uploaded_file):
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            st.error("PDF support not available. Please install PyPDF2: pip install PyPDF2")
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
        """Extract text from DOCX file"""
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
            st.error(f"Error reading DOCX: {str(e)}")
            return None
    
    @staticmethod
    def _extract_from_markdown(uploaded_file):
        """Extract text from Markdown file"""
        content = uploaded_file.read().decode("utf-8")
        # Remove markdown formatting for analysis
        # Remove headers
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # Remove bold/italic
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)
        content = re.sub(r'\*(.*?)\*', r'\1', content)
        # Remove links
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        # Remove code blocks
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        return content
    
    @staticmethod
    def get_supported_formats():
        """Return list of supported file formats based on available dependencies"""
        formats = ["txt", "md"]  # Always available
        if PDF_AVAILABLE:
            formats.append("pdf")
        if DOCX_AVAILABLE:
            formats.extend(["docx", "doc"])
        return formats

# --- Content Scoring Functions ---
class ContentScorer:
    def __init__(self, sentiment_analyzer, summarizer, openai_api_key=None):
        self.sentiment_analyzer = sentiment_analyzer
        self.summarizer = summarizer
        self.openai_api_key = openai_api_key
    
    def readability_score(self, text):
        """Calculate readability metrics"""
        try:
            flesch_ease = flesch_reading_ease(text)
            fk_grade = flesch_kincaid_grade(text)
            ari = automated_readability_index(text)
            
            # Normalize to 0-100 scale
            readability_score = max(0, min(100, flesch_ease))
            
            return {
                'flesch_ease': flesch_ease,
                'fk_grade': fk_grade,
                'ari': ari,
                'readability_score': readability_score,
                'readability_level': self._get_readability_level(flesch_ease)
            }
        except:
            return {
                'flesch_ease': 0,
                'fk_grade': 0,
                'ari': 0,
                'readability_score': 0,
                'readability_level': 'Unknown'
            }
    
    def _get_readability_level(self, flesch_score):
        if flesch_score >= 90:
            return "Very Easy"
        elif flesch_score >= 80:
            return "Easy"
        elif flesch_score >= 70:
            return "Fairly Easy"
        elif flesch_score >= 60:
            return "Standard"
        elif flesch_score >= 50:
            return "Fairly Difficult"
        elif flesch_score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def sentiment_score(self, text):
        """Analyze sentiment and emotional tone"""
        if self.sentiment_analyzer is not None:
            try:
                # Split long text into chunks
                max_length = 512
                chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
                
                sentiments = []
                for chunk in chunks[:5]:  # Limit to first 5 chunks
                    if chunk.strip():
                        result = self.sentiment_analyzer(chunk)
                        sentiments.append(result[0])
                
                if not sentiments:
                    return {'sentiment': 'NEUTRAL', 'confidence': 0.5, 'sentiment_score': 50}
                
                # Handle different label formats (POSITIVE/NEGATIVE vs LABEL_1/LABEL_0)
                positive_scores = []
                negative_scores = []
                
                for s in sentiments:
                    if s['label'] in ['POSITIVE', 'LABEL_1']:
                        positive_scores.append(s['score'])
                    elif s['label'] in ['NEGATIVE', 'LABEL_0']:
                        negative_scores.append(s['score'])
                
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
                
                return {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'sentiment_score': sentiment_score
                }
            except Exception as e:
                # Fall back to basic sentiment analysis
                return self._basic_sentiment_analysis(text)
        else:
            # Use basic sentiment analysis when models aren't available
            return self._basic_sentiment_analysis(text)
    
    def _basic_sentiment_analysis(self, text):
        """Basic sentiment analysis using word lists"""
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like',
            'best', 'awesome', 'perfect', 'brilliant', 'outstanding', 'superb', 'magnificent',
            'happy', 'pleased', 'satisfied', 'delighted', 'thrilled', 'excited', 'positive'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike', 'poor',
            'disappointing', 'frustrating', 'annoying', 'sad', 'angry', 'upset', 'negative',
            'problem', 'issue', 'difficult', 'hard', 'challenging', 'failed', 'wrong'
        ]
        
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
            confidence = min(0.8, positive_ratio * 10)  # Cap confidence at 0.8 for basic analysis
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
            'sentiment_score': sentiment_score
        }
    
    def engagement_score(self, text):
        """Calculate engagement potential based on various factors"""
        # Word count
        words = len(text.split())
        word_score = min(100, (words / 300) * 100) if words <= 300 else max(0, 100 - ((words - 300) / 50) * 10)
        
        # Question marks (engagement)
        questions = text.count('?')
        question_score = min(100, questions * 20)
        
        # Exclamation marks (excitement)
        exclamations = text.count('!')
        excitement_score = min(100, exclamations * 15)
        
        # Call-to-action words
        cta_words = ['click', 'buy', 'subscribe', 'join', 'download', 'learn', 'discover', 'try', 'get', 'start']
        cta_count = sum([text.lower().count(word) for word in cta_words])
        cta_score = min(100, cta_count * 25)
        
        # Power words
        power_words = ['amazing', 'incredible', 'exclusive', 'proven', 'guaranteed', 'instant', 'ultimate', 'secret', 'powerful', 'revolutionary']
        power_count = sum([text.lower().count(word) for word in power_words])
        power_score = min(100, power_count * 20)
        
        # Calculate weighted engagement score
        engagement_score = (
            word_score * 0.3 +
            question_score * 0.2 +
            excitement_score * 0.15 +
            cta_score * 0.2 +
            power_score * 0.15
        )
        
        return {
            'engagement_score': engagement_score,
            'word_count': words,
            'questions': questions,
            'exclamations': exclamations,
            'cta_count': cta_count,
            'power_words': power_count
        }
    
    def seo_score(self, text, target_keywords=None):
        """Calculate SEO-related metrics"""
        words = text.split()
        
        # Keyword density
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
            keyword_score = 50  # Neutral score if no keywords provided
        
        # Content length score
        length_score = min(100, (len(words) / 300) * 100) if len(words) <= 300 else max(50, 100 - ((len(words) - 300) / 100) * 5)
        
        # Heading structure (based on common patterns)
        headings = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
        heading_score = min(100, headings * 25)
        
        seo_score = (keyword_score * 0.4 + length_score * 0.4 + heading_score * 0.2)
        
        return {
            'seo_score': seo_score,
            'keyword_density': keyword_density,
            'word_count': len(words),
            'headings_count': headings
        }
    
    def ai_quality_score(self, text):
        """Use OpenAI to assess content quality"""
        if not self.openai_api_key:
            return {'quality_score': 75, 'feedback': 'OpenAI API key not provided'}
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a content quality expert. Rate the following content on a scale of 0-100 based on clarity, coherence, value, and overall quality. Provide a brief explanation."},
                    {"role": "user", "content": f"Please rate this content:\n\n{text[:1000]}..."}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            
            # Extract score from response
            score_match = re.search(r'(\d+)(?:/100)?', response_text)
            quality_score = int(score_match.group(1)) if score_match else 75
            
            return {
                'quality_score': quality_score,
                'feedback': response_text
            }
        except:
            return {'quality_score': 75, 'feedback': 'Could not assess with AI'}
    
    def comprehensive_score(self, text, target_keywords=None):
        """Generate comprehensive content score"""
        readability = self.readability_score(text)
        sentiment = self.sentiment_score(text)
        engagement = self.engagement_score(text)
        seo = self.seo_score(text, target_keywords)
        ai_quality = self.ai_quality_score(text)
        
        # Calculate overall score (weighted average)
        overall_score = (
            readability['readability_score'] * 0.2 +
            sentiment['sentiment_score'] * 0.15 +
            engagement['engagement_score'] * 0.25 +
            seo['seo_score'] * 0.2 +
            ai_quality['quality_score'] * 0.2
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

# --- Sidebar Configuration ---
st.sidebar.header("📋 Scoring Configuration")

# Scoring mode selection
scoring_mode = st.sidebar.selectbox(
    "Select Scoring Mode",
    ["Single Content", "Batch Analysis", "Comparative Analysis"]
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

# --- Main Content Area ---
if scoring_mode == "Single Content":
    st.header("📝 Single Content Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Type/Paste Text", "Upload File"],
        horizontal=True
    )
    
    content_input = ""
    
    if input_method == "Type/Paste Text":
        content_input = st.text_area(
            "Enter content to analyze",
            height=300,
            placeholder="Paste your content here for comprehensive scoring..."
        )
    else:
        supported_formats = FileProcessor.get_supported_formats()
        st.info(f"📁 Supported formats: {', '.join(supported_formats).upper()}")
        
        uploaded_file = st.file_uploader(
            "Upload your content file",
            type=supported_formats,
            help="Upload a document to analyze its content"
        )
        
        if uploaded_file is not None:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                content_input = FileProcessor.extract_text_from_file(uploaded_file)
                
            if content_input:
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                
                # Show preview of extracted text
                with st.expander("📄 Preview extracted text"):
                    preview_text = content_input[:500] + "..." if len(content_input) > 500 else content_input
                    st.text_area("Extracted content preview:", preview_text, height=150, disabled=True)
            else:
                st.error("❌ Failed to extract text from file")
    
    if content_input and st.button("🔍 Analyze Content", type="primary"):
        if 'models_loaded' in st.session_state:
            scorer = ContentScorer(
                st.session_state.sentiment_analyzer,
                st.session_state.summarizer,
                openai_api_key
            )
            
            with st.spinner("Analyzing content..."):
                results = scorer.comprehensive_score(content_input, target_keywords)
            
            # Display overall score
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.metric(
                    "Overall Content Score",
                    f"{results['overall_score']:.1f}/100",
                    delta=None
                )
            
            with col2:
                grade = "A+" if results['overall_score'] >= 90 else "A" if results['overall_score'] >= 80 else "B" if results['overall_score'] >= 70 else "C" if results['overall_score'] >= 60 else "D"
                st.metric("Grade", grade)
            
            with col3:
                word_count = len(content_input.split())
                st.metric("Word Count", word_count)
            
            # Detailed scores
            st.subheader("📊 Detailed Scoring Breakdown")
            
            score_cols = st.columns(5)
            scores = [
                ("Readability", results['readability']['readability_score'], "📚"),
                ("Sentiment", results['sentiment']['sentiment_score'], "😊"),
                ("Engagement", results['engagement']['engagement_score'], "🎯"),
                ("SEO", results['seo']['seo_score'], "🔍"),
                ("AI Quality", results['ai_quality']['quality_score'], "🤖")
            ]
            
            for i, (name, score, icon) in enumerate(scores):
                with score_cols[i]:
                    st.metric(f"{icon} {name}", f"{score:.1f}")
            
            # Visualizations
            st.subheader("📈 Score Visualization")
            
            # Radar chart
            categories = ['Readability', 'Sentiment', 'Engagement', 'SEO', 'AI Quality']
            values = [
                results['readability']['readability_score'],
                results['sentiment']['sentiment_score'],
                results['engagement']['engagement_score'],
                results['seo']['seo_score'],
                results['ai_quality']['quality_score']
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Content Score'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Content Scoring Radar Chart"
            )
            st.plotly_chart(fig)
            
            # Detailed insights
            st.subheader("🔍 Detailed Insights")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Readability", "😊 Sentiment", "🎯 Engagement", "🔍 SEO", "🤖 AI Quality"])
            
            with tab1:
                st.write(f"**Reading Level:** {results['readability']['readability_level']}")
                st.write(f"**Flesch Reading Ease:** {results['readability']['flesch_ease']:.1f}")
                st.write(f"**Grade Level:** {results['readability']['fk_grade']:.1f}")
                
            with tab2:
                st.write(f"**Overall Sentiment:** {results['sentiment']['sentiment']}")
                st.write(f"**Confidence:** {results['sentiment']['confidence']:.2f}")
                
            with tab3:
                eng = results['engagement']
                st.write(f"**Questions:** {eng['questions']}")
                st.write(f"**Exclamation Points:** {eng['exclamations']}")
                st.write(f"**Call-to-Action Words:** {eng['cta_count']}")
                st.write(f"**Power Words:** {eng['power_words']}")
                
            with tab4:
                if target_keywords:
                    st.write("**Keyword Density:**")
                    for keyword, density in results['seo']['keyword_density'].items():
                        st.write(f"- {keyword}: {density:.2f}%")
                st.write(f"**Headings Found:** {results['seo']['headings_count']}")
                
            with tab5:
                st.write("**AI Feedback:**")
                st.write(results['ai_quality']['feedback'])
            
            # Export results
            st.subheader("💾 Export Results")
            if st.button("📊 Export as JSON"):
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=json_str,
                    file_name=f"content_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

elif scoring_mode == "Batch Analysis":
    st.header("📁 Batch Content Analysis")
    
    # Input method for batch analysis
    batch_input_method = st.radio(
        "Choose batch input method:",
        ["Type/Paste Text", "Upload Files"],
        horizontal=True,
        key="batch_input_method"
    )
    
    if batch_input_method == "Type/Paste Text":
        num_contents = st.number_input("Number of content pieces", min_value=2, max_value=10, value=3)
        
        contents = []
        content_names = []
        
        for i in range(num_contents):
            col1, col2 = st.columns([1, 3])
            with col1:
                name = st.text_input(f"Name {i+1}", value=f"Content {i+1}", key=f"name_{i}")
                content_names.append(name)
            with col2:
                content = st.text_area(f"Content {i+1}", height=100, key=f"batch_content_{i}")
                contents.append(content)
    
    else:  # Upload Files
        supported_formats = FileProcessor.get_supported_formats()
        st.info(f"📁 Supported formats: {', '.join(supported_formats).upper()}")
        
        uploaded_files = st.file_uploader(
            "Upload multiple content files",
            type=supported_formats,
            accept_multiple_files=True,
            help="Upload multiple documents to analyze and compare"
        )
        
        contents = []
        content_names = []
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    content = FileProcessor.extract_text_from_file(uploaded_file)
                    
                if content:
                    contents.append(content)
                    # Use filename without extension as name
                    name = ".".join(uploaded_file.name.split(".")[:-1])
                    content_names.append(name)
                    st.success(f"✅ Processed {uploaded_file.name}")
                else:
                    st.error(f"❌ Failed to process {uploaded_file.name}")
        
        if contents:
            st.info(f"📊 Ready to analyze {len(contents)} files")
            
            # Show preview of extracted files
            with st.expander("📄 Preview extracted content"):
                for i, (name, content) in enumerate(zip(content_names, contents)):
                    st.subheader(f"{i+1}. {name}")
                    preview_text = content[:200] + "..." if len(content) > 200 else content
                    st.text(preview_text)
                    st.write(f"**Length:** {len(content.split())} words")
                    st.divider()
    
    if all(c.strip() for c in contents) and st.button("🔍 Analyze All Content", type="primary"):
        if 'models_loaded' in st.session_state:
            scorer = ContentScorer(
                st.session_state.sentiment_analyzer,
                st.session_state.summarizer,
                openai_api_key
            )
            
            results_list = []
            
            with st.spinner("Analyzing all content pieces..."):
                for i, content in enumerate(contents):
                    result = scorer.comprehensive_score(content, target_keywords)
                    result['name'] = content_names[i]
                    result['content_preview'] = content[:100] + "..." if len(content) > 100 else content
                    results_list.append(result)
            
            # Create comparison DataFrame
            df_scores = pd.DataFrame([
                {
                    'Content': r['name'],
                    'Overall Score': r['overall_score'],
                    'Readability': r['readability']['readability_score'],
                    'Sentiment': r['sentiment']['sentiment_score'],
                    'Engagement': r['engagement']['engagement_score'],
                    'SEO': r['seo']['seo_score'],
                    'AI Quality': r['ai_quality']['quality_score']
                }
                for r in results_list
            ])
            
            # Display results
            st.subheader("📊 Batch Analysis Results")
            st.dataframe(df_scores, use_container_width=True)
            
            # Visualization
            fig = px.bar(
                df_scores.melt(id_vars=['Content'], var_name='Metric', value_name='Score'),
                x='Content',
                y='Score',
                color='Metric',
                title="Content Scores Comparison",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Best and worst performers
            best_content = df_scores.loc[df_scores['Overall Score'].idxmax()]
            worst_content = df_scores.loc[df_scores['Overall Score'].idxmin()]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 **Best Performer:** {best_content['Content']} ({best_content['Overall Score']:.1f})")
            with col2:
                st.error(f"⚠️ **Needs Improvement:** {worst_content['Content']} ({worst_content['Overall Score']:.1f})")

else:  # Comparative Analysis
    st.header("⚖️ Comparative Content Analysis")
    
    st.write("Compare two pieces of content side by side")
    
    # Input method for comparative analysis
    comp_input_method = st.radio(
        "Choose comparison input method:",
        ["Type/Paste Text", "Upload Files"],
        horizontal=True,
        key="comp_input_method"
    )
    
    content_a = ""
    content_b = ""
    
    if comp_input_method == "Type/Paste Text":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Content A")
            content_a = st.text_area("Content A", height=200, key="content_a")
            
        with col2:
            st.subheader("Content B")
            content_b = st.text_area("Content B", height=200, key="content_b")
    
    else:  # Upload Files
        supported_formats = FileProcessor.get_supported_formats()
        st.info(f"📁 Supported formats: {', '.join(supported_formats).upper()}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Content A")
            uploaded_file_a = st.file_uploader(
                "Upload first file",
                type=supported_formats,
                key="file_a",
                help="Upload the first document to compare"
            )
            
            if uploaded_file_a is not None:
                with st.spinner(f"Processing {uploaded_file_a.name}..."):
                    content_a = FileProcessor.extract_text_from_file(uploaded_file_a)
                    
                if content_a:
                    st.success(f"✅ Processed {uploaded_file_a.name}")
                    with st.expander("Preview Content A"):
                        preview_a = content_a[:300] + "..." if len(content_a) > 300 else content_a
                        st.text_area("Content A preview:", preview_a, height=100, disabled=True)
                else:
                    st.error(f"❌ Failed to process {uploaded_file_a.name}")
        
        with col2:
            st.subheader("📄 Content B")
            uploaded_file_b = st.file_uploader(
                "Upload second file",
                type=supported_formats,
                key="file_b",
                help="Upload the second document to compare"
            )
            
            if uploaded_file_b is not None:
                with st.spinner(f"Processing {uploaded_file_b.name}..."):
                    content_b = FileProcessor.extract_text_from_file(uploaded_file_b)
                    
                if content_b:
                    st.success(f"✅ Processed {uploaded_file_b.name}")
                    with st.expander("Preview Content B"):
                        preview_b = content_b[:300] + "..." if len(content_b) > 300 else content_b
                        st.text_area("Content B preview:", preview_b, height=100, disabled=True)
                else:
                    st.error(f"❌ Failed to process {uploaded_file_b.name}")
    
    if content_a and content_b and st.button("🔍 Compare Content", type="primary"):
        if 'models_loaded' in st.session_state:
            scorer = ContentScorer(
                st.session_state.sentiment_analyzer,
                st.session_state.summarizer,
                openai_api_key
            )
            
            with st.spinner("Comparing content..."):
                results_a = scorer.comprehensive_score(content_a, target_keywords)
                results_b = scorer.comprehensive_score(content_b, target_keywords)
            
            # Comparison metrics
            st.subheader("📊 Side-by-Side Comparison")
            
            comparison_data = {
                'Metric': ['Overall Score', 'Readability', 'Sentiment', 'Engagement', 'SEO', 'AI Quality'],
                'Content A': [
                    results_a['overall_score'],
                    results_a['readability']['readability_score'],
                    results_a['sentiment']['sentiment_score'],
                    results_a['engagement']['engagement_score'],
                    results_a['seo']['seo_score'],
                    results_a['ai_quality']['quality_score']
                ],
                'Content B': [
                    results_b['overall_score'],
                    results_b['readability']['readability_score'],
                    results_b['sentiment']['sentiment_score'],
                    results_b['engagement']['engagement_score'],
                    results_b['seo']['seo_score'],
                    results_b['ai_quality']['quality_score']
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison['Difference'] = df_comparison['Content B'] - df_comparison['Content A']
            df_comparison['Winner'] = df_comparison.apply(
                lambda row: 'Content B' if row['Difference'] > 0 else 'Content A' if row['Difference'] < 0 else 'Tie', 
                axis=1
            )
            
            st.dataframe(df_comparison, use_container_width=True)
            
            # Winner summary
            winner_count_a = (df_comparison['Winner'] == 'Content A').sum()
            winner_count_b = (df_comparison['Winner'] == 'Content B').sum()
            
            if winner_count_a > winner_count_b:
                st.success("🏆 **Overall Winner: Content A**")
            elif winner_count_b > winner_count_a:
                st.success("🏆 **Overall Winner: Content B**")
            else:
                st.info("🤝 **Result: It's a tie!**")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Internal Content Scoring Tool | Built with Streamlit & AI</p>
    </div>
    """,
    unsafe_allow_html=True
)
