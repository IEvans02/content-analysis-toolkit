import sqlite3
import json
import hashlib
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px

class FeedbackManager:
    def __init__(self, db_path="client_feedback.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_type TEXT NOT NULL,
                content_hash TEXT,
                content_preview TEXT,
                ai_response TEXT,
                rating TEXT,
                rating_numeric INTEGER,
                comments TEXT,
                client_session TEXT,
                analysis_mode TEXT,
                timestamp TEXT,
                improvement_areas TEXT,
                missing_features TEXT,
                overall_score REAL,
                accuracy_rating INTEGER,
                use_case TEXT
            )
        ''')
        
        # Quick feedback table for thumbs up/down
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quick_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_type TEXT NOT NULL,
                content_hash TEXT,
                positive BOOLEAN,
                analysis_type TEXT,
                timestamp TEXT
            )
        ''')
        
        # Corrections table for training data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT,
                original_prediction TEXT,
                corrected_value TEXT,
                correction_type TEXT,
                explanation TEXT,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_feedback(self, feedback_data):
        """Store detailed feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback 
            (feedback_type, content_hash, content_preview, ai_response, rating, 
             rating_numeric, comments, client_session, analysis_mode, timestamp,
             improvement_areas, missing_features, overall_score, accuracy_rating, use_case)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_data.get('type'),
            feedback_data.get('content_hash'),
            feedback_data.get('content_preview'),
            feedback_data.get('ai_response'),
            feedback_data.get('rating'),
            self._rating_to_numeric(feedback_data.get('rating')),
            feedback_data.get('comments'),
            feedback_data.get('client_session', st.session_state.get('session_id', 'anonymous')),
            feedback_data.get('analysis_mode'),
            feedback_data.get('timestamp'),
            json.dumps(feedback_data.get('improvement_areas', [])),
            json.dumps(feedback_data.get('missing_features', [])),
            feedback_data.get('overall_score'),
            feedback_data.get('accuracy_rating'),
            feedback_data.get('use_case')
        ))
        
        conn.commit()
        conn.close()
        return True
    
    def store_quick_feedback(self, feedback_type, content_hash, positive, analysis_type):
        """Store quick thumbs up/down feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quick_feedback 
            (feedback_type, content_hash, positive, analysis_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (feedback_type, content_hash, positive, analysis_type, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def store_correction(self, correction_data):
        """Store user corrections for training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO corrections 
            (content_hash, original_prediction, corrected_value, correction_type, explanation, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            correction_data.get('content_hash'),
            correction_data.get('original_prediction'),
            correction_data.get('corrected_value'),
            correction_data.get('correction_type'),
            correction_data.get('explanation'),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_feedback_analytics(self, days=30):
        """Get feedback analytics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        
        # Get detailed feedback
        feedback_df = pd.read_sql_query('''
            SELECT * FROM feedback 
            WHERE datetime(timestamp) >= datetime('now', '-{} days')
        '''.format(days), conn)
        
        # Get quick feedback
        quick_df = pd.read_sql_query('''
            SELECT * FROM quick_feedback 
            WHERE datetime(timestamp) >= datetime('now', '-{} days')
        '''.format(days), conn)
        
        # Get corrections
        corrections_df = pd.read_sql_query('''
            SELECT * FROM corrections 
            WHERE datetime(timestamp) >= datetime('now', '-{} days')
        '''.format(days), conn)
        
        conn.close()
        
        return {
            'detailed_feedback': feedback_df,
            'quick_feedback': quick_df,
            'corrections': corrections_df,
            'avg_rating': feedback_df['rating_numeric'].mean() if not feedback_df.empty else 0,
            'total_feedback': len(feedback_df) + len(quick_df),
            'positive_ratio': quick_df['positive'].mean() if not quick_df.empty else 0,
            'avg_accuracy': feedback_df['accuracy_rating'].mean() if not feedback_df.empty and 'accuracy_rating' in feedback_df.columns else 0
        }
    
    def _rating_to_numeric(self, rating):
        """Convert rating to numeric value"""
        rating_map = {
            'Poor': 1,
            'Fair': 2, 
            'Good': 3,
            'Excellent': 4,
            'Not useful': 1,
            'Somewhat useful': 2,
            'Very useful': 3,
            'Extremely useful': 4
        }
        return rating_map.get(rating, 3)
    
    def get_content_hash(self, content):
        """Generate hash for content"""
        return hashlib.md5(content.encode()).hexdigest()[:12]

def render_feedback_analytics():
    """Render feedback analytics dashboard"""
    st.markdown("## 📊 Client Feedback Analytics")
    
    feedback_manager = FeedbackManager()
    analytics = feedback_manager.get_feedback_analytics(days=30)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback", analytics['total_feedback'])
    
    with col2:
        avg_rating = analytics['avg_rating']
        st.metric("Avg Rating", f"{avg_rating:.1f}/4" if avg_rating > 0 else "N/A")
    
    with col3:
        positive_ratio = analytics['positive_ratio']
        st.metric("Positive Feedback", f"{positive_ratio:.1%}" if positive_ratio > 0 else "N/A")
    
    with col4:
        avg_accuracy = analytics['avg_accuracy']
        st.metric("Avg Accuracy", f"{avg_accuracy:.0f}%" if avg_accuracy > 0 else "N/A")
    
    if not analytics['detailed_feedback'].empty:
        st.markdown("### 📈 Feedback Trends")
        
        # Rating distribution
        fig = px.histogram(analytics['detailed_feedback'], x='rating', 
                         title="Rating Distribution",
                         color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Feedback by type
        if 'feedback_type' in analytics['detailed_feedback'].columns:
            feedback_by_type = analytics['detailed_feedback']['feedback_type'].value_counts()
            fig2 = px.pie(values=feedback_by_type.values, names=feedback_by_type.index,
                         title="Feedback by Analysis Type")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Recent comments
        st.markdown("### 💬 Recent Comments")
        recent_comments = analytics['detailed_feedback'].sort_values('timestamp', ascending=False).head(10)
        
        for _, row in recent_comments.iterrows():
            with st.expander(f"{row['feedback_type']} - {row['rating']} - {row['timestamp'][:10]}"):
                st.write(f"**Rating:** {row['rating']}")
                if pd.notna(row['comments']) and row['comments']:
                    st.write(f"**Comments:** {row['comments']}")
                if pd.notna(row['use_case']) and row['use_case']:
                    st.write(f"**Use Case:** {row['use_case']}")
                if pd.notna(row['missing_features']) and row['missing_features'] != '[]':
                    try:
                        missing = json.loads(row['missing_features'])
                        if missing:
                            st.write(f"**Missing Features:** {', '.join(missing)}")
                    except:
                        pass
        
        # Corrections data
        if not analytics['corrections'].empty:
            st.markdown("### 🔧 User Corrections")
            corrections_summary = analytics['corrections']['correction_type'].value_counts()
            fig3 = px.bar(x=corrections_summary.index, y=corrections_summary.values,
                         title="User Corrections by Type")
            st.plotly_chart(fig3, use_container_width=True)
    
    else:
        st.info("No feedback data available yet. Encourage users to provide feedback!")

def render_enhanced_feedback_section(results, content_input, use_llm_analysis, openai_api_key):
    """Render the enhanced feedback section with comprehensive feedback collection"""
    
    feedback_manager = FeedbackManager()
    content_hash = feedback_manager.get_content_hash(content_input)
    
    if use_llm_analysis and openai_api_key:
        st.subheader("🧠 AI-Powered Insights & Feedback")
        
        # Create tabs for AI insights and feedback
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎭 Sentiment Analysis", 
            "🎯 Engagement Analysis", 
            "🤖 Quality Assessment", 
            "📝 Your Feedback"
        ])
        
        with tab1:
            st.markdown("### Detailed Sentiment Analysis")
            if 'ai_reasoning' in results['sentiment']:
                st.write(results['sentiment']['ai_reasoning'])
                
                # Quick feedback buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("👍 Accurate", key="sentiment_good", help="This sentiment analysis was accurate"):
                        feedback_manager.store_quick_feedback("sentiment", content_hash, True, "sentiment_analysis")
                        st.success("✅ Thanks!")
                
                with col2:
                    if st.button("👎 Inaccurate", key="sentiment_bad", help="This sentiment analysis was wrong"):
                        feedback_manager.store_quick_feedback("sentiment", content_hash, False, "sentiment_analysis")
                        st.error("📝 We'll improve!")
                
                # Detailed feedback in expander
                with st.expander("💬 Provide Detailed Sentiment Feedback"):
                    sentiment_rating = st.select_slider(
                        "How accurate was this sentiment analysis?",
                        options=["Poor", "Fair", "Good", "Excellent"],
                        value="Good",
                        key="sentiment_rating"
                    )
                    
                    correct_sentiment = st.selectbox(
                        "What should the correct sentiment be?",
                        ["Positive", "Negative", "Neutral", "Mixed", "No correction needed"],
                        key="correct_sentiment"
                    )
                    
                    sentiment_comments = st.text_area(
                        "Additional comments:",
                        placeholder="What specific aspects were wrong or could be improved?",
                        key="sentiment_comments"
                    )
                    
                    if st.button("Submit Sentiment Feedback", key="submit_sentiment"):
                        feedback_data = {
                            'type': 'sentiment_analysis',
                            'content_hash': content_hash,
                            'content_preview': content_input[:200] + "...",
                            'ai_response': results['sentiment']['ai_reasoning'],
                            'rating': sentiment_rating,
                            'comments': sentiment_comments,
                            'timestamp': datetime.now().isoformat(),
                            'analysis_mode': 'AI-Enhanced'
                        }
                        
                        if correct_sentiment != "No correction needed":
                            feedback_manager.store_correction({
                                'content_hash': content_hash,
                                'original_prediction': results['sentiment']['sentiment'],
                                'corrected_value': correct_sentiment,
                                'correction_type': 'sentiment',
                                'explanation': sentiment_comments
                            })
                        
                        feedback_manager.store_feedback(feedback_data)
                        st.success("✅ Thank you for your detailed feedback!")
                        st.rerun()
            else:
                st.info("Using traditional sentiment analysis - upgrade to AI for enhanced insights")
        
        with tab2:
            st.markdown("### Engagement & Persuasiveness Analysis")
            if 'ai_reasoning' in results['engagement']:
                st.write(results['engagement']['ai_reasoning'])
                
                # Quick feedback
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("👍 Useful", key="engagement_good", help="This engagement analysis was helpful"):
                        feedback_manager.store_quick_feedback("engagement", content_hash, True, "engagement_analysis")
                        st.success("✅ Great!")
                
                with col2:
                    if st.button("👎 Not Helpful", key="engagement_bad", help="This engagement analysis missed the mark"):
                        feedback_manager.store_quick_feedback("engagement", content_hash, False, "engagement_analysis")
                        st.error("📝 We'll improve!")
                
                # Detailed feedback
                with st.expander("💬 Provide Detailed Engagement Feedback"):
                    engagement_rating = st.select_slider(
                        "How useful was this engagement analysis?",
                        options=["Poor", "Fair", "Good", "Excellent"],
                        value="Good",
                        key="engagement_rating"
                    )
                    
                    missing_aspects = st.multiselect(
                        "What engagement aspects were missed?",
                        ["Call-to-action effectiveness", "Emotional appeal", "Audience targeting", 
                         "Persuasion techniques", "Readability for audience", "Brand voice consistency"],
                        key="missing_engagement"
                    )
                    
                    engagement_comments = st.text_area(
                        "Specific feedback:",
                        placeholder="What engagement insights were missing or incorrect?",
                        key="engagement_comments"
                    )
                    
                    if st.button("Submit Engagement Feedback", key="submit_engagement"):
                        feedback_data = {
                            'type': 'engagement_analysis',
                            'content_hash': content_hash,
                            'content_preview': content_input[:200] + "...",
                            'ai_response': results['engagement']['ai_reasoning'],
                            'rating': engagement_rating,
                            'comments': engagement_comments,
                            'missing_features': missing_aspects,
                            'timestamp': datetime.now().isoformat(),
                            'analysis_mode': 'AI-Enhanced'
                        }
                        
                        feedback_manager.store_feedback(feedback_data)
                        st.success("✅ Engagement feedback recorded!")
                        st.rerun()
            else:
                st.info("Using traditional engagement analysis - upgrade to AI for enhanced insights")
        
        with tab3:
            st.markdown("### Content Quality Assessment")
            if results['ai_quality']['method'] == 'AI-Powered Analysis':
                st.write(results['ai_quality']['feedback'])
                
                # Quick feedback
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("👍 Accurate", key="quality_good", help="Quality assessment was spot-on"):
                        feedback_manager.store_quick_feedback("quality", content_hash, True, "quality_analysis")
                        st.success("✅ Excellent!")
                
                with col2:
                    if st.button("👎 Missed Issues", key="quality_bad", help="Quality assessment missed important issues"):
                        feedback_manager.store_quick_feedback("quality", content_hash, False, "quality_analysis")
                        st.error("📝 We'll improve!")
                
                # Detailed feedback
                with st.expander("💬 Provide Detailed Quality Feedback"):
                    quality_rating = st.select_slider(
                        "How accurate was this quality assessment?",
                        options=["Poor", "Fair", "Good", "Excellent"],
                        value="Good",
                        key="quality_rating"
                    )
                    
                    missed_quality_aspects = st.multiselect(
                        "What quality aspects were missed?",
                        ["Grammar issues", "Structure problems", "Clarity issues", 
                         "Factual accuracy", "Tone appropriateness", "Target audience fit"],
                        key="missed_quality"
                    )
                    
                    quality_comments = st.text_area(
                        "Quality feedback:",
                        placeholder="What quality issues were missed or incorrectly identified?",
                        key="quality_comments"
                    )
                    
                    if st.button("Submit Quality Feedback", key="submit_quality"):
                        feedback_data = {
                            'type': 'quality_assessment',
                            'content_hash': content_hash,
                            'content_preview': content_input[:200] + "...",
                            'ai_response': results['ai_quality']['feedback'],
                            'rating': quality_rating,
                            'comments': quality_comments,
                            'missing_features': missed_quality_aspects,
                            'timestamp': datetime.now().isoformat(),
                            'analysis_mode': 'AI-Enhanced'
                        }
                        
                        feedback_manager.store_feedback(feedback_data)
                        st.success("✅ Quality feedback saved!")
                        st.rerun()
            else:
                st.info("AI quality assessment not available - add your OpenAI API key for enhanced analysis")
        
        with tab4:
            st.markdown("### 🎯 Overall Analysis Feedback")
            st.write("Help us improve the entire analysis experience:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                overall_usefulness = st.select_slider(
                    "Overall usefulness of this analysis:",
                    options=["Not useful", "Somewhat useful", "Very useful", "Extremely useful"],
                    value="Very useful",
                    key="overall_usefulness"
                )
                
                accuracy_rating = st.slider(
                    "Overall accuracy (%)",
                    0, 100, 80,
                    key="accuracy_rating"
                )
                
                would_recommend = st.radio(
                    "Would you recommend this tool?",
                    ["Yes, definitely", "Yes, with improvements", "Maybe", "No"],
                    key="recommend"
                )
            
            with col2:
                missing_features = st.multiselect(
                    "What features would be valuable?",
                    ["Competitor comparison", "Industry benchmarks", "Action recommendations", 
                     "Content suggestions", "A/B test insights", "Performance predictions",
                     "Integration with other tools", "Custom analysis types"],
                    key="missing_features"
                )
                
                improvement_areas = st.multiselect(
                    "Priority improvement areas:",
                    ["Sentiment accuracy", "Engagement scoring", "Quality assessment", 
                     "Speed", "User interface", "Explanations", "Export options"],
                    key="improvement_areas"
                )
            
            use_case = st.text_input(
                "What's your primary use case?",
                placeholder="e.g., Marketing copy optimization, Blog post analysis...",
                key="use_case"
            )
            
            overall_comments = st.text_area(
                "Additional feedback and suggestions:",
                placeholder="What would make this tool more valuable for your work?",
                key="overall_comments"
            )
            
            if st.button("💾 Submit Comprehensive Feedback", type="primary", key="submit_comprehensive"):
                comprehensive_feedback = {
                    'type': 'comprehensive_feedback',
                    'content_hash': content_hash,
                    'overall_score': results.get('overall_score', 0),
                    'analysis_mode': 'AI-Enhanced' if openai_api_key else 'Traditional',
                    'rating': overall_usefulness,
                    'comments': f"Use case: {use_case}\n\nFeedback: {overall_comments}\n\nRecommendation: {would_recommend}",
                    'missing_features': missing_features,
                    'improvement_areas': improvement_areas,
                    'timestamp': datetime.now().isoformat(),
                    'accuracy_rating': accuracy_rating,
                    'use_case': use_case
                }
                
                feedback_manager.store_feedback(comprehensive_feedback)
                st.success("🎉 Thank you for your detailed feedback! This helps us improve the tool.")
                st.balloons()
                st.rerun()
    
    else:
        # Traditional analysis feedback (simpler version)
        st.subheader("📝 Analysis Feedback")
        
        with st.expander("💬 How was this analysis?"):
            col1, col2 = st.columns(2)
            
            with col1:
                simple_rating = st.select_slider(
                    "Rate this analysis:",
                    options=["Poor", "Fair", "Good", "Excellent"],
                    value="Good",
                    key="simple_rating"
                )
            
            with col2:
                if st.button("👍 Helpful", key="simple_good"):
                    feedback_manager.store_quick_feedback("overall", content_hash, True, "traditional_analysis")
                    st.success("✅ Thanks!")
                
                if st.button("👎 Not Helpful", key="simple_bad"):
                    feedback_manager.store_quick_feedback("overall", content_hash, False, "traditional_analysis")
                    st.error("📝 We'll improve!")
            
            simple_comments = st.text_area(
                "Comments (optional):",
                placeholder="Any suggestions for improvement?",
                key="simple_comments"
            )
            
            if st.button("Submit Feedback", key="submit_simple"):
                simple_feedback = {
                    'type': 'traditional_analysis',
                    'content_hash': content_hash,
                    'content_preview': content_input[:200] + "...",
                    'rating': simple_rating,
                    'comments': simple_comments,
                    'timestamp': datetime.now().isoformat(),
                    'analysis_mode': 'Traditional'
                }
                
                feedback_manager.store_feedback(simple_feedback)
                st.success("✅ Thank you for your feedback!")
                st.rerun()
