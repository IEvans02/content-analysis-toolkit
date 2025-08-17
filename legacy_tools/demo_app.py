import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

st.set_page_config(page_title="Client Messaging & Competitor Insight Tool - Demo", layout="wide")

st.title("🎯 Client Messaging & Competitor Insight Tool - Demo")
st.markdown("*Interactive demonstration of all features*")

# Mock data for demonstration
brands = ["Your Client", "Competitor A", "Competitor B", "Competitor C"]
themes = ["Innovation", "Customer Success", "Security", "Scalability", "Partnership", "Efficiency"]

# Generate mock sentiment and overlap data
np.random.seed(42)
sentiment_data = pd.DataFrame({
    'Brand': brands * len(themes),
    'Theme': themes * len(brands),
    'Sentiment_Score': np.random.uniform(0.2, 0.9, len(brands) * len(themes)),
    'Frequency': np.random.randint(5, 50, len(brands) * len(themes))
})

# Mock quality metrics
quality_metrics = pd.DataFrame({
    'Brand': brands,
    'Flesch_Kincaid_Grade': [11.2, 9.8, 12.5, 10.1],
    'Clarity_Score': [72, 68, 58, 75],
    'Jargon_Density': [15, 22, 28, 18],
    'Emotional_Tone': ['Confident', 'Approachable', 'Technical', 'Professional'],
    'Differentiation_Score': [85, 62, 45, 70]
})

# Sidebar for feature selection
st.sidebar.title("🔧 Demo Features")
feature = st.sidebar.selectbox(
    "Select Feature to Demo:",
    [
        "1. Advanced Theme Analysis",
        "2. Competitive Intelligence Dashboard", 
        "3. Content Quality Metrics",
        "4. Strategic Recommendations Engine",
        "5. Brand Positioning Map",
        "6. Export & Reporting Suite"
    ]
)

# Feature 1: Advanced Theme Analysis
if feature == "1. Advanced Theme Analysis":
    st.header("🧠 Advanced Theme Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Theme Sentiment Analysis")
        fig_sentiment = px.bar(
            sentiment_data[sentiment_data['Brand'] == 'Your Client'], 
            x='Theme', 
            y='Sentiment_Score',
            title="Your Client - Theme Sentiment Scores",
            color='Sentiment_Score',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        st.subheader("Theme Frequency Comparison")
        fig_freq = px.bar(
            sentiment_data, 
            x='Theme', 
            y='Frequency',
            color='Brand',
            title="Theme Frequency Across All Brands",
            barmode='group'
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    
    st.subheader("Theme Gap Analysis")
    gap_analysis = pd.DataFrame({
        'Theme': themes,
        'Your_Client': [45, 32, 28, 41, 38, 35],
        'Market_Average': [38, 42, 35, 38, 45, 32],
        'Gap': [7, -10, -7, 3, -7, 3]
    })
    
    st.dataframe(gap_analysis.style.format({'Gap': '{:+.0f}'}))
    st.caption("🔍 **Insight**: Your client under-utilizes 'Customer Success' and 'Partnership' themes compared to competitors")

# Feature 2: Competitive Intelligence Dashboard
elif feature == "2. Competitive Intelligence Dashboard":
    st.header("📊 Competitive Intelligence Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Message Overlap Heatmap")
        overlap_matrix = np.random.uniform(0.1, 0.8, (len(brands), len(brands)))
        np.fill_diagonal(overlap_matrix, 1.0)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(overlap_matrix, annot=True, xticklabels=brands, yticklabels=brands, 
                   cmap='RdYlBu_r', ax=ax, cbar_kws={'label': 'Overlap %'})
        ax.set_title('Message Overlap Between Brands')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Unique Value Propositions")
        uvp_data = {
            'Your Client': ['AI-powered insights', 'Real-time analytics'],
            'Competitor A': ['24/7 support', 'Enterprise security'],
            'Competitor B': ['Cost-effective solutions', 'Rapid deployment'],
            'Competitor C': ['Industry expertise', 'Custom integrations']
        }
        
        for brand, uvps in uvp_data.items():
            st.write(f"**{brand}:**")
            for uvp in uvps:
                st.write(f"  • {uvp}")
    
    st.subheader("Competitive Landscape Overview")
    metrics_df = pd.DataFrame({
        'Brand': brands,
        'Market_Share': [25, 35, 20, 20],
        'Message_Strength': [85, 72, 58, 68],
        'Differentiation': [92, 65, 45, 71]
    })
    
    fig = px.scatter(metrics_df, x='Market_Share', y='Message_Strength', 
                    size='Differentiation', hover_name='Brand',
                    title="Competitive Position Analysis",
                    labels={'Market_Share': 'Market Share (%)', 
                           'Message_Strength': 'Message Strength Score'})
    st.plotly_chart(fig, use_container_width=True)

# Feature 3: Content Quality Metrics
elif feature == "3. Content Quality Metrics":
    st.header("📝 Content Quality Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Readability & Clarity Scores")
        fig = px.bar(quality_metrics, x='Brand', y=['Flesch_Kincaid_Grade', 'Clarity_Score'],
                    title="Readability Metrics by Brand", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Content Quality Scorecard")
        st.dataframe(quality_metrics.set_index('Brand'))
    
    with col2:
        st.subheader("Jargon Density Analysis")
        fig = px.pie(quality_metrics, values='Jargon_Density', names='Brand',
                    title="Jargon Density Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Emotional Tone Distribution")
        tone_counts = quality_metrics['Emotional_Tone'].value_counts()
        fig = px.bar(x=tone_counts.index, y=tone_counts.values,
                    title="Emotional Tone Categories")
        st.plotly_chart(fig, use_container_width=True)

# Feature 4: Strategic Recommendations Engine
elif feature == "4. Strategic Recommendations Engine":
    st.header("🎯 Strategic Recommendations Engine")
    
    st.subheader("📈 Automated Insights & Suggestions")
    
    recommendations = [
        {
            "Priority": "High",
            "Category": "Theme Gap",
            "Recommendation": "Increase 'Customer Success' messaging by 25% to match market average",
            "Impact": "Higher trust and relationship focus",
            "Effort": "Medium"
        },
        {
            "Priority": "Medium", 
            "Category": "Clarity",
            "Recommendation": "Reduce jargon density from 15% to 10% for broader appeal",
            "Impact": "Improved accessibility and engagement",
            "Effort": "Low"
        },
        {
            "Priority": "High",
            "Category": "Differentiation",
            "Recommendation": "Amplify 'AI-powered insights' unique value proposition",
            "Impact": "Stronger competitive positioning",
            "Effort": "Low"
        }
    ]
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"🔸 {rec['Priority']} Priority: {rec['Recommendation']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Category", rec['Category'])
            with col2:
                st.metric("Expected Impact", rec['Impact'])
            with col3:
                st.metric("Implementation Effort", rec['Effort'])
    
    st.subheader("💡 Whitespace Opportunities")
    opportunities = pd.DataFrame({
        'Opportunity': ['Sustainability Focus', 'Community Building', 'Thought Leadership'],
        'Market_Gap': ['Low competitor focus', 'Emerging trend', 'Underutilized channel'],
        'Potential_Impact': ['High', 'Medium', 'High'],
        'Competitive_Advantage': ['First-mover', 'Authenticity', 'Expertise']
    })
    st.dataframe(opportunities)

# Feature 5: Brand Positioning Map
elif feature == "5. Brand Positioning Map":
    st.header("🗺️ Brand Positioning Map")
    
    # Interactive positioning map
    positioning_data = pd.DataFrame({
        'Brand': brands,
        'Innovation_Focus': [8.5, 6.2, 4.1, 7.0],
        'Enterprise_Focus': [7.8, 8.9, 5.2, 8.1],
        'Size': [100, 120, 80, 90]
    })
    
    fig = px.scatter(positioning_data, x='Innovation_Focus', y='Enterprise_Focus',
                    size='Size', hover_name='Brand', 
                    title="Brand Positioning: Innovation vs Enterprise Focus",
                    labels={'Innovation_Focus': 'Innovation Focus →',
                           'Enterprise_Focus': 'Enterprise Focus →'})
    
    # Add quadrant labels
    fig.add_annotation(x=2, y=8.5, text="Traditional<br>Enterprise", showarrow=False, font_size=12)
    fig.add_annotation(x=8.5, y=8.5, text="Innovative<br>Enterprise", showarrow=False, font_size=12)
    fig.add_annotation(x=2, y=2, text="Traditional<br>SMB", showarrow=False, font_size=12)
    fig.add_annotation(x=8.5, y=2, text="Innovative<br>SMB", showarrow=False, font_size=12)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Strategic Positioning Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ **Your Client** occupies the premium 'Innovative Enterprise' quadrant")
        st.info("ℹ️ Strong differentiation from **Competitor B** (Traditional SMB)")
    with col2:
        st.warning("⚠️ Direct competition with **Competitor C** in similar space")
        st.info("💡 Opportunity to further differentiate on innovation axis")

# Feature 6: Export & Reporting Suite
elif feature == "6. Export & Reporting Suite":
    st.header("📋 Export & Reporting Suite")
    
    st.subheader("Available Export Formats")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Executive Summary")
        if st.button("Generate PDF Report", key="pdf"):
            st.success("✅ Executive PDF generated!")
            st.markdown("*Would contain: Key insights, positioning map, recommendations*")
    
    with col2:
        st.markdown("### 📈 Data Export")
        if st.button("Download Excel Data", key="excel"):
            st.success("✅ Excel file prepared!")
            st.markdown("*Would contain: All metrics, theme analysis, competitor data*")
    
    with col3:
        st.markdown("### 🎯 Client Presentation")
        if st.button("PowerPoint Slides", key="ppt"):
            st.success("✅ Presentation slides ready!")
            st.markdown("*Would contain: Visual insights, strategic recommendations*")
    
    st.subheader("📋 Report Preview")
    
    # Mock report content
    st.markdown("---")
    st.markdown("## Executive Summary: Brand Messaging Analysis")
    st.markdown("**Analysis Date:** December 2024")
    st.markdown("**Brands Analyzed:** Your Client vs 3 Key Competitors")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", "85/100", "↑12")
    with col2:
        st.metric("Differentiation", "92%", "↑8%") 
    with col3:
        st.metric("Clarity Score", "72%", "→0%")
    with col4:
        st.metric("Theme Coverage", "6/8", "↑2")
    
    st.markdown("### 🎯 Key Recommendations")
    st.markdown("""
    1. **Amplify Customer Success messaging** to match market expectations
    2. **Reduce technical jargon** by 5% to improve accessibility  
    3. **Leverage AI-powered insights** as primary differentiator
    """)
    
    st.markdown("### 📊 Competitive Landscape")
    st.markdown("Your client leads in innovation positioning but has opportunities in partnership messaging.")

# Demo conclusion
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Ready to Build?")
st.sidebar.markdown("This demo showcases the full potential of your messaging analysis tool.")
st.sidebar.markdown("**Next Steps:**")
st.sidebar.markdown("• Finalize feature prioritization")
st.sidebar.markdown("• Begin development sprint")
st.sidebar.markdown("• Integrate with real data sources") 