import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Creative Visualization Tool", layout="wide")

st.title("🎨 Creative Visualization Tool")
st.markdown("*Transform messaging into living creative concepts*")

# Mock creative templates and styles
creative_formats = {
    "Billboard": {"width": 800, "height": 300, "bg_color": "#1a1a1a", "text_color": "#ffffff"},
    "Website Hero": {"width": 1200, "height": 400, "bg_color": "#f8f9fa", "text_color": "#333333"},
    "Social Media": {"width": 600, "height": 600, "bg_color": "#4267B2", "text_color": "#ffffff"},
    "Print Ad": {"width": 400, "height": 600, "bg_color": "#ffffff", "text_color": "#000000"}
}

visual_styles = {
    "Modern Tech": {"colors": ["#667eea", "#764ba2"], "mood": "sleek, minimalist"},
    "Corporate Trust": {"colors": ["#2193b0", "#6dd5ed"], "mood": "professional, reliable"},
    "Innovative Edge": {"colors": ["#ee0979", "#ff6a00"], "mood": "bold, cutting-edge"},
    "Human Centered": {"colors": ["#56ab2f", "#a8e6cf"], "mood": "warm, approachable"}
}

# Sidebar navigation
st.sidebar.title("🎯 Creative Process")
step = st.sidebar.selectbox(
    "Select Step:",
    [
        "1. Messaging Routes",
        "2. Visual Style Selection", 
        "3. Creative Generation",
        "4. Format Comparison",
        "5. Client Presentation"
    ]
)

# Step 1: Messaging Routes
if step == "1. Messaging Routes":
    st.header("📝 Generate Messaging Routes")
    
    st.markdown("**Based on your analysis, here are recommended messaging directions:**")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Analysis")
        st.text_area("Original Client Messaging", 
                    value="We provide enterprise software solutions for digital transformation.",
                    height=100)
        
        st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Manufacturing"])
        st.selectbox("Target Audience", ["C-Suite", "IT Decision Makers", "Operations", "End Users"])
        st.multiselect("Key Themes", ["Innovation", "Security", "Efficiency", "Partnership"], 
                      default=["Innovation", "Efficiency"])
    
    with col2:
        st.subheader("Generated Messaging Routes")
        
        messaging_routes = [
            {
                "route": "Innovation Leader",
                "tagline": "Transform Tomorrow, Today",
                "value_prop": "Leading businesses into the future with cutting-edge digital solutions that turn ambitious visions into reality.",
                "tone": "Confident, forward-thinking",
                "score": 8.5
            },
            {
                "route": "Trusted Partner",
                "tagline": "Your Digital Success Partner",
                "value_prop": "We stand beside businesses through every step of their digital journey, delivering reliable results that matter.",
                "tone": "Supportive, professional",
                "score": 7.8
            },
            {
                "route": "Efficiency Expert", 
                "tagline": "Work Smarter, Achieve More",
                "value_prop": "Streamline operations and unlock productivity with intelligent solutions that make complex processes simple.",
                "tone": "Practical, results-focused",
                "score": 8.1
            }
        ]
        
        for i, route in enumerate(messaging_routes):
            with st.expander(f"Route {i+1}: {route['route']} (Score: {route['score']}/10)"):
                st.markdown(f"**Tagline:** _{route['tagline']}_")
                st.write(f"**Value Proposition:** {route['value_prop']}")
                st.write(f"**Tone:** {route['tone']}")
                
                if st.button(f"Select Route {i+1}", key=f"route_{i}"):
                    st.session_state.selected_route = route
                    st.success("✅ Route selected! Move to Visual Style Selection.")

# Step 2: Visual Style Selection
elif step == "2. Visual Style Selection":
    st.header("🎨 Choose Visual Style")
    
    if 'selected_route' not in st.session_state:
        st.warning("⚠️ Please select a messaging route first!")
        st.stop()
    
    st.markdown(f"**Selected Route:** {st.session_state.selected_route['route']}")
    st.markdown(f"**Tagline:** _{st.session_state.selected_route['tagline']}_")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Visual Style Options")
        
        for style_name, style_info in visual_styles.items():
            with st.expander(f"{style_name} - {style_info['mood']}"):
                # Create color preview
                col_a, col_b = st.columns(2)
                with col_a:
                    st.color_picker("Primary", style_info['colors'][0], disabled=True, key=f"color1_{style_name}")
                with col_b:
                    st.color_picker("Secondary", style_info['colors'][1], disabled=True, key=f"color2_{style_name}")
                
                st.write(f"**Mood:** {style_info['mood']}")
                
                if st.button(f"Select {style_name}", key=f"style_{style_name}"):
                    st.session_state.selected_style = style_name
                    st.success(f"✅ {style_name} style selected!")
    
    with col2:
        st.subheader("Style Recommendations")
        
        # Mock AI recommendations based on messaging route
        route = st.session_state.selected_route['route']
        
        if route == "Innovation Leader":
            st.info("🚀 **Recommended:** Innovative Edge - Bold visuals match forward-thinking message")
        elif route == "Trusted Partner":
            st.info("🤝 **Recommended:** Corporate Trust - Professional style builds credibility")
        else:
            st.info("⚡ **Recommended:** Modern Tech - Clean design emphasizes efficiency")
        
        st.markdown("**Style Compatibility Matrix:**")
        compatibility_data = pd.DataFrame({
            'Style': list(visual_styles.keys()),
            'Innovation Leader': [95, 75, 98, 70],
            'Trusted Partner': [70, 95, 60, 85],
            'Efficiency Expert': [90, 80, 75, 75]
        })
        
        fig = px.bar(compatibility_data, x='Style', y=route, 
                    title=f"Style Match for {route}")
        st.plotly_chart(fig, use_container_width=True)

# Step 3: Creative Generation
elif step == "3. Creative Generation":
    st.header("🔮 Generate Creative Concepts")
    
    if 'selected_route' not in st.session_state or 'selected_style' not in st.session_state:
        st.warning("⚠️ Please complete steps 1 and 2 first!")
        st.stop()
    
    route = st.session_state.selected_route
    style = st.session_state.selected_style
    
    st.markdown(f"**Generating visuals for:** {route['route']} in {style} style")
    
    # Mock creative generation process
    if st.button("🎨 Generate Creative Concepts"):
        with st.spinner("Creating visual concepts... (In real implementation, this would call Midjourney API)"):
            import time
            time.sleep(2)  # Simulate generation time
        
        st.success("✅ Creative concepts generated!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Concept A: Bold & Direct")
            # Mock generated image description
            st.markdown("**Visual Description:**")
            st.markdown(f"- Large bold text: '{route['tagline']}'")
            st.markdown(f"- Color scheme: {visual_styles[style]['colors']}")
            st.markdown("- Clean typography with subtle tech elements")
            st.markdown("- Professional business setting background")
            
            # Placeholder for generated image
            st.info("🖼️ Generated Image Would Appear Here")
            st.caption("(Midjourney/DALL-E generated visual)")
            
        with col2:
            st.subheader("Concept B: Lifestyle Focused")
            st.markdown("**Visual Description:**")
            st.markdown(f"- Integrated text: '{route['tagline']}'")
            st.markdown("- People using technology successfully")
            st.markdown("- Bright, optimistic atmosphere")
            st.markdown("- Focus on human benefits")
            
            st.info("🖼️ Generated Image Would Appear Here")
            st.caption("(Midjourney/DALL-E generated visual)")
        
        st.subheader("📊 Concept Performance Prediction")
        
        performance_data = pd.DataFrame({
            'Concept': ['Concept A', 'Concept B'],
            'Attention Score': [85, 78],
            'Brand Recall': [82, 88],
            'Message Clarity': [90, 75],
            'Emotional Impact': [75, 92]
        })
        
        fig = px.bar(performance_data, x='Concept', 
                    y=['Attention Score', 'Brand Recall', 'Message Clarity', 'Emotional Impact'],
                    title="Predicted Performance Metrics", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

# Step 4: Format Comparison
elif step == "4. Format Comparison":
    st.header("📱 Creative Format Comparison")
    
    if 'selected_route' not in st.session_state:
        st.warning("⚠️ Please complete previous steps first!")
        st.stop()
    
    route = st.session_state.selected_route
    
    st.markdown(f"**Showing:** {route['tagline']} across different formats")
    
    # Create mockups for different formats
    def create_mockup(format_name, format_specs, message):
        # Simple mock image creation
        img = Image.new('RGB', (format_specs['width']//2, format_specs['height']//2), 
                       format_specs['bg_color'])
        draw = ImageDraw.Draw(img)
        
        # Add text (simplified - real implementation would be more sophisticated)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        text_bbox = draw.textbbox((0, 0), message, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (img.width - text_width) // 2
        y = (img.height - text_height) // 2
        
        draw.text((x, y), message, fill=format_specs['text_color'], font=font)
        
        return img
    
    # Display formats in grid
    cols = st.columns(2)
    
    for i, (format_name, specs) in enumerate(creative_formats.items()):
        with cols[i % 2]:
            st.subheader(f"{format_name}")
            
            # Create and display mockup
            mockup = create_mockup(format_name, specs, route['tagline'])
            st.image(mockup, caption=f"{format_name} - {specs['width']}x{specs['height']}")
            
            # Format-specific insights
            if format_name == "Billboard":
                st.caption("✅ High impact, limited text works well")
            elif format_name == "Website Hero":
                st.caption("✅ Perfect for detailed value proposition")
            elif format_name == "Social Media":
                st.caption("✅ Great for engagement and sharing")
            else:
                st.caption("✅ Professional, detailed presentation")
    
    st.subheader("📊 Format Effectiveness Analysis")
    
    effectiveness_data = pd.DataFrame({
        'Format': list(creative_formats.keys()),
        'Reach': [95, 70, 90, 60],
        'Engagement': [60, 85, 95, 70],
        'Conversion': [70, 90, 75, 85],
        'Cost_Efficiency': [40, 95, 85, 60]
    })
    
    # Create line chart instead of radar chart
    fig = go.Figure()
    
    metrics = ['Reach', 'Engagement', 'Conversion', 'Cost_Efficiency']
    formats = effectiveness_data['Format'].tolist()
    
    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=formats,
            y=effectiveness_data[metric],
            mode='lines+markers',
            name=metric,
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title="Format Performance Across Key Metrics",
        xaxis_title="Format",
        yaxis_title="Performance Score",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Step 5: Client Presentation
elif step == "5. Client Presentation":
    st.header("👥 Client Presentation Mode")
    
    if 'selected_route' not in st.session_state:
        st.warning("⚠️ Please complete the creative process first!")
        st.stop()
    
    route = st.session_state.selected_route
    style = st.session_state.get('selected_style', 'Modern Tech')
    
    # Clean presentation view
    st.markdown("---")
    
    # Executive Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Messaging Score", "8.5/10", "↑2.3")
    with col2:
        st.metric("Visual Impact", "High", "↑Strong")
    with col3:
        st.metric("Brand Alignment", "94%", "↑12%")
    
    st.markdown("---")
    
    # Key Recommendation
    st.markdown("## 🎯 Recommended Creative Direction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {route['route']}")
        st.markdown(f"**Tagline:** _{route['tagline']}_")
        st.markdown(f"**Value Proposition:** {route['value_prop']}")
        st.markdown(f"**Visual Style:** {style}")
        st.markdown(f"**Tone:** {route['tone']}")
    
    with col2:
        st.markdown("### Why This Works")
        st.markdown("✅ Differentiates from competitors")
        st.markdown("✅ Resonates with target audience") 
        st.markdown("✅ Aligns with brand values")
        st.markdown("✅ Scalable across channels")
    
    st.markdown("---")
    
    # Creative Applications
    st.markdown("## 🎨 Creative Applications")
    
    application_examples = [
        {"channel": "Digital Advertising", "impact": "High", "timeline": "2 weeks"},
        {"channel": "Website Redesign", "impact": "Very High", "timeline": "4 weeks"},
        {"channel": "Trade Show Materials", "impact": "Medium", "timeline": "3 weeks"},
        {"channel": "Social Media Campaign", "impact": "High", "timeline": "1 week"}
    ]
    
    df = pd.DataFrame(application_examples)
    st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # Next Steps
    st.markdown("## 🚀 Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Immediate Actions")
        st.markdown("1. **Approve creative direction**")
        st.markdown("2. **Finalize visual assets**")
        st.markdown("3. **Plan rollout timeline**")
        st.markdown("4. **Set success metrics**")
    
    with col2:
        st.markdown("### Timeline & Budget")
        st.markdown("**Phase 1:** Creative development (2 weeks)")
        st.markdown("**Phase 2:** Asset production (3 weeks)")
        st.markdown("**Phase 3:** Campaign launch (1 week)")
        st.markdown("**Total Investment:** Competitive with traditional approach")
    
    # Export options
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Presentation"):
            st.success("✅ Presentation exported!")
    
    with col2:
        if st.button("🎨 Download Assets"):
            st.success("✅ Creative assets prepared!")
    
    with col3:
        if st.button("📊 Get Report"):
            st.success("✅ Full report generated!")

# Sidebar progress tracking
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Process Progress")

progress_steps = [
    ("Messaging Routes", "selected_route" in st.session_state),
    ("Visual Style", "selected_style" in st.session_state),
    ("Creative Generation", False),  # Would track actual generation
    ("Format Comparison", False),
    ("Client Presentation", False)
]

for step_name, completed in progress_steps:
    if completed:
        st.sidebar.markdown(f"✅ {step_name}")
    else:
        st.sidebar.markdown(f"⏳ {step_name}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔮 What This Demonstrates")
st.sidebar.markdown("• Transform messaging into visuals")
st.sidebar.markdown("• Multiple creative directions")
st.sidebar.markdown("• Cross-format consistency")
st.sidebar.markdown("• Client-ready presentations")
st.sidebar.markdown("• Data-driven creative decisions") 