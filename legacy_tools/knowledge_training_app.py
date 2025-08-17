import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="Knowledge Training System", layout="wide")

st.title("🧠 Knowledge Training System")
st.markdown("*Train the AI with your messaging expertise*")

# Initialize session state for storing examples
if 'messaging_examples' not in st.session_state:
    st.session_state.messaging_examples = []
if 'scoring_criteria' not in st.session_state:
    st.session_state.scoring_criteria = {
        'clarity': {'weight': 25, 'description': 'How clear and understandable is the message?'},
        'differentiation': {'weight': 30, 'description': 'How well does it stand out from competitors?'},
        'emotional_impact': {'weight': 20, 'description': 'Does it connect emotionally with the audience?'},
        'brand_alignment': {'weight': 25, 'description': 'How well does it align with brand values?'}
    }

# Sidebar navigation
st.sidebar.title("📚 Training Modules")
module = st.sidebar.selectbox(
    "Select Training Module:",
    [
        "1. Good vs Bad Examples",
        "2. Industry Templates", 
        "3. Client Success Stories",
        "4. Custom Scoring Criteria",
        "5. Expert Feedback System"
    ]
)

# Module 1: Good vs Bad Examples
if module == "1. Good vs Bad Examples":
    st.header("✅❌ Good vs Bad Messaging Examples")
    
    st.markdown("**Train the system by providing examples of good and bad messaging with your expert scores.**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Add New Example")
        
        with st.form("messaging_example"):
            industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail", "Other"])
            message_type = st.selectbox("Message Type", ["Value Proposition", "Tagline", "Product Description", "About Us", "Customer Testimonial"])
            
            messaging_text = st.text_area("Messaging Text", height=100)
            
            st.markdown("**Expert Scoring (1-10 scale):**")
            clarity_score = st.slider("Clarity", 1, 10, 5)
            differentiation_score = st.slider("Differentiation", 1, 10, 5)
            emotional_impact_score = st.slider("Emotional Impact", 1, 10, 5)
            brand_alignment_score = st.slider("Brand Alignment", 1, 10, 5)
            
            overall_rating = st.selectbox("Overall Rating", ["Excellent", "Good", "Average", "Poor", "Terrible"])
            
            expert_notes = st.text_area("Expert Notes (Why is this good/bad?)", height=80)
            
            submitted = st.form_submit_button("Add Example")
            
            if submitted and messaging_text:
                example = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'industry': industry,
                    'message_type': message_type,
                    'text': messaging_text,
                    'scores': {
                        'clarity': clarity_score,
                        'differentiation': differentiation_score,
                        'emotional_impact': emotional_impact_score,
                        'brand_alignment': brand_alignment_score
                    },
                    'overall_rating': overall_rating,
                    'expert_notes': expert_notes,
                    'overall_score': (clarity_score + differentiation_score + emotional_impact_score + brand_alignment_score) / 4
                }
                st.session_state.messaging_examples.append(example)
                st.success("✅ Example added to training data!")
    
    with col2:
        st.subheader("📊 Training Data Overview")
        
        if st.session_state.messaging_examples:
            df = pd.DataFrame(st.session_state.messaging_examples)
            
            # Show summary stats
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Examples", len(df))
            with col_b:
                st.metric("Avg Score", f"{df['overall_score'].mean():.1f}")
            with col_c:
                good_examples = len(df[df['overall_score'] >= 7])
                st.metric("Good Examples", f"{good_examples}/{len(df)}")
            
            # Show distribution
            fig = px.histogram(df, x='overall_rating', title="Rating Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show recent examples
            st.markdown("**Recent Examples:**")
            for example in st.session_state.messaging_examples[-3:]:
                with st.expander(f"{example['overall_rating']}: {example['text'][:50]}..."):
                    st.write(f"**Industry:** {example['industry']}")
                    st.write(f"**Type:** {example['message_type']}")
                    st.write(f"**Score:** {example['overall_score']:.1f}/10")
                    st.write(f"**Notes:** {example['expert_notes']}")
        else:
            st.info("No examples yet. Add some messaging examples to start training!")

# Module 2: Industry Templates
elif module == "2. Industry Templates":
    st.header("🏭 Industry-Specific Templates")
    
    st.markdown("**Define messaging criteria and templates for different industries.**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Create Industry Template")
        
        with st.form("industry_template"):
            template_industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"])
            
            st.markdown("**Key Messaging Themes (rank importance 1-5):**")
            innovation_importance = st.slider("Innovation", 1, 5, 3)
            trust_importance = st.slider("Trust & Security", 1, 5, 3)
            efficiency_importance = st.slider("Efficiency", 1, 5, 3)
            support_importance = st.slider("Customer Support", 1, 5, 3)
            cost_importance = st.slider("Cost Effectiveness", 1, 5, 3)
            
            tone_preference = st.selectbox("Preferred Tone", ["Professional", "Approachable", "Technical", "Confident", "Empathetic"])
            
            avoid_terms = st.text_area("Terms to Avoid", placeholder="e.g., synergy, paradigm, revolutionary...")
            preferred_terms = st.text_area("Preferred Terms", placeholder="e.g., proven, reliable, innovative...")
            
            template_notes = st.text_area("Industry-Specific Notes")
            
            if st.form_submit_button("Save Template"):
                st.success(f"✅ Template saved for {template_industry}!")
    
    with col2:
        st.subheader("📊 Industry Comparison")
        
        # Mock industry comparison data
        industry_data = pd.DataFrame({
            'Industry': ['Technology', 'Healthcare', 'Finance', 'Manufacturing'],
            'Innovation_Focus': [9, 6, 4, 5],
            'Trust_Focus': [7, 9, 10, 7],
            'Efficiency_Focus': [8, 7, 6, 9],
            'Support_Focus': [6, 8, 7, 6]
        })
        
        fig = px.bar(industry_data, x='Industry', 
                    y=['Innovation_Focus', 'Trust_Focus', 'Efficiency_Focus', 'Support_Focus'],
                    title="Messaging Focus by Industry", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Template Library:**")
        industries = ["Technology", "Healthcare", "Finance", "Manufacturing"]
        for industry in industries:
            with st.expander(f"{industry} Template"):
                st.write("📋 **Key Themes:** Innovation, Security, Scalability")
                st.write("🗣️ **Tone:** Professional yet approachable")
                st.write("❌ **Avoid:** Jargon, overly technical terms")
                st.write("✅ **Prefer:** Clear benefits, proven results")

# Module 3: Client Success Stories
elif module == "3. Client Success Stories":
    st.header("🎯 Client Success Stories")
    
    st.markdown("**Document successful messaging transformations to train the AI on what works.**")
    
    with st.form("success_story"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 Before (Original Messaging)")
            client_name = st.text_input("Client Name (anonymized)", placeholder="e.g., Tech Startup A")
            industry_sector = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"])
            
            before_messaging = st.text_area("Original Messaging", height=150)
            before_problems = st.text_area("Problems Identified", height=100, 
                                         placeholder="e.g., Too technical, unclear value prop...")
        
        with col2:
            st.subheader("📈 After (Improved Messaging)")
            after_messaging = st.text_area("Improved Messaging", height=150)
            improvements_made = st.text_area("Key Improvements", height=100,
                                           placeholder="e.g., Simplified language, clearer benefits...")
        
        st.subheader("📊 Results & Impact")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            engagement_lift = st.number_input("Engagement Lift (%)", 0, 200, 25)
        with col_b:
            clarity_improvement = st.number_input("Clarity Score Improvement", 0, 100, 15)
        with col_c:
            client_satisfaction = st.selectbox("Client Satisfaction", ["Very High", "High", "Medium", "Low"])
        
        lessons_learned = st.text_area("Key Lessons Learned", 
                                     placeholder="What made this transformation successful?")
        
        if st.form_submit_button("Save Success Story"):
            st.success("✅ Success story added to training data!")
            st.balloons()

# Module 4: Custom Scoring Criteria
elif module == "4. Custom Scoring Criteria":
    st.header("⚖️ Custom Scoring Criteria")
    
    st.markdown("**Define how the AI should evaluate messaging quality based on your expertise.**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ Adjust Scoring Weights")
        
        st.markdown("**Current Criteria:**")
        
        total_weight = 0
        new_criteria = {}
        
        for criterion, details in st.session_state.scoring_criteria.items():
            weight = st.slider(
                f"{criterion.replace('_', ' ').title()}", 
                0, 50, 
                details['weight'],
                help=details['description'],
                key=f"weight_{criterion}"
            )
            new_criteria[criterion] = {
                'weight': weight,
                'description': details['description']
            }
            total_weight += weight
        
        if total_weight != 100:
            st.warning(f"⚠️ Weights sum to {total_weight}%, should equal 100%")
        
        if st.button("Update Weights"):
            st.session_state.scoring_criteria = new_criteria
            st.success("✅ Scoring criteria updated!")
    
    with col2:
        st.subheader("➕ Add New Criterion")
        
        with st.form("new_criterion"):
            new_criterion_name = st.text_input("Criterion Name", placeholder="e.g., authenticity")
            new_criterion_weight = st.number_input("Weight (%)", 0, 50, 10)
            new_criterion_desc = st.text_area("Description", 
                                            placeholder="How should this be evaluated?")
            
            if st.form_submit_button("Add Criterion"):
                if new_criterion_name and new_criterion_desc:
                    st.session_state.scoring_criteria[new_criterion_name.lower().replace(' ', '_')] = {
                        'weight': new_criterion_weight,
                        'description': new_criterion_desc
                    }
                    st.success(f"✅ Added '{new_criterion_name}' criterion!")
                    st.rerun()
        
        st.subheader("📊 Current Scoring Formula")
        
        # Visual representation of scoring weights
        if st.session_state.scoring_criteria:
            criteria_df = pd.DataFrame([
                {'Criterion': k.replace('_', ' ').title(), 'Weight': v['weight']}
                for k, v in st.session_state.scoring_criteria.items()
            ])
            
            fig = px.pie(criteria_df, values='Weight', names='Criterion', 
                        title="Scoring Weight Distribution")
            st.plotly_chart(fig, use_container_width=True)

# Module 5: Expert Feedback System
elif module == "5. Expert Feedback System":
    st.header("🎓 Expert Feedback System")
    
    st.markdown("**Continuously improve the AI by providing feedback on its recommendations.**")
    
    # Mock AI recommendations for feedback
    mock_recommendations = [
        {
            'text': "Our AI-powered platform revolutionizes business intelligence with cutting-edge analytics.",
            'ai_score': 6.5,
            'ai_feedback': "Good technical focus but may be too jargon-heavy for broad appeal"
        },
        {
            'text': "We help businesses make smarter decisions with clear, actionable insights.",
            'ai_score': 8.2,
            'ai_feedback': "Clear value proposition with accessible language"
        },
        {
            'text': "Transform your data into competitive advantage through our innovative solutions.",
            'ai_score': 7.1,
            'ai_feedback': "Strong benefit focus but 'innovative solutions' is somewhat generic"
        }
    ]
    
    st.subheader("🤖 Review AI Recommendations")
    
    for i, rec in enumerate(mock_recommendations):
        with st.expander(f"Recommendation {i+1}: AI Score {rec['ai_score']}/10"):
            st.write(f"**Message:** {rec['text']}")
            st.write(f"**AI Analysis:** {rec['ai_feedback']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                expert_score = st.slider(f"Your Score", 1, 10, int(rec['ai_score']), key=f"expert_{i}")
            with col2:
                agreement = st.selectbox("Agree with AI?", ["Agree", "Partially", "Disagree"], key=f"agree_{i}")
            with col3:
                if st.button("Submit Feedback", key=f"submit_{i}"):
                    st.success("✅ Feedback recorded!")
            
            expert_notes = st.text_area("Expert Notes", key=f"notes_{i}",
                                      placeholder="Why do you agree/disagree? What would you change?")
    
    st.subheader("📈 Learning Progress")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Examples Trained", "47", "↑12")
    with col2:
        st.metric("Accuracy Rate", "78%", "↑5%")
    with col3:
        st.metric("Expert Agreement", "82%", "↑3%")
    with col4:
        st.metric("Confidence Level", "High", "↑")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Training Progress")
if st.session_state.messaging_examples:
    progress = min(len(st.session_state.messaging_examples) / 50, 1.0)
    st.sidebar.progress(progress)
    st.sidebar.caption(f"{len(st.session_state.messaging_examples)}/50 examples needed for good training")
else:
    st.sidebar.progress(0)
    st.sidebar.caption("Start adding examples to begin training!")

st.sidebar.markdown("**Benefits of Training:**")
st.sidebar.markdown("• AI learns your quality standards")
st.sidebar.markdown("• Better recommendations over time")
st.sidebar.markdown("• Consistent evaluation criteria")
st.sidebar.markdown("• Reduced manual review time") 