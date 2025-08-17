import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import PyPDF2
import requests
from bs4 import BeautifulSoup
import io

st.set_page_config(page_title="Data Management Hub", layout="wide")

st.title("📊 Data Management Hub")
st.markdown("*Central interface for managing all your messaging analysis data*")

# Initialize session state for data storage
if 'training_data' not in st.session_state:
    st.session_state.training_data = []
if 'competitor_data' not in st.session_state:
    st.session_state.competitor_data = []
if 'client_projects' not in st.session_state:
    st.session_state.client_projects = []

# Sidebar navigation
st.sidebar.title("🗂️ Data Categories")
data_category = st.sidebar.selectbox(
    "Select Data Type:",
    [
        "📝 Training Examples",
        "🏢 Client Projects", 
        "🎯 Competitor Analysis",
        "📊 Analytics Dashboard",
        "⚙️ System Settings"
    ]
)

# Helper functions
def save_data_to_json(data, filename):
    """Save data to JSON file for persistence"""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def load_data_from_json(filename):
    """Load data from JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_url(url):
    """Extract text from website URL"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text[:5000]  # Limit to first 5000 characters
    except Exception as e:
        st.error(f"Error extracting from URL: {e}")
        return ""

# Training Examples Section
if data_category == "📝 Training Examples":
    st.header("📝 Expert Training Examples")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Add New Example")
        
        with st.form("training_example"):
            # Basic information
            industry = st.selectbox("Industry", 
                                  ["Manufacturing", "Technology", "Professional Services", 
                                   "Healthcare", "Finance", "Other"])
            
            message_type = st.selectbox("Message Type",
                                      ["Value Proposition", "Tagline", "Website Copy", 
                                       "Sales Collateral", "Social Media", "Other"])
            
            messaging_text = st.text_area("Messaging Text", height=100,
                                        placeholder="Enter the messaging example...")
            
            # Expert scoring
            st.markdown("**Expert Scoring (1-10):**")
            clarity_score = st.slider("Clarity", 1, 10, 5)
            differentiation_score = st.slider("Differentiation", 1, 10, 5)
            brand_fit_score = st.slider("Brand Fit", 1, 10, 5)
            b2b_effectiveness = st.slider("B2B Effectiveness", 1, 10, 5)
            
            # Context and notes
            client_context = st.text_area("Client Context", height=50,
                                        placeholder="Mid-size manufacturer, cost-focused...")
            
            expert_notes = st.text_area("Expert Notes", height=50,
                                      placeholder="Why this works/doesn't work...")
            
            success_metrics = st.text_input("Success Metrics",
                                          placeholder="Generated 23 qualified leads...")
            
            competitive_gap = st.text_area("Competitive Gap Analysis", height=50,
                                         placeholder="Most competitors focus on features...")
            
            if st.form_submit_button("📋 Add Training Example"):
                if messaging_text:
                    new_example = {
                        "id": len(st.session_state.training_data) + 1,
                        "timestamp": datetime.now().isoformat(),
                        "industry": industry,
                        "message_type": message_type,
                        "messaging_text": messaging_text,
                        "scores": {
                            "clarity": clarity_score,
                            "differentiation": differentiation_score,
                            "brand_fit": brand_fit_score,
                            "b2b_effectiveness": b2b_effectiveness,
                            "overall": round((clarity_score + differentiation_score + 
                                           brand_fit_score + b2b_effectiveness) / 4, 1)
                        },
                        "client_context": client_context,
                        "expert_notes": expert_notes,
                        "success_metrics": success_metrics,
                        "competitive_gap": competitive_gap
                    }
                    
                    st.session_state.training_data.append(new_example)
                    save_data_to_json(st.session_state.training_data, "training_data.json")
                    st.success("✅ Training example added!")
                    st.experimental_rerun()
    
    with col2:
        st.subheader("Training Data Overview")
        
        if st.session_state.training_data:
            # Summary stats
            df = pd.DataFrame([{
                "Industry": item["industry"],
                "Type": item["message_type"],
                "Overall Score": item["scores"]["overall"],
                "Clarity": item["scores"]["clarity"],
                "Differentiation": item["scores"]["differentiation"]
            } for item in st.session_state.training_data])
            
            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Examples", len(st.session_state.training_data))
            with col_b:
                st.metric("Avg Score", f"{df['Overall Score'].mean():.1f}/10")
            with col_c:
                st.metric("Industries", df["Industry"].nunique())
            
            # Recent examples
            st.markdown("**Recent Examples:**")
            for item in st.session_state.training_data[-5:]:
                with st.expander(f"{item['industry']} - Score: {item['scores']['overall']}/10"):
                    st.write(f"**Text:** {item['messaging_text'][:200]}...")
                    st.write(f"**Expert Notes:** {item['expert_notes']}")
                    if st.button(f"Delete", key=f"del_train_{item['id']}"):
                        st.session_state.training_data = [x for x in st.session_state.training_data if x['id'] != item['id']]
                        st.experimental_rerun()
        else:
            st.info("No training examples yet. Add your first example to get started!")

# Client Projects Section
elif data_category == "🏢 Client Projects":
    st.header("🏢 Client Project Database")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Add New Project")
        
        with st.form("client_project"):
            client_name = st.text_input("Client Name")
            project_type = st.selectbox("Project Type",
                                      ["Messaging Framework", "Brand Refresh", "Product Launch",
                                       "Crisis Communication", "Market Entry", "Other"])
            
            industry = st.selectbox("Industry", 
                                  ["Manufacturing", "Technology", "Professional Services", 
                                   "Healthcare", "Finance", "Other"])
            
            # File uploads
            st.markdown("**Project Materials:**")
            interview_files = st.file_uploader("Interview Transcripts", 
                                             accept_multiple_files=True, 
                                             type=['txt', 'pdf'])
            
            current_messaging = st.text_area("Current Messaging", height=100)
            competitor_urls = st.text_area("Competitor URLs (one per line)", height=100)
            
            # Project details
            project_goals = st.text_area("Project Goals", height=80)
            target_audience = st.text_input("Target Audience")
            budget_range = st.selectbox("Budget Range", 
                                      ["<£10k", "£10k-£25k", "£25k-£50k", "£50k+"])
            
            if st.form_submit_button("📁 Add Client Project"):
                if client_name:
                    # Process uploaded files
                    processed_files = []
                    for file in interview_files:
                        if file.name.endswith('.pdf'):
                            text = extract_text_from_pdf(file)
                        else:
                            text = file.read().decode("utf-8")
                        
                        processed_files.append({
                            "filename": file.name,
                            "content": text[:2000]  # First 2000 chars
                        })
                    
                    # Process competitor URLs
                    competitor_data = []
                    if competitor_urls:
                        for url in competitor_urls.split('\n'):
                            url = url.strip()
                            if url:
                                text = extract_text_from_url(url)
                                competitor_data.append({
                                    "url": url,
                                    "content": text
                                })
                    
                    new_project = {
                        "id": len(st.session_state.client_projects) + 1,
                        "timestamp": datetime.now().isoformat(),
                        "client_name": client_name,
                        "project_type": project_type,
                        "industry": industry,
                        "current_messaging": current_messaging,
                        "project_goals": project_goals,
                        "target_audience": target_audience,
                        "budget_range": budget_range,
                        "interview_files": processed_files,
                        "competitor_data": competitor_data,
                        "status": "Active"
                    }
                    
                    st.session_state.client_projects.append(new_project)
                    save_data_to_json(st.session_state.client_projects, "client_projects.json")
                    st.success("✅ Client project added!")
                    st.experimental_rerun()
    
    with col2:
        st.subheader("Active Projects")
        
        if st.session_state.client_projects:
            for project in st.session_state.client_projects[-10:]:
                with st.expander(f"{project['client_name']} - {project['project_type']}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Industry:** {project['industry']}")
                        st.write(f"**Status:** {project['status']}")
                        st.write(f"**Budget:** {project['budget_range']}")
                    with col_b:
                        st.write(f"**Files:** {len(project['interview_files'])}")
                        st.write(f"**Competitors:** {len(project['competitor_data'])}")
                        st.write(f"**Added:** {project['timestamp'][:10]}")
                    
                    if st.button(f"🚀 Analyze Project", key=f"analyze_{project['id']}"):
                        st.info("Would launch analysis in main app...")
        else:
            st.info("No client projects yet. Add your first project to get started!")

# Competitor Analysis Section
elif data_category == "🎯 Competitor Analysis":
    st.header("🎯 Competitor Intelligence Database")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Add Competitor Data")
        
        with st.form("competitor_data"):
            competitor_name = st.text_input("Competitor Name")
            industry = st.selectbox("Industry", 
                                  ["Manufacturing", "Technology", "Professional Services", 
                                   "Healthcare", "Finance", "Other"])
            
            data_source = st.selectbox("Data Source",
                                     ["Website", "Marketing Materials", "Social Media", 
                                      "Press Release", "Other"])
            
            source_url = st.text_input("Source URL (optional)")
            
            # Content input
            content_text = st.text_area("Competitor Content", height=150,
                                      placeholder="Paste competitor messaging, copy, etc...")
            
            # Analysis
            message_type = st.selectbox("Message Type",
                                      ["Value Proposition", "Tagline", "Product Description",
                                       "About Us", "Homepage Copy", "Other"])
            
            # Quick scoring
            differentiation_level = st.selectbox("Differentiation Level",
                                                ["Generic", "Somewhat Unique", "Highly Differentiated"])
            
            tone_analysis = st.multiselect("Tone Characteristics",
                                         ["Professional", "Friendly", "Technical", "Corporate",
                                          "Innovative", "Trustworthy", "Aggressive", "Conservative"])
            
            competitive_notes = st.text_area("Competitive Analysis Notes", height=80,
                                           placeholder="Key themes, gaps, opportunities...")
            
            if st.form_submit_button("🎯 Add Competitor Data"):
                if competitor_name and content_text:
                    new_competitor = {
                        "id": len(st.session_state.competitor_data) + 1,
                        "timestamp": datetime.now().isoformat(),
                        "competitor_name": competitor_name,
                        "industry": industry,
                        "data_source": data_source,
                        "source_url": source_url,
                        "content_text": content_text,
                        "message_type": message_type,
                        "differentiation_level": differentiation_level,
                        "tone_analysis": tone_analysis,
                        "competitive_notes": competitive_notes
                    }
                    
                    st.session_state.competitor_data.append(new_competitor)
                    save_data_to_json(st.session_state.competitor_data, "competitor_data.json")
                    st.success("✅ Competitor data added!")
                    st.experimental_rerun()
    
    with col2:
        st.subheader("Competitor Intelligence")
        
        if st.session_state.competitor_data:
            # Summary by industry
            df = pd.DataFrame([{
                "Competitor": item["competitor_name"],
                "Industry": item["industry"],
                "Differentiation": item["differentiation_level"],
                "Source": item["data_source"]
            } for item in st.session_state.competitor_data])
            
            st.markdown("**Industry Breakdown:**")
            industry_counts = df["Industry"].value_counts()
            st.bar_chart(industry_counts)
            
            st.markdown("**Recent Competitor Analysis:**")
            for item in st.session_state.competitor_data[-5:]:
                with st.expander(f"{item['competitor_name']} - {item['differentiation_level']}"):
                    st.write(f"**Content:** {item['content_text'][:200]}...")
                    st.write(f"**Tone:** {', '.join(item['tone_analysis'])}")
                    st.write(f"**Notes:** {item['competitive_notes']}")
        else:
            st.info("No competitor data yet. Add competitor analysis to get started!")

# Analytics Dashboard
elif data_category == "📊 Analytics Dashboard":
    st.header("📊 Data Analytics Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Examples", len(st.session_state.training_data))
        st.metric("Client Projects", len(st.session_state.client_projects))
        st.metric("Competitor Records", len(st.session_state.competitor_data))
    
    with col2:
        if st.session_state.training_data:
            avg_score = np.mean([item["scores"]["overall"] for item in st.session_state.training_data])
            st.metric("Avg Training Score", f"{avg_score:.1f}/10")
            
            industries = set(item["industry"] for item in st.session_state.training_data)
            st.metric("Industries Covered", len(industries))
    
    with col3:
        if st.session_state.client_projects:
            active_projects = len([p for p in st.session_state.client_projects if p.get("status") == "Active"])
            st.metric("Active Projects", active_projects)
    
    # Data export section
    st.markdown("---")
    st.subheader("🔄 Data Export & Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Data:**")
        if st.button("📥 Export All Data"):
            all_data = {
                "training_data": st.session_state.training_data,
                "client_projects": st.session_state.client_projects,
                "competitor_data": st.session_state.competitor_data,
                "export_timestamp": datetime.now().isoformat()
            }
            
            json_string = json.dumps(all_data, indent=2, default=str)
            st.download_button(
                label="💾 Download JSON",
                data=json_string,
                file_name=f"messaging_data_export_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        st.markdown("**Import Data:**")
        uploaded_file = st.file_uploader("Upload JSON Data", type=['json'])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                if st.button("📤 Import Data"):
                    if "training_data" in data:
                        st.session_state.training_data.extend(data["training_data"])
                    if "client_projects" in data:
                        st.session_state.client_projects.extend(data["client_projects"])
                    if "competitor_data" in data:
                        st.session_state.competitor_data.extend(data["competitor_data"])
                    st.success("✅ Data imported successfully!")
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error importing data: {e}")

# System Settings
elif data_category == "⚙️ System Settings":
    st.header("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Management")
        
        if st.button("🗑️ Clear All Training Data"):
            if st.checkbox("I understand this will delete all training data"):
                st.session_state.training_data = []
                save_data_to_json([], "training_data.json")
                st.success("Training data cleared!")
        
        if st.button("🗑️ Clear All Client Projects"):
            if st.checkbox("I understand this will delete all client projects"):
                st.session_state.client_projects = []
                save_data_to_json([], "client_projects.json")
                st.success("Client projects cleared!")
    
    with col2:
        st.subheader("Integration Settings")
        
        st.markdown("**API Configurations:**")
        openai_key = st.text_input("OpenAI API Key", type="password",
                                 help="For advanced AI analysis")
        
        github_token = st.text_input("GitHub Token", type="password",
                                   help="For version control integration")
        
        if st.button("💾 Save Settings"):
            # In a real app, you'd save these securely
            st.success("Settings saved!")

# Load existing data on startup
if st.sidebar.button("🔄 Reload Data"):
    st.session_state.training_data = load_data_from_json("training_data.json")
    st.session_state.client_projects = load_data_from_json("client_projects.json") 
    st.session_state.competitor_data = load_data_from_json("competitor_data.json")
    st.success("Data reloaded!")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Quick Stats")
st.sidebar.metric("Training Examples", len(st.session_state.training_data))
st.sidebar.metric("Client Projects", len(st.session_state.client_projects))
st.sidebar.metric("Competitor Records", len(st.session_state.competitor_data))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 Quick Actions")
if st.sidebar.button("🚀 Analyze with Main App"):
    st.sidebar.info("Would redirect to main analysis app with selected data")

if st.sidebar.button("🎨 Generate Creative"):
    st.sidebar.info("Would redirect to creative visualization with project data")

if st.sidebar.button("🧠 Train AI Model"):
    st.sidebar.info("Would start training process with accumulated data") 