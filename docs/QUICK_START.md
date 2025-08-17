# 🚀 Quick Start Guide - Content Scoring Tool

## ✅ **Working Setup (Tested)**

Your Content Scoring Tool is now fully functional! Here's how to run it:

### **Step 1: Activate Virtual Environment**
```bash
source venv/bin/activate
```

### **Step 2: Run the Tool**
```bash
streamlit run content_scoring_tool_lite.py
```

### **Step 3: Access in Browser**
The tool will automatically open at: `http://localhost:8501` (or the next available port)

## 📁 **File Upload Support**

✅ **Fully Supported Formats:**
- **📄 PDF files** - Text extraction from PDF documents
- **📝 Word documents** - .docx and .doc files  
- **📋 Text files** - Plain .txt files
- **📝 Markdown files** - .md files with formatting removal

## 🎯 **How to Use**

### **Single Content Analysis**
1. Choose "Type/Paste Text" or "Upload File"
2. Add your content or upload a document
3. Optionally add target keywords for SEO analysis
4. Click "🔍 Analyze Content"
5. Review detailed scores and insights

### **Batch Analysis**  
1. Select "Batch Analysis" mode
2. Choose "Upload Files" for multiple documents
3. Upload 2-10 files for comparison
4. Click "🔍 Analyze All Content"
5. Compare results in interactive charts

### **Comparative Analysis**
1. Select "Comparative Analysis" mode  
2. Upload or paste two pieces of content
3. Click "🔍 Compare Content"
4. See side-by-side winner analysis

## 📊 **What You Get**

- **Overall Score** (0-100) with letter grade
- **5 Detailed Metrics:**
  - 📚 Readability (Flesch ease, grade level)
  - 😊 Sentiment (positive/negative/neutral)  
  - 🎯 Engagement (CTAs, questions, power words)
  - 🔍 SEO (keywords, structure, length)
  - ⭐ Quality (AI or rule-based assessment)

- **Interactive Visualizations:**
  - Radar charts showing all metrics
  - Bar charts for comparisons
  - Detailed breakdowns in tabs

- **Export Options:**
  - JSON reports for further analysis
  - Downloadable results

## 🛠️ **If You Encounter Issues**

### **Missing Dependencies**
```bash
# Install additional packages if needed
pip install seaborn wordcloud
```

### **Different Python Environment**
If using conda (base) environment instead:
```bash
python3 -m streamlit run content_scoring_tool_lite.py
```

### **Port Already in Use**
Streamlit will automatically find the next available port (8502, 8503, etc.)

## 🎉 **Ready to Score Content!**

Your tool now supports:
- ✅ Text input and file uploads
- ✅ Multiple file formats  
- ✅ Comprehensive scoring metrics
- ✅ Interactive visualizations
- ✅ Batch and comparative analysis
- ✅ Export capabilities

Perfect for evaluating blog posts, marketing content, documentation, and any text-based materials!
