# 🔬 Unified Content Analysis Tool

A comprehensive tool that combines **content scoring** with **text summarization** capabilities, integrating the best of both worlds for complete content analysis.

## 🚀 **What's New - Unified Features**

### **🔬 Four Analysis Modes:**

#### **1. Content Scoring** 
- **📊 Comprehensive metrics**: Readability, sentiment, engagement, SEO
- **🎯 Actionable insights**: Grade scoring with detailed breakdowns  
- **📈 Visual analysis**: Interactive radar charts and score visualizations

#### **2. Text Summarization**
- **📝 AI-powered summaries**: Using BART large CNN model
- **🎯 Theme extraction**: Identify main topics and themes
- **☁️ Word clouds**: Visual representation of content
- **📊 Text statistics**: Word count, sentence analysis

#### **3. Combined Analysis** 
- **🔬 Full spectrum analysis**: Both scoring and summarization in one view
- **📋 Tabbed interface**: Organized results across multiple views
- **📈 Rich visualizations**: Radar charts, word clouds, and statistics
- **💡 Complete insights**: Everything you need in one place

#### **4. Document Comparison**
- **⚖️ Multi-document analysis**: Compare 2-5 documents simultaneously  
- **📊 Comparative scoring**: Side-by-side metric comparisons
- **📝 Summary comparison**: Individual and comparative summaries
- **🏆 Performance ranking**: Identify best and worst performers

## 📁 **Enhanced File Support**

- **📄 PDF documents** - Full text extraction
- **📝 Word files** - .docx and .doc support  
- **📋 Text files** - Plain .txt files
- **📝 Markdown** - .md files with formatting removal
- **🔄 Batch uploads** - Multiple file processing
- **👁️ Content preview** - Verify extracted text

## 🎯 **Key Features**

### **Content Scoring Metrics:**
- **📚 Readability**: Flesch ease, grade level, ARI scores
- **😊 Sentiment**: AI-powered positive/negative/neutral analysis
- **🎯 Engagement**: CTAs, questions, power words, personal pronouns  
- **🔍 SEO**: Keyword density, structure, length optimization

### **Summarization Capabilities:**
- **📄 Text summarization**: Configurable length (short/medium/long)
- **🎯 Theme extraction**: Main topics and key themes
- **⚖️ Document comparison**: Find commonalities and differences
- **📊 Content statistics**: Comprehensive text analysis

### **Advanced Features:**
- **🤖 Model flexibility**: Graceful degradation when models unavailable
- **🔧 Configurable settings**: Adjustable analysis parameters and keywords
- **📈 Interactive visualizations**: Plotly charts and matplotlib graphics
- **💾 Export capabilities**: JSON results for further analysis

## 🛠️ **Setup & Installation**

### **Quick Start:**
```bash
# Activate environment
source venv/bin/activate

# Install dependencies  
pip install -r requirements_unified.txt

# Run the unified tool
streamlit run unified_content_analysis_tool.py
```

### **For Conda Users:**
```bash
python3 -m pip install -r requirements_unified.txt
python3 -m streamlit run unified_content_analysis_tool.py
```

## 📊 **How to Use**

### **Content Scoring Mode:**
1. Select "Content Scoring" from sidebar
2. Choose text input or file upload
3. Add target keywords (optional)
4. Click "🔍 Analyze Content"
5. Review comprehensive scoring and insights

### **Text Summarization Mode:**
1. Select "Text Summarization" from sidebar  
2. Choose summary length preference
3. Input text or upload document
4. Click "📝 Generate Summary"
5. Review summary, themes, and word cloud

### **Combined Analysis Mode:**
1. Select "Combined Analysis" from sidebar
2. Configure both scoring and summary settings
3. Input content via text or file upload
4. Click "🔬 Full Analysis"  
5. Explore results across three tabs:
   - 📊 Content Scores
   - 📝 Summary & Themes  
   - 📈 Visualizations

### **Document Comparison Mode:**
1. Select "Document Comparison" from sidebar
2. Set number of documents (2-5)
3. Upload or paste content for each document
4. Click "⚖️ Compare Documents"
5. Review comparative analysis and rankings

## 🎯 **Integration Benefits**

### **Unified Workflow:**
- **Single tool** for all content analysis needs
- **Consistent interface** across all analysis types
- **Shared file processing** for all modes
- **Integrated visualizations** with consistent styling

### **Enhanced Insights:**
- **Content quality scoring** combined with **thematic analysis**
- **Readability metrics** alongside **summary generation**
- **Engagement analysis** with **comparative summaries**
- **Complete content picture** in one analysis

### **Improved Efficiency:**
- **One model loading** session for all features
- **Shared configuration** across analysis modes
- **Unified file handling** for all document types
- **Streamlined user experience**

## 🔧 **Technical Architecture**

### **Model Integration:**
- **Sentiment Analysis**: DistilBERT fine-tuned model
- **Text Summarization**: Facebook BART-large-CNN
- **Graceful Fallbacks**: Basic analysis when models unavailable
- **Smart Model Loading**: Cached resources for performance

### **Content Processing:**
- **Unified FileProcessor**: Handles all file types consistently
- **ContentScorer**: Comprehensive scoring across 4 metrics
- **ContentSummarizer**: Advanced summarization and theme extraction
- **Error Handling**: Robust processing with user feedback

## 🎉 **Perfect For:**

- **Content marketers** analyzing campaign materials
- **Technical writers** evaluating documentation quality
- **Researchers** comparing multiple documents  
- **Bloggers** optimizing post quality and engagement
- **Students** analyzing academic papers and essays
- **Business analysts** reviewing reports and proposals

## ⚡ **Performance Features:**

- **Model caching** for faster subsequent analysis
- **Chunked processing** for large documents
- **Background processing** with progress indicators
- **Optimized visualizations** with Plotly and Matplotlib
- **Responsive design** for all screen sizes

The Unified Content Analysis Tool provides everything needed for comprehensive content evaluation in a single, powerful interface! 🚀
