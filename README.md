# 🔬 Content Analysis Toolkit

**Professional content analysis with thematic summaries, word maps, and comprehensive scoring**

*Inspired by Simmons & Schmid design aesthetic*

## 🚀 Quick Start

```bash
# Simple one-command launch
python3 run_content_tool.py
```

The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
Tim_AI_Work/
├── 🔬 run_content_tool.py          # Quick launcher script
├── 📊 content_tools/               # Main content analysis tools
│   ├── content_analysis_tool.py           # 🌟 MAIN TOOL (Marketing focused)
│   ├── app.py                      # Original interview transcript analyzer
│   ├── content_scoring_tool_lite.py       # Lightweight version
│   ├── unified_content_analysis_tool.py   # Full ML version
│   ├── content_scoring_tool.py     # Complete scoring tool
│   ├── install_file_support.py    # Dependency installer
│   └── setup_scoring_tool.py       # Setup helper
├── 🗂️ legacy_tools/               # Other specialized tools
├── 📚 docs/                       # Documentation
├── 📋 requirements/               # Dependency files
└── 🐍 venv/                       # Python virtual environment
```

## ✨ Features

### 🎯 **Thematic Summary & Analysis**
- **Smart theme extraction** with categorization
- **Beautiful word clouds** (general + theme-focused)
- **Key excerpt identification** from any content
- **Export capabilities** for summaries and analysis

### 📊 **Content Scoring**
- **Readability analysis** (Flesch-Kincaid, ARI)
- **Sentiment analysis** with detailed scoring
- **SEO optimization** metrics and recommendations
- **Engagement scoring** for better content performance

### ⚖️ **Document Comparison**
- **Multi-document analysis** with theme correlation
- **Visual comparisons** with radar charts
- **Common theme identification** across documents
- **Performance benchmarking**

### 📁 **File Support**
- **PDF documents** (.pdf)
- **Word documents** (.docx, .doc)
- **Text files** (.txt)
- **Markdown files** (.md)

## 🎨 Design

Inspired by the clean, sophisticated aesthetic of **Simmons & Schmid**:
- **Inter font** for clean typography
- **Gradient headers** with professional color palette
- **Rounded cards** with subtle shadows
- **Smooth animations** and hover effects
- **Intuitive tabbed interface** for organized results

## 🛠️ Setup

### Prerequisites
- Python 3.11+
- Virtual environment (included)

### Installation
```bash
# Clone or navigate to the project
cd Tim_AI_Work

# Activate virtual environment
source venv/bin/activate

# Install any missing dependencies (automatic on first run)
python3 content_tools/install_file_support.py

# Launch the tool
python3 run_content_tool.py
```

## 📖 Documentation

Detailed documentation available in the `docs/` folder:
- Content scoring guide
- Setup instructions
- Troubleshooting
- Feature explanations

## 🎯 Perfect For

- **Content creators** analyzing blog posts and articles
- **Marketers** optimizing content performance
- **Researchers** extracting themes from interviews/transcripts
- **Writers** improving readability and engagement
- **Teams** comparing document performance

---

*Built with attention to detail and a knack for establishing insights that matter*
