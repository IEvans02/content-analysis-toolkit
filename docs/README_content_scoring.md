# Internal Content Scoring Tool

A comprehensive content analysis tool that scores text across multiple dimensions including readability, sentiment, engagement, SEO, and overall quality.

## 🚀 Quick Start

### Option 1: Lite Version (Recommended for testing)
The lite version uses basic NLP techniques and doesn't require heavy ML dependencies:

```bash
# Install lite dependencies
pip install -r requirements_lite.txt

# For file upload support (optional)
pip install python-docx PyPDF2

# Run the lite version
streamlit run content_scoring_tool_lite.py
```

**Note:** If you're using conda, make sure to use the correct Python:
```bash
# Use python3 or conda's python
python3 -m pip install -r requirements_lite.txt
python3 -m pip install python-docx PyPDF2
python3 -m streamlit run content_scoring_tool_lite.py
```

### Option 2: Full Version (Advanced features)
The full version includes advanced ML models but requires PyTorch >=2.6.0:

```bash
# Install full dependencies
pip install -r requirements_scoring_tool.txt

# Run the full version
streamlit run content_scoring_tool.py
```

If you encounter PyTorch version issues, run the setup script first:
```bash
python setup_scoring_tool.py
```

## 📊 Features

### Scoring Metrics
- **📚 Readability**: Flesch Reading Ease, Grade Level, ARI
- **😊 Sentiment**: Positive/Negative/Neutral analysis with confidence scores
- **🎯 Engagement**: Questions, CTAs, power words, personal pronouns
- **🔍 SEO**: Keyword density, content length, heading structure
- **⭐ Quality**: AI-powered or rule-based quality assessment

### File Upload Support 🆕
- **📄 PDF files**: Extract text from PDF documents
- **📝 Word documents**: Support for .docx and .doc files
- **📋 Text files**: Plain text .txt files
- **📝 Markdown files**: .md files with automatic formatting removal
- **Batch uploads**: Upload multiple files for comparison
- **Preview functionality**: See extracted content before analysis

### Analysis Modes
1. **Single Content**: Comprehensive analysis with text input or file upload
2. **Batch Analysis**: Compare multiple pieces via text or file uploads
3. **Comparative Analysis**: Side-by-side comparison with dual input methods

### Visualization
- Interactive radar charts
- Detailed scoring breakdowns
- Comparative bar charts
- Export capabilities (JSON)

## 🛠️ Configuration

### OpenAI API (Optional)
For AI-powered quality scoring, set your OpenAI API key:
- Environment variable: `OPENAI_API_KEY`
- Or enter in the app interface

### Target Keywords
Add comma-separated keywords for SEO scoring:
```
keyword1, keyword2, keyword3
```

## 📝 Usage Examples

### Single Content Analysis
1. Paste your content in the text area
2. Optionally add target keywords
3. Click "Analyze Content"
4. Review detailed scores and insights
5. Export results if needed

### Batch Analysis
1. Set number of content pieces
2. Add names and content for each piece
3. Click "Analyze All Content"
4. Compare results in table and chart format

### Comparative Analysis
1. Enter two pieces of content
2. Click "Compare Content"
3. See side-by-side metrics and winner

## 🔧 Troubleshooting

### Common Issues

**PyTorch Version Error**
```
ValueError: Due to a serious vulnerability issue in torch.load...
```
Solution: Upgrade PyTorch to >=2.6.0 or use the lite version

**Missing Dependencies**
```
ModuleNotFoundError: No module named 'textstat'
```
Solution: Install requirements with `pip install -r requirements_lite.txt`

**OpenAI API Issues**
- Make sure your API key is valid
- Check your usage limits
- The tool works without OpenAI (basic quality assessment)

## 📋 Dependencies

### Lite Version
- streamlit
- pandas
- numpy  
- matplotlib
- seaborn
- textstat
- plotly
- wordcloud
- openai (optional)

### Full Version
Includes all lite dependencies plus:
- transformers
- torch >=2.6.0
- scikit-learn
- nltk

## 🎯 Scoring Algorithm

Overall score is calculated as weighted average:
- Readability: 20%
- Sentiment: 15%
- Engagement: 25%
- SEO: 20%
- Quality: 20%

Each metric is normalized to 0-100 scale.

## 🔮 Future Enhancements

- [ ] Tone analysis (formal, casual, professional)
- [ ] Content categorization
- [ ] A/B testing features
- [ ] Integration with content management systems
- [ ] API endpoints for programmatic access
- [ ] Custom scoring weights
- [ ] Historical trend analysis

## 🤝 Contributing

This is an internal tool. For feature requests or bug reports, contact the development team.

## 📄 License

Internal use only - proprietary software.
