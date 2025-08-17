import streamlit as st
import PyPDF2
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

st.title("Competitor Messaging Analysis - Starter App")

st.header("1. Upload Competitor PDFs")
pdf_files = st.file_uploader("Upload one or more competitor PDFs", type=["pdf"], accept_multiple_files=True)

st.header("2. Enter Competitor Website URLs")
urls_input = st.text_area("Enter one or more competitor website URLs (one per line)")
urls = [u.strip() for u in urls_input.splitlines() if u.strip()]

# --- Extract text from PDFs ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

pdf_texts = []
if pdf_files:
    for pdf in pdf_files:
        pdf_text = extract_text_from_pdf(pdf)
        pdf_texts.append(pdf_text)

# --- Scrape text from websites ---
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove scripts/styles
        for s in soup(['script', 'style']):
            s.decompose()
        text = ' '.join(soup.stripped_strings)
        return text
    except Exception as e:
        return f"Error fetching {url}: {e}"

url_texts = []
if urls:
    for url in urls:
        url_text = extract_text_from_url(url)
        url_texts.append(url_text)

st.header("3. Extracted Text Preview")
if pdf_texts:
    for i, text in enumerate(pdf_texts):
        st.subheader(f"PDF {i+1} Extracted Text (first 500 chars)")
        st.write(text[:500] + ("..." if len(text) > 500 else ""))
if url_texts:
    for i, text in enumerate(url_texts):
        st.subheader(f"URL {i+1} Extracted Text (first 500 chars)")
        st.write(text[:500] + ("..." if len(text) > 500 else ""))

# --- Word Cloud Helper ---
def clean_text(text, exclude=None):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = text.split()
    if exclude:
        words = [w for w in words if w not in exclude]
    return ' '.join(words)

exclude_terms = {"interviewer", "interviewee"}

st.header("4. Word Clouds for Each Competitor")
if pdf_texts:
    st.subheader("PDFs")
    cols = st.columns(len(pdf_texts))
    for i, (col, text) in enumerate(zip(cols, pdf_texts)):
        with col:
            st.markdown(f"**PDF {i+1}**")
            wc = WordCloud(width=400, height=300, background_color='white').generate(clean_text(text, exclude=exclude_terms))
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
if url_texts:
    st.subheader("Websites")
    cols = st.columns(len(url_texts))
    for i, (col, text) in enumerate(zip(cols, url_texts)):
        with col:
            st.markdown(f"**URL {i+1}**")
            wc = WordCloud(width=400, height=300, background_color='white').generate(clean_text(text, exclude=exclude_terms))
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

st.header("5. Your Own Content (for Future Comparison)")
own_text = st.text_area("Paste your own content here (optional)")

st.header("6. Next Steps (Template Placeholders)")
st.markdown("""
- [ ] Extract and display key terms/phrases
- [ ] Compare to your own content (upload or paste)
- [ ] Highlight commonalities and gaps
- [ ] Add summarization or advanced analysis (optional)
""") 