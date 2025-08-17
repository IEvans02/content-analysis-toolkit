import streamlit as st
import openai
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import os
from collections import Counter
from transformers import pipeline

# --- Instructions for API Key ---
st.markdown("""
**Instructions:**
- Set your OpenAI API key in your environment as `OPENAI_API_KEY`, or add it to Streamlit secrets.
- Install requirements: `pip install streamlit openai wordcloud matplotlib`
""")

# --- Set your OpenAI API key here or use st.secrets ---
openai_api_key = os.environ.get("OPENAI_API_KEY")
try:
    if not openai_api_key and "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass
if not openai_api_key:
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
openai.api_key = openai_api_key

st.title("Interview Messaging Comparison & Word Map")

# --- Hugging Face Summarizer (cached) ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

summarizer = load_summarizer()

# --- Input Section ---
st.header("Paste or Upload Multiple Interview Transcripts")

num_transcripts = st.number_input("Number of transcripts", min_value=2, max_value=5, value=2, step=1)
transcripts = []

for i in range(num_transcripts):
    st.markdown(f"**Transcript {i+1}**")
    text = st.text_area(f"Transcript {i+1}", key=f"text_{i}", height=150)
    uploaded = st.file_uploader(f"Or upload Transcript {i+1}", type=["txt"], key=f"file_{i}")
    if uploaded:
        text = uploaded.read().decode("utf-8")
    transcripts.append(text)

# --- Comparison and Summary with Hugging Face ---
if all(t.strip() for t in transcripts) and st.button("Compare & Generate Word Maps"):
    with st.spinner("Analyzing with Hugging Face Summarizer..."):
        # Comparison: Summarize each transcript individually
        st.subheader("Summary for Each Transcript")
        for idx, t in enumerate(transcripts):
            summary = summarizer(t, max_length=130, min_length=30, do_sample=False)
            st.markdown(f"**Transcript {idx+1}**: {summary[0]['summary_text']}")

        # Combined summary: All transcripts together
        st.subheader("Summary of Commonalities and Differences")
        summary_input = "\n\n".join([f"Transcript {idx+1}:\n{t}" for idx, t in enumerate(transcripts)])
        summary = summarizer(summary_input, max_length=200, min_length=50, do_sample=False)
        st.markdown(summary[0]['summary_text'])

    # --- Word Cloud Generation ---
    def clean_text(text, exclude=None):
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        words = text.split()
        if exclude:
            words = [w for w in words if w not in exclude]
        return ' '.join(words)

    exclude_terms = {"interviewer", "interviewee"}

    st.subheader("Word Clouds for Each Transcript")
    cols = st.columns(num_transcripts)
    for idx, (col, t) in enumerate(zip(cols, transcripts)):
        with col:
            st.markdown(f"**Transcript {idx+1}**")
            wc = WordCloud(width=400, height=300, background_color='white', stopwords=None).generate(clean_text(t, exclude=exclude_terms))
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    # --- Commonality Word Cloud ---
    def get_common_words_multi(texts):
        sets = [set(clean_text(t).split()) for t in texts]
        exclude = {"interviewer", "interviewee"}
        if sets:
            common = set.intersection(*sets) - exclude
            all_words = clean_text(" ".join(texts)).split()
            common_counts = Counter([w for w in all_words if w in common])
            return " ".join([w for w in all_words if w in common])
        return ""

    st.subheader("Word Cloud of Common Words Across All Transcripts")
    common_text = get_common_words_multi(transcripts)
    if common_text.strip():
        wc_common = WordCloud(width=400, height=300, background_color='white').generate(common_text)
        fig_common, ax_common = plt.subplots(figsize=(5, 4))
        ax_common.imshow(wc_common, interpolation='bilinear')
        ax_common.axis("off")
        st.pyplot(fig_common)
    else:
        st.info("No common words found across all transcripts.")

    # --- Thematic Analysis ---
    st.subheader("Thematic Analysis for Each Transcript")
    for idx, t in enumerate(transcripts):
        theme_prompt = f"List the main themes or topics in the following transcript.\n\n{t}"
        theme_summary = summarizer(theme_prompt, max_length=60, min_length=20, do_sample=False)
        st.markdown(f"**Transcript {idx+1} Themes:** {theme_summary[0]['summary_text']}")

    # --- Thematic Word Clouds ---
    st.subheader("Thematic Word Clouds for Each Transcript")
    for idx, t in enumerate(transcripts):
        theme_prompt = f"List the main themes or topics in the following transcript.\n\n{t}"
        theme_summary = summarizer(theme_prompt, max_length=60, min_length=20, do_sample=False)
        themes_text = theme_summary[0]['summary_text']
        wc = WordCloud(width=400, height=300, background_color='white').generate(themes_text)
        st.markdown(f"**Transcript {idx+1} Thematic Word Cloud**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    # --- Thematic Word Cloud of Common Themes ---
    st.subheader("Word Cloud of Common Themes Across All Transcripts")
    theme_summaries = []
    for idx, t in enumerate(transcripts):
        theme_prompt = f"List the main themes or topics in the following transcript.\n\n{t}"
        theme_summary = summarizer(theme_prompt, max_length=60, min_length=20, do_sample=False)
        theme_summaries.append(theme_summary[0]['summary_text'])
    # Tokenize and find common words/phrases
    exclude_theme_terms = {"interviewer", "interviewee", "company", "biggest", "challenge"}
    theme_sets = [set(re.sub(r'[^\w\s]', '', s).lower().split()) - exclude_theme_terms for s in theme_summaries]
    if theme_sets:
        common_theme_words = set.intersection(*theme_sets)
        if common_theme_words:
            # Build a text string for the word cloud
            all_themes = " ".join(theme_summaries)
            filtered_words = [w for w in re.sub(r'[^\w\s]', '', all_themes).lower().split() if w in common_theme_words]
            common_themes_text = " ".join(filtered_words)
            wc_common_themes = WordCloud(width=400, height=300, background_color='white').generate(common_themes_text)
            fig_common_themes, ax_common_themes = plt.subplots(figsize=(5, 4))
            ax_common_themes.imshow(wc_common_themes, interpolation='bilinear')
            ax_common_themes.axis("off")
            st.pyplot(fig_common_themes)
        else:
            st.info("No common themes found across all transcripts.")
    else:
        st.info("No themes to compare.") 