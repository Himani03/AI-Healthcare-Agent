import streamlit as st

st.set_page_config(
    page_title="Stay Summarizer",
    page_icon=None,
    layout="wide"
)

# Custom CSS - Dark Professional Theme
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #ecf0f1 !important;
        font-family: 'Inter', sans-serif;
    }
    p, div, li {
        color: #bdc3c7;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ecf0f1;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1rem;
        color: #95a5a6;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Content Box */
    .content-box {
        background-color: #1e2530;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #2c3e50;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Stay Summarizer</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated patient discharge summaries</p>', unsafe_allow_html=True)

st.markdown("""
<div class="content-box">
    <h3>🚧 Module Under Development</h3>
    <p style="margin-top: 1rem;">This feature will allow medical professionals to:</p>
    <ul style="text-align: left; display: inline-block; margin-top: 1rem;">
        <li>Upload patient discharge notes or daily logs</li>
        <li>Generate concise summaries for doctors and nurses</li>
        <li>Highlight key events and vital trends during the stay</li>
    </ul>
</div>
""", unsafe_allow_html=True)
