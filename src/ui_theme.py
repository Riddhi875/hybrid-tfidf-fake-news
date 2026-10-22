
import streamlit as st


def apply_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F7F7F2;
            color: #1F2937;
        }

        .stApp p,
        .stApp li,
        .stApp label {
            color: #1F2937;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #172554 !important;
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {
            color: #4B5563 !important;
        }

        input, textarea,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {
            color: #111827 !important;
            background-color: #FFFFFF !important;
            -webkit-text-fill-color: #111827 !important;
        }

        [data-baseweb="select"] > div {
            color: #111827;
            background-color: #FFFFFF;
        }

        .stButton button {
            color: #FFFFFF !important;
            background-color: #1D4ED8 !important;
        }

        .stButton button p {
            color: #FFFFFF !important;
        }

        .stApp a {
            color: #1D4ED8 !important;
        }

        .hero-card, .metric-card, .disclaimer {
            background-color: #FFFFFF !important;
            color: #1F2937 !important;
            border: 1px solid #D1D5DB;
            border-radius: 12px;
            padding: 1rem;
        }

        .hero-card *, .metric-card *, .disclaimer * {
            color: #1F2937 !important;
        }

        [data-testid="stAlert"],
        [data-testid="stAlert"] p {
            color: #1F2937 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
