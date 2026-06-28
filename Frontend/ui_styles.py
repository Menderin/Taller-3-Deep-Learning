"""Custom CSS styles for the Streamlit app."""

import base64
from pathlib import Path
import streamlit as st

def get_base64_of_bin_file(bin_file: Path) -> str:
    """Read binary file and return base64 string."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def apply_custom_css():
    """Apply custom CSS to improve aesthetics and compactness."""
    
    # Check if the background image exists
    bg_path = Path(__file__).resolve().parent / "assets" / "fondo.png"
    bg_css = ""
    if bg_path.exists():
        bg_base64 = get_base64_of_bin_file(bg_path)
        bg_css = f"""
        .stApp {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        """

    css = f"""
    <style>
        {bg_css}

        /* Hide the default Streamlit sidebar toggle button */
        [data-testid="collapsedControl"] {{
            display: none;
        }}
        section[data-testid="stSidebar"] {{
            display: none;
        }}

        /* Custom styling for tabs to make them look sleek */
        [data-baseweb="tab-list"] {{
            gap: 24px;
            background-color: transparent;
            justify-content: center;
        }}
        [data-baseweb="tab"] {{
            padding-top: 10px;
            padding-bottom: 10px;
            color: rgba(255, 255, 255, 0.7);
            border-bottom-color: transparent !important;
        }}
        [data-baseweb="tab"] p {{
            font-size: 1.3rem !important;
            font-weight: 600;
        }}
        [data-baseweb="tab"][aria-selected="true"] {{
            color: white;
            border-bottom: 3px solid white !important;
            border-bottom-color: white !important;
        }}
        [data-baseweb="tab-highlight"] {{
            background-color: white;
            display: none; /* We use border-bottom instead for a cleaner underline */
        }}

        /* Center page headings while preserving regular body alignment. */
        h1, h2, h3 {{
            text-align: center;
        }}
        h2 {{
            font-size: 2rem !important;
        }}
        h3 {{
            font-size: 1.55rem !important;
        }}

        /* Make text readable without a massive block behind everything */
        h1, h2, h3, h4, p, label, span {{
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);
        }}

        .active-model {{
            margin: 0.4rem auto 1.25rem auto;
            color: rgba(255, 255, 255, 0.78);
            font-size: 0.9rem;
            text-align: center;
        }}
        .hero-subtitle {{
            margin: 0.25rem auto 0.5rem auto;
            font-size: 2.8rem;
            font-weight: 600;
            text-align: center;
        }}

        .app-footer {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.65rem;
            margin-top: 2.5rem;
            padding: 1.25rem 0 0.5rem 0;
            border-top: 1px solid rgba(255, 255, 255, 0.18);
            color: rgba(255, 255, 255, 0.72);
            font-size: 0.9rem;
            text-align: center;
        }}
        .app-footer a {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 600;
            text-decoration: none;
        }}
        .app-footer a:hover {{
            text-decoration: underline;
        }}
        .github-logo {{
            width: 20px;
            height: 20px;
            border-radius: 0;
        }}

        /* Apply semi-transparent background ONLY to content blocks (columns) */
        [data-testid="column"] {{
            background-color: rgba(15, 20, 25, 0.75);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.4);
        }}

        /* Remove the massive block background */
        .block-container {{
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }}

        /* Compact metric boxes */
        [data-testid="stMetricValue"] {{
            font-size: 1.5rem !important;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.9rem !important;
        }}

        /* Adjust images slightly */
        img {{
            border-radius: 8px;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
