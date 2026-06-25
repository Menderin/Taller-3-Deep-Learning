"""Streamlit orchestrator.

Run with:
    streamlit run Frontend/streamlit_main.py
"""

from src.inference.streamlit_app import run_app


if __name__ == "__main__":
    run_app()
