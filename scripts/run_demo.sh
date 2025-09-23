#!/usr/bin/env bash
set -e
python -m pip install -r requirements.txt
export $(grep -v '^#' .env | xargs -d '\n' -I {} echo {}) 2>/dev/null || true
streamlit run streamlit_app.py
