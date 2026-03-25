#!/bin/zsh
cd /Users/db/Desktop/Ai

if [ -x "./.venv/bin/python" ]; then
	PYTHON_BIN="./.venv/bin/python"
else
	PYTHON_BIN="./venv/bin/python"
fi

"$PYTHON_BIN" -m streamlit run signals.py --server.port 8502
