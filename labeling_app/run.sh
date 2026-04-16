#!/bin/bash
# Run the transcript labeling webapp

cd "$(dirname "$0")"

# Install dependencies if needed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "🚀 Starting Labeling App..."
python3 app.py
