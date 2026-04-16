#!/usr/bin/env python3
"""
Quick labeling webapp for transcripts.
Arrow left = failure (0), Arrow right = success (1)
"""

import json
import os
from datetime import datetime
from flask import Flask, render_template, jsonify, request

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Configuration
INPUT_FILE = os.path.join(BASE_DIR, "../data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/judge_validation_sample_1000.json")
GOLDEN_SET_DIR = os.path.join(BASE_DIR, "../results/golden_set")
LABELS_FILE = os.path.join(GOLDEN_SET_DIR, f"labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

os.makedirs(GOLDEN_SET_DIR, exist_ok=True)

# Load data
with open(INPUT_FILE) as f:
    data = json.load(f)
    records = data['records']

labeled_data = {}
current_index = 0


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/get-record')
def get_record():
    """Get current record without labels"""
    global current_index
    
    if current_index >= len(records):
        return jsonify({
            'done': True,
            'total_labeled': len(labeled_data),
            'total_records': len(records)
        })
    
    record = records[current_index]
    return jsonify({
        'done': False,
        'index': current_index,
        'total': len(records),
        'task_description': record['task_description'],
        'persona': record['metadata']['persona'],
        'transcript': record['transcript'],
        'total_labeled': len(labeled_data)
    })


@app.route('/api/label', methods=['POST'])
def label():
    """Save label and move to next record"""
    global current_index
    
    data = request.json
    label_value = data.get('label')  # 0 = failure (left), 1 = success (right)
    
    if current_index < len(records):
        record_id = records[current_index]['sample_index']
        labeled_data[record_id] = {
            'label': label_value,
            'task_description': records[current_index]['task_description'],
            'persona': records[current_index]['metadata']['persona'],
            'timestamp': datetime.now().isoformat()
        }
        current_index += 1
    
    return jsonify({'success': True, 'total_labeled': len(labeled_data)})


@app.route('/api/save-golden-set', methods=['POST'])
def save_golden_set():
    """Save current labels to golden set file"""
    output_file = LABELS_FILE
    with open(output_file, 'w') as f:
        json.dump({
            'total_labeled': len(labeled_data),
            'total_records': len(records),
            'timestamp': datetime.now().isoformat(),
            'labels': labeled_data
        }, f, indent=2)
    
    return jsonify({
        'success': True,
        'file': output_file,
        'count': len(labeled_data)
    })


@app.route('/api/stats')
def stats():
    """Get labeling statistics"""
    return jsonify({
        'total_labeled': len(labeled_data),
        'total_records': len(records),
        'progress': f"{len(labeled_data)}/{len(records)}"
    })


if __name__ == '__main__':
    print(f"📝 Labeling App Started!")
    print(f"📊 Total records to label: {len(records)}")
    print(f"💾 Golden set will be saved to: {GOLDEN_SET_DIR}")
    print(f"🌐 Open http://localhost:5000 in your browser")
    print(f"⌨️  Arrow Left = Failure (0), Arrow Right = Success (1)")
    app.run(debug=True, port=9001)
