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
RESULTS_DIR = os.path.join(BASE_DIR, "../results")
LABELS_FILE = os.path.join(RESULTS_DIR, "labeled-transcripts.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
with open(INPUT_FILE) as f:
    data = json.load(f)
    records = data['records']

SOURCE_TO_SAMPLE_INDEX = {
    str(record.get('source_index')): int(record.get('sample_index'))
    for record in records
    if record.get('source_index') is not None and record.get('sample_index') is not None
}

labeled_data = {}
current_index = 0


def get_record_key(record):
    """Use sample_index as the canonical transcript identifier for progress."""
    return str(record.get('sample_index'))


def extract_persona(meta_data):
    """Return only persona data from either full metadata or persona-shaped input."""
    if not isinstance(meta_data, dict):
        return None
    if 'persona' in meta_data and isinstance(meta_data.get('persona'), dict):
        return meta_data.get('persona')
    return meta_data


def persist_labels():
    """Persist labels to disk with the golden-set schema."""
    transcripts = sorted(
        labeled_data.values(),
        key=lambda item: item['sample_index']
    )

    with open(LABELS_FILE, 'w') as f:
        json.dump({'transcripts': transcripts}, f, indent=2)


def advance_to_next_unlabeled():
    """Move current_index to the next unlabeled record."""
    global current_index

    while current_index < len(records):
        record_id = get_record_key(records[current_index])
        if record_id not in labeled_data:
            break
        current_index += 1


def load_existing_labels():
    """Load saved labels from disk and restore progress."""
    global labeled_data, current_index

    if not os.path.exists(LABELS_FILE):
        return

    try:
        with open(LABELS_FILE) as f:
            saved = json.load(f)

        # Backward compatibility: accept older {'labels': {...}} files and migrate in-memory.
        if isinstance(saved, dict) and 'transcripts' in saved:
            saved_transcripts = saved.get('transcripts', [])
        elif isinstance(saved, dict) and 'labels' in saved:
            saved_transcripts = []
            for key, value in saved.get('labels', {}).items():
                sample_index = value.get('sample_index', key)
                saved_transcripts.append({
                    'sample_index': int(sample_index),
                    'judge_prediction_score': value.get('judge_prediction_score'),
                    'human_label': value.get('human_label', value.get('label')),
                    'task_description': value.get('task_description'),
                    'meta_data': extract_persona(value.get('meta_data', value.get('metadata')))
                })
        else:
            saved_transcripts = []

        normalized_labels = {}
        for item in saved_transcripts:
            sample_index = item.get('sample_index')
            if sample_index is None and item.get('source_index') is not None:
                sample_index = SOURCE_TO_SAMPLE_INDEX.get(str(item.get('source_index')))
            if sample_index is None:
                continue
            normalized_labels[str(sample_index)] = {
                'sample_index': int(sample_index),
                'judge_prediction_score': item.get('judge_prediction_score'),
                'human_label': item.get('human_label'),
                'task_description': item.get('task_description'),
                'meta_data': extract_persona(item.get('meta_data', item.get('metadata')))
            }

        labeled_data = normalized_labels
        current_index = 0
        advance_to_next_unlabeled()
    except (json.JSONDecodeError, OSError, TypeError):
        # Start fresh if the saved file is malformed or unreadable.
        labeled_data = {}
        current_index = 0


load_existing_labels()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/get-record')
def get_record():
    """Get current record without labels"""
    global current_index

    advance_to_next_unlabeled()
    
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
        current_record = records[current_index]
        record_id = get_record_key(current_record)
        labeled_data[record_id] = {
            'sample_index': current_record['sample_index'],
            'judge_prediction_score': current_record.get('judge_prediction_score'),
            'human_label': label_value,
            'task_description': current_record.get('task_description'),
            'meta_data': extract_persona(current_record.get('metadata'))
        }
        current_index += 1
        advance_to_next_unlabeled()
        persist_labels()
    
    return jsonify({'success': True, 'total_labeled': len(labeled_data)})


@app.route('/api/save-golden-set', methods=['POST'])
def save_golden_set():
    """Force-save current labels to the fixed results file."""
    persist_labels()
    
    return jsonify({
        'success': True,
        'file': LABELS_FILE,
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
    print(f"💾 Auto-save file: {LABELS_FILE}")
    print(f"♻️  Restored labels: {len(labeled_data)}")
    print(f"🌐 Open http://localhost:5000 in your browser")
    print(f"⌨️  Arrow Left = Failure (0), Arrow Right = Success (1)")
    app.run(debug=True, port=9001)
