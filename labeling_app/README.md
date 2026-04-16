# 📝 Transcript Labeling Web App

Quick web app for labeling transcripts and creating a golden set.

## Setup

### Option 1: Using the run script (easiest)
```bash
chmod +x run.sh
./run.sh
```

### Option 2: Manual setup
```bash
pip install -r requirements.txt
python3 app.py
```

## Usage

1. **Start the app** - Opens at `http://localhost:5000`
2. **Label transcripts** using keyboard shortcuts:
   - **← Left Arrow** = Failure (target agent failed)
   - **→ Right Arrow** = Success (target agent succeeded)
   - **Ctrl + S** = Save golden set to file

3. **Submit labels** - Click buttons or use arrow keys
4. **Monitor progress** - Progress bar shows completion status
5. **Save golden set** - When done (or use Ctrl+S)

## Output

Labeled data is saved to:
```
../results/golden_set/labels_YYYYMMDD_HHMMSS.json
```

Each label includes:
- label (0 = failure, 1 = success)
- task_description
- persona information
- timestamp

## Structure

```
labeling_app/
├── app.py              # Flask backend
├── templates/
│   └── index.html      # Web UI
├── requirements.txt    # Dependencies
└── run.sh             # Launch script
```

## Tips

- Use full screen for better readability
- Scroll through transcripts to see full conversations
- Save frequently with Ctrl+S
- Progress automatically updates with each label
