import os
import json
import glob
from difflib import SequenceMatcher

def is_similar(str1, str2, threshold=0.75):
    """Returns True if the two strings are highly similar."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio() > threshold

def truncate_natural_end(transcript):
    """
    Truncates a transcript by detecting RLHF repetitive loops 
    and explicit user conversational sign-offs.
    """
    if not transcript:
        return transcript

    # 1. Fuzzy Robotic Loop Detection
    # Look for Target turns that are highly similar to previous Target turns.
    for i in range(len(transcript)):
        if transcript[i].get("speaker") == "Target":
            curr_content = transcript[i].get("content", "").strip()
            
            # Ignore very short generic responses to avoid false positives
            if len(curr_content) < 20:
                continue
            
            for j in range(i - 1, -1, -1):
                if transcript[j].get("speaker") == "Target":
                    prev_content = transcript[j].get("content", "").strip()
                    
                    if is_similar(curr_content, prev_content):
                        # Truncate to exclude the start of the repetitive loop
                        return transcript[:i]

    # 2. Comprehensive User Sign-off Detection
    closing_phrases = [
        "consider the case closed",
        "end our conversation",
        "satisfied with the outcome",
        "no further questions",
        "that will be all",
        "i'm all set",
        "that's all for now",
        "matter resolved",
        "close this conversation",
        "gotta go",
        "catch you up",
        "adiós",
        "adios",
        "have a great day",
        "have a good day",
        "thanks again for your help",
        "i think we've covered everything",
        "i can get back to work now"
    ]
    
    for i, turn in enumerate(transcript):
        if turn.get("speaker") == "User":
            content = turn.get("content", "").lower()
            if any(phrase in content for phrase in closing_phrases):
                # Keep the user's sign-off, and allow the Target EXACTLY one polite final response.
                # Avoid IndexError if the user's sign-off was the very last turn.
                return transcript[:min(i + 2, len(transcript))]

    # Return unchanged if no end conditions are met
    return transcript

def clean_directory(target_dir):
    all_jsons = glob.glob(os.path.join(target_dir, "*.json"))
    target_files = [f for f in all_jsons if not f.endswith("_masked.json")]
    
    total_files_modified = 0
    total_conversations_truncated = 0

    print(f"Scanning {len(target_files)} files for early stopping failures...")

    for file_path in target_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        file_modified = False
        
        for item in data:
            if 'transcript' in item:
                original_len = len(item['transcript'])
                item['transcript'] = truncate_natural_end(item['transcript'])
                
                if len(item['transcript']) < original_len:
                    file_modified = True
                    total_conversations_truncated += 1
                    
        if file_modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            total_files_modified += 1
            print(f"  -> Truncated spirals in: {os.path.basename(file_path)}")

    print("-" * 40)
    print("Cleanup Complete!")
    print(f"Files Modified: {total_files_modified}")
    print(f"Total Conversations Truncated: {total_conversations_truncated}")

if __name__ == "__main__":
    # Specify the directory containing your JSON files here
    DIRECTORY_PATH = "data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4"
    
    clean_directory(DIRECTORY_PATH)