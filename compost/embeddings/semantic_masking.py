"""
Semantic Masking Module
Implements Named Entity Recognition (NER) based redaction to obscure 
explicit occupational titles and demographic labels before embedding generation.

This forces the classifier to rely on deep stylistic variations rather than 
explicit textual announcements of identity.
"""

import re
import spacy
from typing import List, Dict, Tuple
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticMasker:
    """
    Redacts explicit demographic identifiers and occupational titles using spaCy NER.
    """
    
    # Domain-specific occupation titles to redact
    OCCUPATION_KEYWORDS = {
        "nurse", "doctor", "physician", "surgeon", "therapist", "teacher", "professor",
        "engineer", "architect", "analyst", "manager", "director", "ceo", "executive",
        "developer", "programmer", "designer", "artist", "writer", "journalist",
        "lawyer", "attorney", "judge", "prosecutor", "clerk", "accountant", "auditor",
        "clerk", "secretary", "administrator", "assistant", "cleaner", "janitor",
        "housekeeper", "maid", "driver", "taxi", "truck", "bus", "conductor",
        "chef", "cook", "waiter", "bartender", "cashier", "salesperson", "retail",
        "police", "officer", "detective", "agent", "military", "soldier", "firefighter",
        "electrician", "plumber", "carpenter", "mechanic", "technician", "laborer",
        "editor", "producer", "director", "actor", "singer", "musician", "performer",
        "consultant", "advisor", "coach", "trainer", "instructor", "sheriff"
    }
    
    # Demographic terms to redact
    DEMOGRAPHIC_KEYWORDS = {
        "hispanic", "latino", "mexican", "spanish", "cuban", "puerto rican",
        "black", "african american", "asian", "chinese", "indian", "japanese",
        "white", "caucasian", "european", "middle eastern", "arab",
        "native american", "indigenous", "first nations",
        "male", "female", "man", "woman", "boy", "girl", "guy", "lady",
        "he", "she", "him", "her", "his", "hers", "myself", "himself", "herself"
    }
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the semantic masker with a spaCy NLP model.
        
        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model {model_name} not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            self.nlp = spacy.load(model_name)
    
    def redact_explicit_identifiers(self, text: str) -> str:
        """
        Redacts explicit demographic and occupational identifiers from text.
        Used for explicit variant prompts where persona mentions identity explicitly.
        
        Args:
            text: Input text to redact
            
        Returns:
            Text with identifiers replaced by [MASKED] tokens
        """
        doc = self.nlp(text)
        
        # First pass: redact named entities (PERSON, ORG, etc.)
        redacted = text
        for ent in reversed(doc.ents):  # Reverse to preserve character indices
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                # Replace with [MASKED] to preserve token structure
                redacted = redacted[:ent.start_char] + "[MASKED]" + redacted[ent.end_char:]
        
        # Second pass: redact demographic keywords (case-insensitive)
        for keyword in self.DEMOGRAPHIC_KEYWORDS:
            # Use word boundaries to avoid partial matches
            pattern = rf"\b{re.escape(keyword)}\b"
            redacted = re.sub(pattern, "[MASKED]", redacted, flags=re.IGNORECASE)
        
        # Third pass: redact occupation keywords
        for keyword in self.OCCUPATION_KEYWORDS:
            pattern = rf"\b{re.escape(keyword)}\b"
            redacted = re.sub(pattern, "[MASKED-ROLE]", redacted, flags=re.IGNORECASE)
        
        return redacted.strip()
    
    def get_masking_applied(self, text: str) -> bool:
        """
        Check if redaction would modify the text (i.e., identifiers present).
        
        Args:
            text: Input text to check
            
        Returns:
            True if redaction would change the text, False otherwise
        """
        doc = self.nlp(text)
        
        # Check for NER entities
        if any(ent.label_ in ["PERSON", "ORG", "GPE"] for ent in doc.ents):
            return True
        
        # Check for demographic keywords
        text_lower = text.lower()
        if any(word in text_lower for word in self.DEMOGRAPHIC_KEYWORDS):
            return True
        
        # Check for occupation keywords
        if any(word in text_lower for word in self.OCCUPATION_KEYWORDS):
            return True
        
        return False
    
    def process_transcript(
        self, 
        transcript: List[Dict], 
    ) -> List[Dict]:
        """
        Process a full transcript, redacting based on variant type.
        
        Args:
            transcript: List of turn dicts with 'speaker' and 'content' keys
            
        Returns:
            Processed transcript with redacted content for explicit variants
        """        
        # Redact identifiers
        processed_turns = []
        for turn in transcript:
            turn_copy = turn.copy()
            if "content" in turn_copy:
                turn_copy["content_original"] = turn_copy["content"]
                turn_copy["content"] = self.redact_explicit_identifiers(turn_copy["content"])
                turn_copy["redaction_applied"] = self.get_masking_applied(turn["content"])
            processed_turns.append(turn_copy)
        
        return processed_turns
    
    def process_dataframe(self, df, variant_type: str = "implicit") -> None:
        """
        In-place modification of DataFrame responses: redact explicit variant text.
        
        Args:
            df: Pandas DataFrame with 'response' and 'variant_type' columns
            variant_type: Default variant type if not in DataFrame
        """
        if "response" not in df.columns:
            logger.warning("DataFrame does not have 'response' column")
            return
        
        # If variant_type is in DataFrame, use per-row application
        if "variant_type" in df.columns:
            df["response"] = df.apply(
                lambda row: self.redact_explicit_identifiers(row["response"]),
                axis=1
            )
        else:
            df["response"] = df["response"].apply(self.redact_explicit_identifiers)
        
        logger.info(f"Semantic masking complete. Processed {len(df)} rows.")


def create_semantic_masker() -> SemanticMasker:
    """Factory function to create a SemanticMasker instance."""
    return SemanticMasker()
