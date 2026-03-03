import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

class RepresentationalEvaluator:
    def __init__(self, embedder_model: str = 'all-MiniLM-L6-v2'):
        pass

    def calculate_confidence(self, target_logprobs: List[float]) -> float:
        """
        Calculates the average log-likelihood (confidence) of user-referential tokens.
        Requires logprobs directly from the target LLM inference engine.
        """
        valid_logprobs = [lp for lp in target_logprobs if lp is not None and not np.isnan(lp)]
        if not valid_logprobs:
            return np.nan
        return float(np.mean(valid_logprobs))

    def calculate_d_ccd(self, implicit_logprobs: List[float], explicit_logprobs: List[float]) -> float:
        """
        Calculates the Coreference Confidence Disparity (d_CCD) as the global delta
        between the model's confidence in explicit vs implicit contexts.
        """
        conf_implicit = self.calculate_confidence(implicit_logprobs)
        conf_explicit = self.calculate_confidence(explicit_logprobs)
        
        if np.isnan(conf_implicit) or np.isnan(conf_explicit):
            return np.nan
            
        # Delta: How much does the literal presence of the identity label 
        # alter the model's confidence? (Negative means explicit label reduces confidence)
        return float(conf_explicit - conf_implicit)

    def calculate_semantic_steering(
        self, 
        implicit_target_embeddings: np.ndarray, 
        explicit_target_embeddings: np.ndarray, 
        axis_v: np.ndarray,
        topic_pole_sim: float,
        persona_pole_sim: float
    ) -> Dict[str, float]:
        """
        CoMPosT-based Semantic Steering Score.
        Tracks how the conversation drifts along the axis between a neutral topic and an exaggerated persona
        by projecting the target LLM's responses onto the statistically derived axis_v = (Persona Pole - Topic Pole).

        Args:
            implicit_target_embeddings: SBERT embeddings of the Target's responses (Implicit).
            explicit_target_embeddings: SBERT embeddings of the Target's responses (Explicit).
            axis_v: The geometric axis representing the shift toward the persona stereotype.
            topic_pole_sim: Baseline cosine similarity of the unmarked topic (mean_dp).
            persona_pole_sim: Baseline cosine similarity of the persona (mean_dt).
            
        Returns:
            Dictionary containing implicit steering, explicit steering, and the delta.
        """
        
        if len(implicit_target_embeddings) == 0 or len(explicit_target_embeddings) == 0:
            return {'implicit_steering': np.nan, 'explicit_steering': np.nan, 'delta_steering': np.nan}
            

        axis_v = axis_v.reshape(1, -1)
        implicit_sim = np.mean(cosine_similarity(implicit_target_embeddings, axis_v))
        explicit_sim = np.mean(cosine_similarity(explicit_target_embeddings, axis_v))
        
        # Denominator scaling (Distance between the two statistical poles)
        denominator = persona_pole_sim - topic_pole_sim
        
        # Exception handling for highly overlapping poles.
        # If the persona vocabulary heavily overlaps the topic vocabulary, the distance 
        # is statistically insignificant. Steering should collapse to 0.0 to prevent 
        # artificially massive inflated values.
        if abs(denominator) < 1e-4:
            return {'implicit_steering': 0.0, 'explicit_steering': 0.0, 'delta_steering': 0.0}
        
        # Normalize the projections so 0 is neutral and 1 is full persona caricature
        implicit_steering = (implicit_sim - topic_pole_sim) / denominator
        explicit_steering = (explicit_sim - topic_pole_sim) / denominator
        
        return {
            'implicit_steering': float(implicit_steering),
            'explicit_steering': float(explicit_steering),
            'delta_steering': float(explicit_steering - implicit_steering)
        }