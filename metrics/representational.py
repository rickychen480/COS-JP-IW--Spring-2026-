import numpy as np
from typing import List, Dict


class RepresentationalEvaluator:
    def __init__(self, embedder_model: str = "all-MiniLM-L6-v2"):
        pass

    @staticmethod
    def _cosine_similarities(vectors: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """Computes cosine similarity between many vectors and one axis."""
        vecs = np.asarray(vectors, dtype=float)
        ref = np.asarray(axis, dtype=float)

        if vecs.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if ref.ndim == 1:
            ref = ref.reshape(1, -1)

        vec_norm = np.linalg.norm(vecs, axis=1)
        ref_norm = np.linalg.norm(ref, axis=1)[0]
        denom = vec_norm * ref_norm

        sims = np.full(vecs.shape[0], np.nan, dtype=float)
        valid = denom > 0
        if not np.any(valid):
            return sims

        dots = np.dot(vecs[valid], ref[0])
        sims[valid] = dots / denom[valid]
        return sims

    @staticmethod
    def _normalize_and_clip_steering(
        similarities: np.ndarray, topic_pole_sim: float, persona_pole_sim: float
    ) -> np.ndarray:
        """Normalizes cosine similarities onto [0, 1] steering scale."""
        sims = np.asarray(similarities, dtype=float)
        denominator = persona_pole_sim - topic_pole_sim
        if abs(denominator) < 1e-8:
            return np.full_like(sims, np.nan, dtype=float)

        steer = (sims - topic_pole_sim) / denominator
        return np.clip(steer, 0.0, 1.0)

    @staticmethod
    def _mean_cosine_similarity(vectors: np.ndarray, axis: np.ndarray) -> float:
        """Computes mean cosine similarity between many vectors and one axis."""
        sims = RepresentationalEvaluator._cosine_similarities(vectors, axis)
        if not np.isfinite(sims).any():
            return np.nan
        return float(np.nanmean(sims))

    def calculate_semantic_steering_trajectory(
        self,
        implicit_turn_embeddings: List[np.ndarray],
        explicit_turn_embeddings: List[np.ndarray],
        axis_v: np.ndarray,
        topic_pole_sim: float,
        persona_pole_sim: float,
    ) -> Dict[str, List[float]]:
        """
        Computes turn-indexed semantic steering trajectories.

        Each item in implicit_turn_embeddings / explicit_turn_embeddings corresponds
        to one dialogue with shape [num_target_turns, emb_dim]. For each turn index,
        similarities are averaged across dialogues that have that turn.
        """

        def aggregate_turn_sims(dialogue_turn_embs: List[np.ndarray]) -> np.ndarray:
            per_dialogue_sims = []
            max_turns = 0

            for emb in dialogue_turn_embs:
                arr = np.asarray(emb, dtype=float)
                if arr.size == 0:
                    continue
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.ndim != 2:
                    raise ValueError("each dialogue turn embedding must be a 2D array")

                sims = self._cosine_similarities(arr, axis_v)
                per_dialogue_sims.append(sims)
                max_turns = max(max_turns, sims.shape[0])

            if max_turns == 0:
                return np.array([], dtype=float)

            turn_means = np.full(max_turns, np.nan, dtype=float)
            for i in range(max_turns):
                vals = [s[i] for s in per_dialogue_sims if i < len(s) and np.isfinite(s[i])]
                if vals:
                    turn_means[i] = float(np.mean(vals))
            return turn_means

        axis_v = np.asarray(axis_v, dtype=float).reshape(1, -1)
        imp_turn_sims = aggregate_turn_sims(implicit_turn_embeddings)
        exp_turn_sims = aggregate_turn_sims(explicit_turn_embeddings)

        imp_traj = self._normalize_and_clip_steering(
            imp_turn_sims, topic_pole_sim, persona_pole_sim
        )
        exp_traj = self._normalize_and_clip_steering(
            exp_turn_sims, topic_pole_sim, persona_pole_sim
        )

        max_len = max(len(imp_traj), len(exp_traj))
        delta_traj = np.full(max_len, np.nan, dtype=float)
        for i in range(max_len):
            imp_val = imp_traj[i] if i < len(imp_traj) else np.nan
            exp_val = exp_traj[i] if i < len(exp_traj) else np.nan
            if np.isfinite(imp_val) and np.isfinite(exp_val):
                delta_traj[i] = exp_val - imp_val

        return {
            "implicit_trajectory": imp_traj.astype(float).tolist(),
            "explicit_trajectory": exp_traj.astype(float).tolist(),
            "delta_trajectory": delta_traj.astype(float).tolist(),
        }

    def calculate_confidence(self, target_logprobs: List[float]) -> float:
        """
        Calculates the average log-likelihood (confidence) of user-referential tokens.
        Requires logprobs directly from the target LLM inference engine.
        """
        valid_logprobs = [
            lp for lp in target_logprobs if lp is not None and not np.isnan(lp)
        ]
        if not valid_logprobs:
            return np.nan
        return float(np.mean(valid_logprobs))

    def calculate_d_ccd(
        self, implicit_logprobs: List[float], explicit_logprobs: List[float]
    ) -> float:
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
        persona_pole_sim: float,
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
            return {
                "implicit_steering": np.nan,
                "explicit_steering": np.nan,
                "delta_steering": np.nan,
            }

        axis_v = axis_v.reshape(1, -1)
        implicit_sim = self._mean_cosine_similarity(implicit_target_embeddings, axis_v)
        explicit_sim = self._mean_cosine_similarity(explicit_target_embeddings, axis_v)

        if np.isnan(implicit_sim) or np.isnan(explicit_sim):
            return {
                "implicit_steering": np.nan,
                "explicit_steering": np.nan,
                "delta_steering": np.nan,
            }

        # Exception handling for highly overlapping poles.
        if abs(persona_pole_sim - topic_pole_sim) < 1e-8:
            return {
                "implicit_steering": np.nan,
                "explicit_steering": np.nan,
                "delta_steering": np.nan,
            }

        implicit_steering = self._normalize_and_clip_steering(
            np.array([implicit_sim], dtype=float), topic_pole_sim, persona_pole_sim
        )[0]
        explicit_steering = self._normalize_and_clip_steering(
            np.array([explicit_sim], dtype=float), topic_pole_sim, persona_pole_sim
        )[0]

        return {
            "implicit_steering": float(implicit_steering),
            "explicit_steering": float(explicit_steering),
            "delta_steering": float(explicit_steering - implicit_steering),
        }
