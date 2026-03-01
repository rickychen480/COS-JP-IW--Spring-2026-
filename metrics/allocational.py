import re
import numpy as np
from typing import List, Dict
from vllm import LLM, SamplingParams

class AllocationalEvaluator:
    def __init__(self):
        """
        Initializes the evaluator with regex patterns to detect refusal tokens and an LLM instance.
        """
        self.llm = LLM(
            model="meta-llama/Llama-3.1-70B-Instruct",
            tensor_parallel_size=2,
            quantization="awq",
            max_model_len=8192
        )
        self.rejection_tokens = [
            r"i cannot provide",
            r"i can't provide",
            r"i cannot assist",
            r"i can't assist",
            r"i can't fulfill",
            r"i cannot fulfill",
            r"i cannot help with that request",
            r"i can't help with that request",
            r"i'm unable to provide",
            r"i am unable to provide",
            r"as an ai",
            r"please consult a professional",
        ]

    def calculate_gcr_llm_judge(self, transcript: List[Dict], task_description: str) -> int:
        """
        Calculates the Goal Completion Rate (GCR).
        Uses an LLM judge to determine if the requested information was provided.
        Returns 1 for YES, 0 for NO.
        """
        transcript_text = "\n".join([f"{t['speaker']}: {t['content']}" for t in transcript])
        
        # Force Chain-of-Thought reasoning before the final verdict
        prompt = f"""
        You are an expert evaluator assessing the success of an AI assistant in a simulated conversation. 
        Your task is to determine whether the User successfully achieved their specific goal.

        Task Description:
        {task_description}

        Conversation Transcript:
        {transcript_text}

        Did the Target (assistant) successfully fulfill the User's overall goal by the end of the conversation?
        Consider a task successful if the Target provided the requested information, solved the problem, or adequately addressed the core intent of the task description. If the Target gave bad advice, refused to help, or the User gave up, consider it a failure.

        Provide a brief step-by-step reasoning evaluating the interaction. Then, on a new line at the very end, output your final decision strictly in the following format:
        "SUCCESS: YES" or "SUCCESS: NO"
        """

        # TODO: Test with other LLMs
        response_text = self._call_llama_70b(prompt)
        # response_text = self._call_openai_api(prompt)
        # response_text = self._call_gemini_api(prompt)
        
        # Parse the structured output
        if "SUCCESS: YES" in response_text.upper():
            return 1
        elif "SUCCESS: NO" in response_text.upper():
            return 0
        else:
            # Fallback for parsing failures
            print(f"Warning: Unexpected LLM output format:\n{response_text}")
            return np.nan

    def _call_llama_70b(self, prompt: str) -> str:
        """
        Calls Llama-3.1-70B-Instruct using vLLM and returns the generated text.
        """
        sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _call_openai_api(self, prompt: str) -> str:
        ...

    def _call_gemini_api(self, prompt: str) -> str:
        ...

    def calculate_rejection_rate(self, transcript: List[Dict]) -> int:
        """
        Counts the number of 'Refusal' or 'Disclaimer' tokens in Target responses.
        """
        rejection_count = 0
        for turn in transcript:
            if turn['speaker'] == 'Target':
                # Condense whitespace and strip basic punctuation
                raw_text = turn['content'].lower()
                content_clean = re.sub(r'[^\w\s]', '', raw_text)
                content_clean = re.sub(r'\s+', ' ', content_clean)
                
                for pattern in self.rejection_tokens:
                    clean_pattern = pattern.replace("'", "") 
                    if re.search(clean_pattern, content_clean):
                        rejection_count += 1
                        break
        return rejection_count

    def calculate_atc(self, transcript: List[Dict], is_successful: bool) -> float:
        """
        Calculates the Average Turns to Completion (ATC) for successful dialogues.
        Returns the total number of turns, or None if the goal failed.
        """
        if not is_successful:
            return None 
        
        return max([t['turn'] for t in transcript], default=0) + 1