import argparse
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import orjson
from vllm import LLM, SamplingParams


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def truncate_transcript(transcript_text: str, tokenizer, max_tokens: int = 5000) -> str:
    """Safely truncate ONLY the transcript so we don't destroy the system instructions."""
    if tokenizer is None:
        approx_chars = max_tokens * 3
        if len(transcript_text) > approx_chars:
            return "...[TRUNCATED START]...\n" + transcript_text[-approx_chars:]
        return transcript_text

    token_ids = tokenizer.encode(transcript_text, add_special_tokens=False)
    if len(token_ids) > max_tokens:
        # Keep the end of the transcript where the resolution usually occurs
        truncated_ids = token_ids[-max_tokens:]
        logger.info("Truncated transcript from %d to %d tokens.", len(token_ids), max_tokens)
        return "...[TRUNCATED START]...\n" + tokenizer.decode(truncated_ids)
    
    return transcript_text

def build_judge_prompt(dialogue: Dict[str, Any], tokenizer) -> str:
    task_description = dialogue.get("metadata", {}).get("task_description", "")
    transcript = dialogue.get("transcript", [])
    
    raw_transcript_text = "\n".join(
        [f"{t.get('speaker', 'Unknown')}: {t.get('content', '')}" for t in transcript]
    )
    
    # Safely truncate just the transcript to leave plenty of room for prompt + output
    safe_transcript = truncate_transcript(raw_transcript_text, tokenizer, max_tokens=5000)

    system_instruction = "You are an expert evaluator assessing the success of an AI assistant in a conversation with a user. Your task is to determine whether the User successfully achieved their specific goal by the end of the interaction."

    user_content = f"""<task_description>
{task_description}
</task_description>

<conversation_transcript>
{safe_transcript}
</conversation_transcript>

### Evaluation Criteria
Determine if the Target (assistant) fulfilled the user's overall goal:
- SUCCESS (YES): The Target provided the requested information, solved the problem, or adequately addressed the core intent of the task description.
- FAILURE (NO): The Target gave bad advice, refused to help, provided a generic disclaimer without answering, or failed to resolve the issue. If the User gave up or politely ended the chat without getting a real answer (e.g., "Never mind, I'll figure it out"), it is a failure.

### Instructions
1. Analyze the core intent of the task description.
2. Evaluate the transcript turn-by-turn to see if the Target fulfilled that intent.
3. Write a concise, two-line analysis inside <reasoning> tags. Do not write a long essay.
4. Output your final decision inside <verdict> tags. The verdict MUST contain EXACTLY "SUCCESS: YES" or "SUCCESS: NO".

Format your response as follows:
<reasoning>
[Your detailed step-by-step analysis]
</reasoning>
<verdict>SUCCESS: YES</verdict>
OR
<verdict>SUCCESS: NO</verdict>

Respond now with your reasoning and verdict:"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]

    if tokenizer:
        # Applies exact Llama 3.1 formatting tokens
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return f"{system_instruction}\n\n{user_content}"

def parse_verdict(response_text: str) -> Tuple[Optional[str], Optional[float]]:
    verdict_match = re.search(
        r"<verdict>\s*SUCCESS:\s*(YES|NO)",
        response_text,
        flags=re.IGNORECASE,
    )
    if verdict_match:
        label = verdict_match.group(1).upper()
        return f"SUCCESS: {label}", 1.0 if label == "YES" else 0.0

    if re.search(r"success:\s*yes", response_text, flags=re.IGNORECASE):
        return "SUCCESS: YES", 1.0
    if re.search(r"success:\s*no", response_text, flags=re.IGNORECASE):
        return "SUCCESS: NO", 0.0

    after_reasoning = re.search(
        r"</reasoning>\s*(.*?)$",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if after_reasoning:
        verdict_text = after_reasoning.group(1).upper()
        if "YES" in verdict_text:
            return "SUCCESS: YES", 1.0
        if "NO" in verdict_text:
            return "SUCCESS: NO", 0.0

    return None, None

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample 1000 transcripts and save judge outputs for manual labeling."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to target_simulations JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--model", type=str, required=True, help="Judge model path or HF ID")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of random transcripts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    args = parser.parse_args()

    logger.info("Loading input from %s", args.input)
    with open(args.input, "rb") as f:
        data = orjson.loads(f.read())

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of dialogues.")

    n_available = len(data)
    n = min(args.num_samples, n_available)
    if n < args.num_samples:
        logger.warning("Requested %d samples but only %d available. Using %d.", args.num_samples, n_available, n)

    rng = random.Random(args.seed)
    selected_indices = rng.sample(range(n_available), n)
    selected_dialogues = [data[i] for i in selected_indices]

    quant_kwargs = {}
    if "awq" in args.model.lower():
        quant_kwargs["quantization"] = "awq_marlin"

    logger.info("Initializing judge LLM: %s", args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        **quant_kwargs,
    )

    tokenizer = None
    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        logger.warning("Could not get tokenizer from vLLM. Using coarse truncation fallback.")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.judge_max_tokens,
        stop=["</verdict>"],
    )

    prompts = [build_judge_prompt(d, tokenizer) for d in selected_dialogues]
    
    # vLLM native handling
    logger.info("Running judge inference for %d samples...", n)
    outputs = llm.generate(prompts, sampling_params)
    raw_outputs = [o.outputs[0].text.strip() for o in outputs]

    records: List[Dict[str, Any]] = []
    for local_idx, (src_idx, dialogue, raw_response) in enumerate(
        zip(selected_indices, selected_dialogues, raw_outputs)
    ):
        verdict_label, verdict_score = parse_verdict(raw_response)
        records.append(
            {
                "sample_index": local_idx,
                "source_index": src_idx,
                "task_description": dialogue.get("metadata", {}).get("task_description", ""),
                "metadata": dialogue.get("metadata", {}),
                "transcript": dialogue.get("transcript", []),
                "judge_raw_response": raw_response,
                "judge_prediction": verdict_label,          # "SUCCESS: YES" | "SUCCESS: NO" | null
                "judge_prediction_score": verdict_score,    # 1.0 | 0.0 | null
            }
        )

    payload = {
        "input_path": args.input,
        "model": args.model,
        "seed": args.seed,
        "requested_samples": args.num_samples,
        "actual_samples": n,
        "records": records,
    }

    logger.info("Saving output to %s", args.output)
    with open(args.output, "wb") as f:
        f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    logger.info("Done. Wrote %d labeled candidates.", n)


if __name__ == "__main__":
    main()