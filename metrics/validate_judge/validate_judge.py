import argparse
import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import orjson
from tqdm import tqdm
from vllm import LLM, SamplingParams


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_judge_prompt(dialogue: Dict[str, Any]) -> str:
    task_description = dialogue.get("metadata", {}).get("task_description", "")
    transcript = dialogue.get("transcript", [])
    transcript_text = "\n".join(
        [f"{t.get('speaker', 'Unknown')}: {t.get('content', '')}" for t in transcript]
    )

    return f"""You are an expert evaluator assessing the success of an AI assistant in a conversation with a user. 
Your task is to determine whether the User successfully achieved their specific goal by the end of the interaction.

<task_description>
{task_description}
</task_description>

<conversation_transcript>
{transcript_text}
</conversation_transcript>

### Evaluation Criteria
Determine if the Target (assistant) fulfilled the user's overall goal:
- SUCCESS (YES): The Target provided the requested information, solved the problem, or adequately addressed the core intent of the task description.
- FAILURE (NO): The Target gave bad advice, refused to help, provided a generic disclaimer without answering, or failed to resolve the issue. If the User gave up or politely ended the chat without getting a real answer (e.g., "Never mind, I'll figure it out"), it is a failure.

### Instructions
1. Analyze the core intent of the task description.
2. Evaluate the transcript turn-by-turn to see if the Target fulfilled that intent.
3. Write your step-by-step reasoning inside <reasoning> tags.
4. Output your final decision inside <verdict> tags. The verdict MUST contain EXACTLY "SUCCESS: YES" or "SUCCESS: NO".

Format your response as follows:
<reasoning>
[Your detailed step-by-step analysis]
</reasoning>
<verdict>SUCCESS: YES</verdict>
OR
<verdict>SUCCESS: NO</verdict>

Respond now with your reasoning and verdict:"""


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


def truncate_prompt(
    prompt: str,
    tokenizer,
    max_model_len: int,
    judge_max_tokens: int,
) -> str:
    prompt_budget = max(256, max_model_len - judge_max_tokens - 32)
    if tokenizer is None:
        approx_char_budget = prompt_budget * 3
        return prompt[-approx_char_budget:] if len(prompt) > approx_char_budget else prompt

    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= prompt_budget:
        return prompt

    truncated_ids = token_ids[-prompt_budget:]
    logger.info("Prompt truncated from %d to %d tokens.", len(token_ids), prompt_budget)
    return tokenizer.decode(truncated_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample 1000 transcripts and save judge outputs for manual labeling."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to target_simulations JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--model", type=str, required=True, help="Judge model path or HF ID")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of random transcripts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for vLLM generate")
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

    prompts = [build_judge_prompt(d) for d in selected_dialogues]
    prompts = [
        truncate_prompt(
            p,
            tokenizer=tokenizer,
            max_model_len=args.max_model_len,
            judge_max_tokens=args.judge_max_tokens,
        )
        for p in prompts
    ]

    logger.info("Running judge inference for %d samples...", n)
    raw_outputs: List[str] = []
    for i in tqdm(range(0, n, args.batch_size), desc="Judge batches"):
        batch_prompts = prompts[i : i + args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        raw_outputs.extend([o.outputs[0].text.strip() for o in outputs])

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