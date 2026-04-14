"""
Semantic Masking Module
Implements redaction to obscure race, gender, and occupation labels before embedding generation.

This forces the classifier to rely on deep stylistic variations rather than 
explicit textual announcements of identity.

Usage:
python semantic_masking.py \
    --input data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations.json \
    --output data/transcripts/Llama-3.1-70B-Instruct-AWQ-INT4/target_simulations_masked.json \
    --model "/n/fs/dynamicbias/models/Llama-3.1-8B-Instruct"
"""

import argparse
import orjson
import logging
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_prompt(tokenizer, text: str) -> str:
    """Formats the prompt using the model's chat template."""
    system_prompt = (
        "You are a strict linguistic redaction tool. Your ONLY job is to output the redacted text. "
        "Do not add introductions, explanations, or conversational filler."
    )
    user_prompt = (
        "Analyze the following text. Replace any noun, phrase, or contextual hint that explicitly "
        "identifies the speaker's race, gender, occupation, or specific socio-economic status with "
        "the token [MASKED]. Preserve the syntax, emotion, and pragmatic structure perfectly.\n\n"
        f"Text to redact: '{text}'"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def process_file(input_path: str, output_path: str, model_name: str):
    logger.info(f"Loading data from {input_path}...")
    with open(input_path, "rb") as f:
        data = orjson.loads(f.read())

    # 1. Initialize vLLM Engine and Tokenizer
    logger.info(f"Initializing vLLM engine for {model_name}...")
    llm = LLM(model=model_name, max_model_len=8192)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

    # 2. Extract texts and build batch prompts
    logger.info("Extracting turns for redaction...")
    turns_to_process = []
    prompts = []

    for d in data:
        for turn in d.get("transcript", []):
            original_text = turn.get("content", "")
            if original_text.strip():
                turns_to_process.append(turn)
                prompts.append(build_prompt(tokenizer, original_text))

    # 3. Run Batch Inference
    logger.info(f"Running batch inference on {len(prompts)} texts.")
    outputs = llm.generate(prompts, sampling_params)

    # 4. Map outputs back to the JSON structure
    logger.info("Applying masked text to JSON structure...")
    for turn, output in zip(turns_to_process, outputs):
        original_text = turn.get("content", "")
        masked_text = output.outputs[0].text.strip()

        # Fallback in case the LLM outputs an empty string
        if not masked_text:
            masked_text = original_text

        turn["content_unmasked"] = original_text
        turn["content"] = masked_text
        turn["masking_applied"] = original_text != masked_text

    # 5. Save the modified data
    logger.info(f"Saving masked data to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM Semantic Masking.")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model ID or local path"
    )

    args = parser.parse_args()
    process_file(args.input, args.output, args.model)
    logger.info("Semantic Masking complete.")
