import json
import argparse
import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def check_early_stopping(msg_u, msg_t):
    """Heuristic to detect natural conversation ends, dead ends, and model refusals."""
    msg_u_lower = msg_u.lower()
    msg_t_lower = msg_t.lower()

    # Standard Goodbye Phrases (User or Target)
    end_phrases = [
        "bye",
        "goodbye",
        "have a great day",
        "hasta luego",
        "have a good one",
        "adiós",
        "take care",
    ]
    if any(phrase in msg_u_lower for phrase in end_phrases) or any(
        phrase in msg_t_lower for phrase in end_phrases
    ):
        return True

    # Hard Refusals from the Target Model (Safety Filters / Gatekeeping)
    refusal_phrases = [
        "i can't help with that request",
        "i cannot fulfill",
        "i am unable to provide",
        "as an ai",
        "i'm sorry, but i cannot",
    ]
    if any(phrase in msg_t_lower for phrase in refusal_phrases):
        return True

    # User Simulator 'Dead End' or Frustration Detection
    dead_end_phrases = [
        "dead end",
        "not getting anywhere",
        "getting nowhere",
        "going in circles",
        "wasting my time",
        "forget it",
    ]
    if any(phrase in msg_u_lower for phrase in dead_end_phrases):
        return True

    # Natural Resolution (User is satisfied and leaving)
    resolution_phrases = [
        "thanks for trying to help",
        "thanks for your help",
        "i'll go ahead and do that now",
        "appreciate your help",
        "you're welcome",
    ]
    if any(
        msg_u_lower.endswith(phrase)
        or msg_u_lower.endswith(phrase + ".")
        or msg_u_lower.endswith(phrase + "!")
        for phrase in resolution_phrases
    ):
        return True

    # Target asks if there's anything else, and user says no
    if any(word in msg_t_lower for word in ["anything else", "assist you"]) and any(
        word in msg_u_lower
        for word in ["no", "nope", "i'm good", "that's all", "nothing else"]
    ):
        return True

    return False


def run_simulation(input_file, output_file, model_path, max_turns=10, limit=None, tp=1, quant=None, chunk_index=0, total_chunks=1):
    # LOAD DATA
    with open(input_file, "r") as f:
        scenarios = json.load(f)

    # Calculate chunk boundaries
    chunk_size = math.ceil(len(scenarios) / total_chunks)
    start_idx = chunk_index * chunk_size
    end_idx = min(start_idx + chunk_size, len(scenarios))
    scenarios = scenarios[start_idx:end_idx]
    
    print(f"Processing chunk {chunk_index}/{total_chunks}: scenarios {start_idx}-{end_idx-1}")

    # Allow running a small slice for testing
    if limit:
        scenarios = scenarios[:limit]
        print(f"TEST MODE: Running only {len(scenarios)} dialogues.")

    # INITIALIZE ENGINE (Loads model once for both Agents)
    print(f"Loading model: {model_path}...")
    llm = LLM(
        model=model_path, 
        tensor_parallel_size=tp, 
        quantization=quant,
        max_model_len=8192
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    sampling_params_u = SamplingParams(temperature=0.7, max_tokens=512)
    sampling_params_t = SamplingParams(temperature=0.7, max_tokens=512, logprobs=1)

    # INITIALIZE STATE
    active_dialogues = []
    for sc in scenarios:
        active_dialogues.append(
            {
                "dialogue_id": sc["dialogue_id"],
                "variant_type": sc.get("variant_type", "Unknown"),
                "meta": sc["metadata"],
                "history_u": [{"role": "system", "content": sc["user_agent"]["system_prompt"]}],
                "history_t": [{"role": "system", "content": sc["audit_target"]["system_prompt"]}],
                "transcript": [],
                "done": False,
            }
        )

    # DUAL AGENT LOOP
    for turn in range(max_turns):
        print(f"--- Turn {turn + 1}/{max_turns} ---")

        # --- AGENT U (User Simulator) MOVES ---
        batch_u = [d for d in active_dialogues if not d["done"]]
        if not batch_u:
            break

        prompts_u = [
            tokenizer.apply_chat_template(
                d["history_u"], tokenize=False, add_generation_prompt=True
            )
            for d in batch_u
        ]

        # Batch Generate
        outputs_u = llm.generate(prompts_u, sampling_params_u)

        for d, out in zip(batch_u, outputs_u):
            msg_u = out.outputs[0].text.strip()
            d["history_u"].append({"role": "assistant", "content": msg_u})
            d["history_t"].append({"role": "user", "content": msg_u})
            d["transcript"].append({"turn": turn, "speaker": "User", "content": msg_u})

        # --- AGENT T (Target Model) MOVES ---
        prompts_t = [
            tokenizer.apply_chat_template(
                d["history_t"], tokenize=False, add_generation_prompt=True
            )
            for d in batch_u
        ]
        outputs_t = llm.generate(prompts_t, sampling_params_t)

        for d, out in zip(batch_u, outputs_t):
            msg_t = out.outputs[0].text.strip()
            
            # EXTRACT TARGET LOGPROBS FOR d-CCD METRIC
            if out.outputs[0].logprobs:
                persona_name = d["meta"]["persona"].get("name", "").lower()
                
                # Base referential words
                ref_words = {"you", "your", "yours", "you're", "you've", "you'll", "you'd"}
                if persona_name:
                    ref_words.add(persona_name)
                
                for token_dict in out.outputs[0].logprobs:
                    for tok_id, logprob_obj in token_dict.items():
                        tok_text = logprob_obj.decoded_token
                        if tok_text is not None:
                            # Clean the token text to match against our reference list
                            clean_tok = tok_text.strip().lower()
                            clean_tok = ''.join(c for c in clean_tok if c.isalpha() or c == "'")
                            
                            if clean_tok in ref_words:
                                # Append the log-likelihood (float) of the referential token
                                d["meta"].setdefault("target_logprobs", []).append(logprob_obj.logprob)

            last_msg_u = d["transcript"][-1]["content"]

            d["history_t"].append({"role": "assistant", "content": msg_t})
            d["history_u"].append({"role": "user", "content": msg_t})
            d["transcript"].append({"turn": turn, "speaker": "Target", "content": msg_t})

            if check_early_stopping(last_msg_u, msg_t):
                d["done"] = True

        # INCREMENTALLY SAVE RESULTS
        final_output = [
            {
                "dialogue_id": d["dialogue_id"], 
                "variant_type": d["variant_type"],
                "metadata": d["meta"], 
                "transcript": d["transcript"]
            }
            for d in active_dialogues
        ]

        with open(output_file, "w") as f:
            json.dump(final_output, f, indent=2)
    
    print(f"Simulation complete. Saved {len(final_output)} dialogues to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to input JSON")
    parser.add_argument("--out", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--quant", type=str, default=None, help="Quantization method, e.g., 'awq'")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (number of GPUs to split the model across)")
    parser.add_argument("--chunk_index", type=int, default=0, help="Index of the current chunk (from SLURM)")
    parser.add_argument("--total_chunks", type=int, default=1, help="Total number of parallel chunks")
    parser.add_argument(
        "--limit", type=int, default=None, help="Test mode: number of dialogues to run"
    )
    args = parser.parse_args()

    run_simulation(
        args.data, 
        args.out, 
        args.model, 
        max_turns=10, 
        limit=args.limit, 
        tp=args.tp, 
        quant=args.quant,
        chunk_index=args.chunk_index,
        total_chunks=args.total_chunks
    )