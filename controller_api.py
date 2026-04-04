"""
Orchestrate the dual agent transcript generation w/ an LLM API.

Usage:
python controller_api.py \
  --data data/prompts/target_simulations.json \
  --out data/transcripts/gpt-4o-mini/target_simulations.json \
  --concurrency 10

python controller_api.py \
  --data data/prompts/control_simulations.json \
  --out data/transcripts/gpt-4o-mini/control_simulations.json \
  --concurrency 10

python controller_api.py \
  --data data/prompts/default_topics.json \
  --out data/transcripts/gpt-4o-mini/default_topics.json \
  --concurrency 10
"""

import json
import argparse
import asyncio
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# LOAD THE SECRET KEY
key_path = os.path.expanduser("~/.openai_key")
if os.path.exists(key_path):
    load_dotenv(key_path)
else:
    print(f"Warning: Secret key file not found at {key_path}")

# Initialize the async client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

async def generate_with_retry(client, messages, model, max_retries=6, **kwargs):
    """Helper function to retry API calls with exponential backoff for various network/rate errors."""
    retryable_errors = ["429", "rate limit", "500", "502", "503", "504", "timeout", "connection"]
    
    for attempt in range(max_retries):
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
        except Exception as e:
            error_str = str(e).lower()
            if any(err in error_str for err in retryable_errors):
                wait_time = 2 ** attempt  # Waits 1s, 2s, 4s, 8s, 16s, 32s
                if attempt > 0:
                    print(f"Network/Rate limit hit. Pausing for {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                # If it's a fatal error (e.g., bad API key, invalid request format), do not retry
                raise e
    
    raise Exception(f"Failed after {max_retries} retries due to persistent API errors.")

async def run_single_dialogue(sc, max_turns, semaphore, model="gpt-4o-mini"):
    """Runs a single dual-agent simulation asynchronously."""
    async with semaphore:
        dialogue = {
            "dialogue_id": sc["dialogue_id"],
            "variant_type": sc.get("variant_type", "Unknown"),
            "metadata": sc["metadata"],
            "history_u": [
                {"role": "system", "content": sc["user_agent"]["system_prompt"]}
            ],
            "history_t": [
                {"role": "system", "content": sc["audit_target"]["system_prompt"]}
            ],
            "transcript": [],
            "done": False,
        }

        # DUAL AGENT LOOP
        for turn in range(max_turns):
            # --- AGENT U (User Simulator) ---
            try:
                response_u = await generate_with_retry(
                    client, dialogue["history_u"], model, 
                    temperature=0.65, top_p=0.90, max_tokens=512
                )
                msg_u = response_u.choices[0].message.content.strip()
            except Exception as e:
                # LOGGING UPDATE: Return an explicit failure object
                return {
                    "status": "failed",
                    "dialogue_id": dialogue["dialogue_id"],
                    "agent": "User Simulator",
                    "error": str(e),
                    "failed_at_turn": turn
                }

            dialogue["history_u"].append({"role": "assistant", "content": msg_u})
            dialogue["history_t"].append({"role": "user", "content": msg_u})
            dialogue["transcript"].append({"turn": turn, "speaker": "User", "content": msg_u})

            # --- AGENT T (Target Model) ---
            try:
                response_t = await generate_with_retry(
                    client, dialogue["history_t"], model, 
                    temperature=0.7, max_tokens=512, logprobs=True, top_logprobs=1
                )
                msg_t = response_t.choices[0].message.content.strip()
                
                # EXTRACT TARGET LOGPROBS FOR d-CCD METRIC
                if response_t.choices[0].logprobs and response_t.choices[0].logprobs.content:
                    persona_name = dialogue["metadata"]["persona"].get("name", "").lower()

                    # Base referential words
                    ref_words = {
                        "you",
                        "your",
                        "yours",
                        "you're",
                        "you've",
                        "you'll",
                        "you'd",
                    }
                    if persona_name:
                        ref_words.add(persona_name)
                    
                    for token_info in response_t.choices[0].logprobs.content:
                        clean_tok = token_info.token.strip().lower()
                        clean_tok = "".join(
                            c for c in clean_tok if c.isalpha() or c == "'"
                        )
                        
                        if clean_tok in ref_words:
                            # Append the log-likelihood (float) of the referential token
                            dialogue["metadata"].setdefault("target_logprobs", []).append(
                                token_info.logprob
                            )

            except Exception as e:
                # LOGGING UPDATE: Return an explicit failure object
                return {
                    "status": "failed",
                    "dialogue_id": dialogue["dialogue_id"],
                    "agent": "Target Model",
                    "error": str(e),
                    "failed_at_turn": turn
                }

            dialogue["history_t"].append({"role": "assistant", "content": msg_t})
            dialogue["history_u"].append({"role": "user", "content": msg_t})
            dialogue["transcript"].append({"turn": turn, "speaker": "Target", "content": msg_t})

            # Check early stopping
            if check_early_stopping(msg_u, msg_t):
                break

        return {
            "status": "success",
            "dialogue_id": dialogue["dialogue_id"],
            "variant_type": dialogue["variant_type"],
            "metadata": dialogue["metadata"],
            "transcript": dialogue["transcript"]
        }

async def run_simulation_async(input_file, output_file, model, max_turns, limit, concurrency):
    # Setup Checkpoint and Error Log Files
    checkpoint_file = output_file.replace(".json", ".jsonl")
    error_file = output_file.replace(".json", "_errors.jsonl")
    
    if not checkpoint_file.endswith(".jsonl"):
        checkpoint_file += ".jsonl"
    if not error_file.endswith(".jsonl"):
        error_file += ".jsonl"

    completed_ids = set()
    results = []

    # Check for existing checkpoint to resume
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            for line_number, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        completed_ids.add(data["dialogue_id"])
                        
                        # Strip out the 'status' key we used internally before appending to final results
                        if "status" in data:
                            del data["status"]
                            
                        results.append(data)
                    except json.JSONDecodeError:
                        print(f"Warning: Corrupted JSON on line {line_number} of checkpoint. Skipping.")
                        continue
        print(f"Resuming: Loaded {len(completed_ids)} completed dialogues from checkpoint.")

    # Load and filter scenarios
    with open(input_file, "r") as f:
        all_scenarios = json.load(f)

    if limit:
        all_scenarios = all_scenarios[:limit]

    pending_scenarios = [sc for sc in all_scenarios if sc["dialogue_id"] not in completed_ids]
    
    print(f"Found {len(all_scenarios)} total scenarios. {len(pending_scenarios)} remaining to process.")

    if not pending_scenarios:
        print("All scenarios are already completed! Skipping to final JSON compilation.")
    else:
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [run_single_dialogue(sc, max_turns, semaphore, model) for sc in pending_scenarios]
        
        # Process tasks as they complete (Write-Through)
        with open(checkpoint_file, "a", encoding="utf-8") as f_ckpt, \
             open(error_file, "a", encoding="utf-8") as f_err:
             
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Simulating Dialogues"):
                try:
                    result = await f
                    
                    if result["status"] == "failed":
                        # Write failures to the dedicated error log
                        f_err.write(json.dumps(result) + "\n")
                        f_err.flush()
                    else:
                        # Write successes to the main checkpoint
                        # Make a copy to avoid mutating the original before saving to checkpoint
                        save_data = dict(result)
                        del save_data["status"]
                        
                        f_ckpt.write(json.dumps(save_data) + "\n")
                        f_ckpt.flush()
                        results.append(save_data)
                        
                except Exception as e:
                    print(f"Task encountered an unhandled exception: {e}")

    # Atomic Final Save
    print(f"Compilation complete. Saving all {len(results)} successful dialogues to {output_file}...")
    temp_out = output_file + ".tmp"
    with open(temp_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    os.replace(temp_out, output_file) # Safely overwrites the target file in one atomic step
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to input JSON")
    parser.add_argument("--out", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None, help="Test mode: number of dialogues to run")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API calls")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    asyncio.run(run_simulation_async(
        args.data,
        args.out,
        args.model,
        max_turns=10,
        limit=args.limit,
        concurrency=args.concurrency
    ))
