"""
python controller.py \
  --data intersectional_scenarios.json \
  --out pilot_results.json \
  --model /scratch/rc5898/hf_models/Llama-3.1-8B-Instruct \
  --limit 1

hf download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir /scratch/$USER/hf_models/Llama-3.1-8B-Instruct
"""

import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def run_simulation(input_file, output_file, model_path, max_turns=10, limit=None):
    # LOAD DATA
    with open(input_file, "r") as f:
        scenarios = json.load(f)
    
    # Allow running a small slice for testing
    if limit:
        scenarios = scenarios[:limit]
        print(f"TEST MODE: Running only {len(scenarios)} dialogues.")

    # INITIALIZE ENGINE (Loads model once for both Agents)
    print(f"Loading model: {model_path}...")
    llm = LLM(model=model_path, tensor_parallel_size=1) 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    # INITIALIZE STATE (Maintain a 'running state' for every dialogue in the batch)
    active_dialogues = []
    for sc in scenarios:
        active_dialogues.append({
            "id": sc["dialogue_id"],
            "meta": sc["metadata"],
            # Agent U's private history (System Prompt U + Conversation)
            "history_u": [{"role": "system", "content": sc["user_agent"]["system_prompt"]}],
            # Agent T's private history (System Prompt T + Conversation)
            "history_t": [{"role": "system", "content": sc["audit_target"]["system_prompt"]}],
            # The shared public transcript
            "transcript": [],
            "done": False
        })

    # DUAL AGENT LOOP
    for turn in range(max_turns):
        print(f"--- Turn {turn + 1}/{max_turns} ---")
        
        # --- AGENT U (User Simulator) MOVES ---
        # Filter for active dialogues
        batch_u = [d for d in active_dialogues if not d["done"]]
        if not batch_u: break
        
        # Prepare prompts for Agent U
        prompts_u = [
            tokenizer.apply_chat_template(d["history_u"], tokenize=False, add_generation_prompt=True)
            for d in batch_u
        ]
        
        # Batch Generate (High Throughput)
        outputs_u = llm.generate(prompts_u, sampling_params)
        
        # Update State
        for d, out in zip(batch_u, outputs_u):
            msg_u = out.outputs[0].text.strip()
            
            # Update U's own history (I said this)
            d["history_u"].append({"role": "assistant", "content": msg_u}) # 'assistant' role b/c model generated it
            
            # Update T's history (User said this)
            d["history_t"].append({"role": "user", "content": msg_u})
            
            # Log to public transcript
            d["transcript"].append({"turn": turn, "speaker": "User", "content": msg_u})

        # --- AGENT T (Target Model) MOVES ---
        # Prepare prompts for Agent T
        prompts_t = [
            tokenizer.apply_chat_template(d["history_t"], tokenize=False, add_generation_prompt=True)
            for d in batch_u
        ]
        
        outputs_t = llm.generate(prompts_t, sampling_params)
        
        for d, out in zip(batch_u, outputs_t):
            msg_t = out.outputs[0].text.strip()
            
            # Update T's own history (I said this)
            d["history_t"].append({"role": "assistant", "content": msg_t})
            
            # Update U's history (Target said this - acts as User input for next turn)
            d["history_u"].append({"role": "user", "content": msg_t})
            
            # Log
            d["transcript"].append({"turn": turn, "speaker": "Target", "content": msg_t})

            # Check if the User signaled the end
            if "bye" in msg_u.lower() or "have a good day" in msg_u.lower():
                d["done"] = True

            # Check if the Target signaled the end
            if "is there anything else" in msg_t.lower() and "no" in msg_u.lower():
                d["done"] = True

    # SAVE RESULTS
    # TODO: Verify format aligns with the paper's needs
    final_output = [
        {"id": d["id"], "metadata": d["meta"], "transcript": d["transcript"]} 
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
    parser.add_argument("--limit", type=int, default=None, help="Test mode: number of dialogues to run")
    args = parser.parse_args()
    
    run_simulation(args.data, args.out, args.model, limit=args.limit)