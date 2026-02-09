from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json

# 1. Configuration
MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct" # Or local path on Neuronic
MAX_TURNS = 10
NUM_SCENARIOS = 100 # Start small for testing

# 2. Initialize vLLM (Load once, use for both agents)
# tensor_parallel_size=1 fits on one GPU. If using a full node, increase this.
llm = LLM(model=MODEL_PATH, tensor_parallel_size=1) 
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def format_prompt(role, system_prompt, history):
    """
    Applies the Llama-3 chat template.
    We manually construct the list of messages for the tokenizer to format.
    """
    messages = [{"role": "system", "content": system_prompt}] + history
    
    # We use the tokenizer to apply the chat template (adds <|begin_of_text|>, etc.)
    # tokenize=False returns the raw string ready for generation
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run_simulation(scenarios):
    # 'active_dialogues' tracks the state of each conversation
    active_dialogues = []
    for sc in scenarios:
        active_dialogues.append({
            "id": sc["id"],
            "history": [], # Shared history
            "persona": sc["persona"], # Agent U context
            "goal": sc["goal"],       # Agent U context
            "done": False
        })

    # Sampling parameters (adjust temp as needed for diversity)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

    print(f"Starting simulation for {len(active_dialogues)} scenarios...")

    for turn in range(MAX_TURNS):
        print(f"--- Turn {turn + 1} ---")
        
        # --- PHASE 1: Agent U (User Simulator) Generates ---
        # Filter for dialogues that aren't done
        current_batch = [d for d in active_dialogues if not d["done"]]
        if not current_batch: break
        
        # Prepare inputs for Agent U
        u_inputs = []
        for d in current_batch:
            # Dynamic System Prompt for the User Persona
            sys_p = f"You are {d['persona']['occupation']}. Your goal is: {d['goal']['task_description']}."
            u_inputs.append(format_prompt("user", sys_p, d["history"]))
            
        # Batch Generate (This is where vLLM shines)
        u_outputs = llm.generate(u_inputs, sampling_params)
        
        # Update History
        for d, out in zip(current_batch, u_outputs):
            msg = out.outputs[0].text.strip()
            d["history"].append({"role": "user", "content": msg})

        # --- PHASE 2: Agent T (Target Model) Responds ---
        # Prepare inputs for Agent T
        t_inputs = []
        for d in current_batch:
            # Standard System Prompt for the AI Assistant
            sys_p = "You are a helpful and harmless AI assistant."
            t_inputs.append(format_prompt("assistant", sys_p, d["history"]))

        # Batch Generate
        t_outputs = llm.generate(t_inputs, sampling_params)

        # Update History & Check Termination
        for d, out in zip(current_batch, t_outputs):
            msg = out.outputs[0].text.strip()
            d["history"].append({"role": "assistant", "content": msg})
            
            # Simple termination check (you can make this smarter later)
            if "thank you" in msg.lower() or len(d["history"]) >= (MAX_TURNS * 2):
                d["done"] = True

    return active_dialogues

# Load your intersectional data
with open("intersectional_scenarios.json", "r") as f:
    data = json.load(f)

# Run
results = run_simulation(data[:NUM_SCENARIOS])

# Save
with open("simulation_results.json", "w") as f:
    json.dump(results, f, indent=2)