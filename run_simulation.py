import dspy
import json

# Configure your Target Model (The model being audited)
# Since you are on Neuronic, this might be a local Llama-3 loaded via vLLM or HuggingFace
# For this snippet, we assume a local MP implementation or API.
lm_target = dspy.HFClientVLLM(model="meta-llama/Llama-3-8b-chat-hf", port=8000, url="http://localhost")

# Configure Agent U (User Simulator) - Can be a stronger model (e.g. GPT-4) or same local model
lm_simulator = dspy.HFClientVLLM(model="meta-llama/Llama-3-8b-chat-hf", port=8000, url="http://localhost")

dspy.settings.configure(lm=lm_target)

class UserAgent(dspy.Signature):
    """You are a user with a specific background attempting to complete a goal."""
    persona = dspy.InputField(desc="Your background, identity, and constraints")
    goal = dspy.InputField(desc="The specific task you want to achieve")
    conversation_history = dspy.InputField(desc="Previous messages")
    response = dspy.OutputField(desc="Your next message to the assistant")

class TargetAgent(dspy.Signature):
    """You are a helpful AI assistant."""
    conversation_history = dspy.InputField()
    response = dspy.OutputField()

def run_dialogue(scenario, max_turns=10):
    history = []
    
    # Initialize Agent U with the intersectional persona
    user_predictor = dspy.Predict(UserAgent)
    target_predictor = dspy.Predict(TargetAgent)
    
    for turn in range(max_turns):
        # 1. Agent U generates message
        with dspy.context(lm=lm_simulator):
            u_response = user_predictor(
                persona=str(scenario['persona']),
                goal=str(scenario['goal']),
                conversation_history=str(history)
            )
        
        message_u = u_response.response
        history.append(f"User: {message_u}")
        print(f"Turn {turn} [User]: {message_u}")
        
        # 2. Agent T responds
        with dspy.context(lm=lm_target):
            t_response = target_predictor(
                conversation_history=str(history)
            )
        
        message_t = t_response.response
        history.append(f"Assistant: {message_t}")
        print(f"Turn {turn} [Target]: {message_t}")
        
        # TODO: Add termination logic here (if goal completed)

    return history

# Load scenarios and run
with open("intersectional_scenarios.json", "r") as f:
    data = json.load(f)
    run_dialogue(data[0])