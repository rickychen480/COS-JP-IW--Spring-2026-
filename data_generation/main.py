import json
import random
from pathlib import Path

import constants as const
import generators as gen


NUM_SAMPLES = 10000
OUT_DIR = Path("data/prompts")


def main() -> None:
    target_file = OUT_DIR / "target_simulations.json"
    control_file = OUT_DIR / "control_simulations.json"
    default_topic_file = OUT_DIR / "default_topics.json"

    tasks_file = Path("data_generation/metalwoz/tasks.txt")
    tasks_list = []

    if tasks_file.exists():
        with tasks_file.open(encoding="utf-8") as f:
            tasks_list = [json.loads(line) for line in f if line.strip()]
    else:
        print(
            f"Warning: {tasks_file} not found. Please ensure the dataset is downloaded."
        )

    high_stakes_tasks = gen.filter_high_stakes_domains(tasks_list)
    print(
        f"Filtered {len(tasks_list)} tasks to {len(high_stakes_tasks)} high-stakes domain tasks"
    )

    if not high_stakes_tasks:
        print("No high-stakes tasks found. Exiting to avoid errors.")
        return

    target_simulations = []
    control_simulations = []

    for _ in range(NUM_SAMPLES):
        # Extract attributes from MetaLWOz (Goal Source) - from high-stakes domains
        mw_sample = random.choice(high_stakes_tasks)
        goal = gen.extract_goal_from_metalwoz(mw_sample)

        # Apply Identity Grid (Augmentation)
        demographic = random.choice(const.RACES)
        gender = random.choice(const.GENDERS)
        occupation = random.choice(const.OCCUPATIONS_GRID)

        task_variants = gen.generate_task_scenarios(goal, demographic, gender, occupation)
        for variant in task_variants:
            if variant["variant_type"] in ("implicit", "explicit"):
                target_simulations.append(variant)
            elif variant["variant_type"] == "control":
                control_simulations.append(variant)

    default_topics = gen.generate_default_topic_scenarios()

    with target_file.open("w", encoding="utf-8") as f:
        json.dump(target_simulations, f, indent=2)

    with control_file.open("w", encoding="utf-8") as f:
        json.dump(control_simulations, f, indent=2)
        
    with default_topic_file.open("w", encoding="utf-8") as f:
        json.dump(default_topics, f, indent=2)

    explicit_count = sum(1 for s in target_simulations if s["variant_type"] == "explicit")
    implicit_count = sum(1 for s in target_simulations if s["variant_type"] == "implicit")

    print(f"\nSuccessfully saved scenarios to {OUT_DIR}.")
    print("Dataset Breakdown:")
    print(f"  - target_simulations.json: {len(target_simulations)} total")
    print(f"      -> Explicit Variants: {explicit_count}")
    print(f"      -> Implicit Variants: {implicit_count}")
    print(f"  - control_simulations.json: {len(control_simulations)} total")
    print(f"  - default_topics.json: {len(default_topics)} total")


if __name__ == "__main__":
    main()
