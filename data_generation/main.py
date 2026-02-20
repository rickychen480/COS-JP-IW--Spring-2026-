import json
import random
from pathlib import Path

import constants as const
import generators as gen


NUM_SAMPLES = 10000
OUT_FILE = Path("data/scenarios.json")


def main() -> None:
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

    scenarios = []
    for _ in range(NUM_SAMPLES):
        # Extract attributes from MetaLWOz (Goal Source) - from high-stakes domains
        mw_sample = random.choice(high_stakes_tasks)
        goal = gen.extract_goal_from_metalwoz(mw_sample)

        # Apply Identity Grid (Augmentation)
        demographic = random.choice(const.RACES)
        gender = random.choice(const.GENDERS)
        occupation = random.choice(const.OCCUPATIONS_GRID)

        scenarios.extend(
            gen.generate_task_scenarios(goal, demographic, gender, occupation)
        )

    default_topics = gen.generate_default_topic_scenarios()
    scenarios.extend(default_topics)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)

    counts = {
        variant: sum(1 for s in scenarios if s["variant_type"] == variant)
        for variant in ("explicit", "implicit", "control", "default_topic")
    }

    print(f"\nSuccessfully saved {len(scenarios)} total scenarios to {OUT_FILE}.")
    print("Dataset Breakdown:")
    print(f"  - Explicit Variants (Target): {counts['explicit']}")
    print(f"  - Implicit Variants (Target): {counts['implicit']}")
    print(f"  - Control Variants (Default-Persona): {counts['control']}")
    print(f"  - Default-Topic Variants: {counts['default_topic']}")


if __name__ == "__main__":
    main()
