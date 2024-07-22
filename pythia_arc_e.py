import subprocess
from huggingface_hub import HfApi

api = HfApi()
refs = api.list_repo_refs("EleutherAI/pythia-14m")
revisions = [ref.ref.split('/')[-1] for ref in refs.branches]
revisions.reverse() 
# print(revisions)
revisions.pop(0)

base_command = [
    "lm_eval",
    "--model", "hf",
    "--model_args", "pretrained=EleutherAI/pythia-14m",
    "--tasks", "arc_easy",
    "--device", "cuda:0",
    "--batch_size", "32",
    "--output_path", "output/pythia_arc",
    # "--use_cache", "output/pythia_arc"
]
for revision in revisions:
    full_command = base_command.copy()
    full_command[4] += f",revision={revision},dtype=float"
    
    print(f"Running command: {' '.join(full_command)}")
    
    # Run the command
    try:
        result = subprocess.run(full_command, check=True, text=True, capture_output=True)
        print(f"Command for revision {revision} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running command for revision {revision}: {e}")
        print(f"Error output: {e.stderr}")

print("All evaluations completed")