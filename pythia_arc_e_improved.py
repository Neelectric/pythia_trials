import subprocess
from huggingface_hub import HfApi
from tqdm import tqdm
import lm_eval
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# attempt to auto recognize the device!
device = "cpu"
if torch.cuda.is_available(): 
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
    device = "mps"
print(f"using device {device}")

api = HfApi()
model_id = "EleutherAI/pythia-14m"
refs = api.list_repo_refs(model_id)
revisions = [ref.ref.split('/')[-1] for ref in refs.branches]
revisions.reverse() 
revisions.pop(0)
print(len(revisions))

accuracies = []
revision_list = []

for revision in tqdm(revisions, dynamic_ncols=True):
    print(revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision, 
        cache_dir = "./pythia-14m/" + revision,
        device_map=device,
        )
    wrapped_model = lm_eval.models.huggingface.HFLM(pretrained=model)

    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=wrapped_model,
        tasks=["arc_easy"],
        num_fewshot=0,
        batch_size=64,
    )
    accuracy = results["results"]["arc_easy"]["acc,none"]
    accuracies.append(accuracy)
    if len(accuracies) == 4:
        break
    del model


print("All evaluations completed")

data = pd.DataFrame({
    'Revisions': revision_list,
    'Accuracy': accuracies
})

# Plotting the revisions against the accuracies using seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Revisions', y='Accuracy', marker='o')
plt.xlabel('Revisions (Step Counts)')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs Revisions (Step Counts)')
plt.xticks(rotation=45)
plt.grid(True)

# Save the plot as an image file
plt.savefig('model_accuracy_vs_revisions.png')

plt.show()