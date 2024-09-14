import subprocess
from huggingface_hub import HfApi
from tqdm import tqdm
import lm_eval
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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
# print(revisions)
revisions.pop(0)

for revision in tqdm(revisions):
    print(revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision, 
        cache_dir = "./pythia-14m/" + revision,
        device_map=device,
        )
    wrapped_model = lm_eval.models.huggingface.HFLM(pretrained=model)

    # To get a task dict for `evaluate`
    # task_dict = lm_eval.tasks.get_task_dict(
    #     [
    #         "arc_easy", # A stock task
    #         ],
    #         )

    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=wrapped_model,
        tasks=["arc_easy"],
        num_fewshot=0,
        batch_size="auto"
    )
    del model
    break


print("All evaluations completed")