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
model_id = "meta-llama/Llama-3.2-1B-Instruct"


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # cache_dir = "./pythia-14m/" + revision,
    device_map=device,
    )
wrapped_model = lm_eval.models.huggingface.HFLM(pretrained=model)

results = lm_eval.simple_evaluate( # call simple_evaluate
    model=wrapped_model,
    tasks=["arc_easy"],
    num_fewshot=0,
    batch_size="auto"
)
del model

print("All evaluations completed")