# system imports
import time
import json

# external imports
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM, AutoTokenizer, OlmoForCausalLM
import torch
from tqdm import tqdm
from datasets import load_dataset

# local imports

# enivornment setup
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.mps.manual_seed(42)

# -------------------------Start of Script------------------------- #
# attempt to auto recognize the device!
device = "cpu"
if torch.cuda.is_available(): 
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
    device = "mps"
print(f"using device {device}")

model_id_olmo = "allenai/OLMo-7B-0724-Instruct-hf"
cache_dir_olmo = "./models/" + model_id_olmo

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id_olmo,
    # revision="step3000",
    cache_dir=cache_dir_olmo,
    device_map=device,
    )

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_id_olmo,
    cache_dir=cache_dir_olmo,
    )

# messages = [{"role": "user", "content": "What is 2+2?"}]
messages = "What is 19+21?"
# inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
inputs = tokenizer(messages, return_tensors="pt").to(model.device)

response = model.generate(**inputs, 
                          max_new_tokens=50, 
                          do_sample=False, 
                          )
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])

