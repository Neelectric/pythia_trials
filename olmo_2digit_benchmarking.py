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

model_id_olmo_1b_base = "allenai/OLMo-1B-0724-hf"
model_id_olmo_1b_sft = "hamishivi/OLMo-1B-0724-SFT-hf"
model_id_olmo_1b_inst = "hamishivi/OLMo-1B-0724-Instruct-hf"

model_id_olmo_7b_base = "allenai/OLMo-7B-0724-hf"
model_id_olmo_7b_sft = "allenai/OLMo-7B-0724-SFT-hf"
model_id_olmo_7b_inst = "allenai/OLMo-7B-0724-Instruct-hf"

model_id_olmo = model_id_olmo_7b_inst
cache_dir_olmo = "./models/" + model_id_olmo

print(f"loading {model_id_olmo}")
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

with open("datasets/2digit_sum_dataset.json") as f:
    dataset = json.load(f)

bsz = 10


for i in tqdm(range(0, len(dataset), bsz), dynamic_ncols=True):
    batch = dataset[i*bsz : (i+bsz) + bsz]
    print(f"selecting items from {i*bsz} to {(i+bsz) + bsz}")
    print(f"batch is now {batch}")

    # question = instance[0]
    # answer = instance[1]

    # prompt = f"Question. What is {question}? Answer."
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # output_ids = model.generate(**inputs, 
    #                         max_new_tokens=20, 
    #                         do_sample=False, 
    #                         )
    # prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # tqdm.write(prediction)
    break
    
