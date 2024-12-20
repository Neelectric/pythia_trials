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

model_id_pythia = "EleutherAI/pythia-12B-deduped"

model_id = model_id_olmo_7b_sft
cache_dir = "./models/" + model_id


print(f"loading {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    # revision="step4000-tokens16B",
    cache_dir=cache_dir,
    device_map=device,
    )

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_id,
    cache_dir=cache_dir,
    )

with open("datasets/2digit_sum_dataset.json") as f:
    dataset = json.load(f)

with open("datasets/2digit_sum_dataset.json") as f:
    dataset = json.load(f)

bsz = 50

n_correct = 0
n_total = 0

for i in tqdm(range(0, len(dataset), bsz), dynamic_ncols=True):
    instance_batch = dataset[i : i+bsz]
    question_batch = [instance[0] for instance in instance_batch]
    answer_batch = [instance[1] for instance in instance_batch]
    prompts = [f"Question: What is {question}? Answer: {question}=" for question in question_batch]
    inputs = tokenizer(prompts, return_tensors="pt").to(model.device)
    input_lengths = [len(input) for input in inputs["input_ids"]]
    output_ids = model.generate(**inputs, 
                            max_new_tokens=10, 
                            do_sample=False, 
                            pad_token_id=tokenizer.eos_token_id,
                            )
    prediction_batch = tokenizer.batch_decode(output_ids[:, 10:], skip_special_tokens=True)
    for prediction, answer in zip(prediction_batch, answer_batch):
        if answer in prediction:
            n_correct +=1
        n_total +=1
print(f"Out of total {n_total} questions, we got {n_correct} correct. This is {(n_correct/n_total)*100:.3f}%")