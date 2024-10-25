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

model_id_pythia = "EleutherAI/pythia-2.8B-deduped"
cache_dir_pythia = "./models/" + model_id_pythia


print(f"loading {model_id_pythia}")
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id_pythia,
    # revision="step4000-tokens16B",
    cache_dir=cache_dir_pythia,
    device_map=device,
    )

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_id_pythia,
    cache_dir=cache_dir_pythia,
    )

with open("datasets/2digit_sum_dataset.json") as f:
    dataset = json.load(f)

bsz = 50

n_correct = 0
n_total = 0

for i in tqdm(range(0, len(dataset), bsz), dynamic_ncols=True):
    instance_batch = dataset[i : i+bsz]
    # print(f"selecting items from {i} to {i + bsz}")
    # print(f"batch is now {instance_batch}")
    question_batch = [instance[0] for instance in instance_batch]
    answer_batch = [instance[1] for instance in instance_batch]
    # print(question_batch)
    # print(answer_batch)
    prompts = [f"Question. What is {question}? Answer." for question in question_batch]
    # print(prompts)
    inputs = tokenizer(prompts, return_tensors="pt").to(model.device)
    input_lengths = [len(input) for input in inputs["input_ids"]]
    # print(input_lengths)
    # print(inputs)
    output_ids = model.generate(**inputs, 
                            max_new_tokens=10, 
                            do_sample=False, 
                            )
    prediction_batch = tokenizer.batch_decode(output_ids[:, 10:], skip_special_tokens=True)
    for prediction, answer in zip(prediction_batch, answer_batch):
        if answer in prediction:
            n_correct +=1
        n_total +=1
    # tqdm.write(str(prediction_batch))
print(f"Out of total {n_total} questions, we got {n_correct} correct. This is {n_correct/n_total:.2f}")
    


    # prompt = f"Question. What is {question}? Answer."
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # output_ids = model.generate(**inputs, 
    #                         max_new_tokens=20, 
    #                         do_sample=False, 
    #                         )
    # prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # tqdm.write(prediction)
    
    
