# system imports
import time
import json

# external imports
from transformers import GPTNeoXForCausalLM, AutoModel, AutoTokenizer, OlmoForCausalLM
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

model_id_pythia = "EleutherAI/pythia-6.9B-deduped"
cache_dir_pythia = "./models/" + model_id_pythia

model = GPTNeoXForCausalLM.from_pretrained(
  pretrained_model_name_or_path=model_id_pythia,
  # revision="step3000",
  cache_dir=cache_dir_pythia,
  device_map=device,
)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_id_pythia,
    cache_dir=cache_dir_pythia,
)

# prompt = "The most important political question in the world is"
# prompt = "Question. Kristin and her son Justin went to visit her mother Carol on a nice Sunday afternoon. They went out for a movie together and had a good time. If Justin is Kristin's son, and Carol is Kristin's mom, what is the relationship between Justin and Carol? Answer. "

eval_bqa = "LogicBench/LogicBench(Eval)/BQA/"
modus_ponens = "first_order_logic/modus_ponens/data_instances.json"

with open(eval_bqa+modus_ponens, "r") as file:
    data = json.load(file)

print(data["type"])
print(data["axiom"])
samples = data["samples"]
sample = samples[0]

context = sample["context"]
qa_pair = sample["qa_pairs"][0]
question = qa_pair["question"]
answer = qa_pair["answer"]
prompt = "Question. " + context + " " + question + " Answer. "


inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
tokens = model.generate(**inputs,
                        do_sample=False,
                        max_new_tokens=5,
                        # repetition_penalty=1.0008,
                        )
output = tokenizer.decode(tokens[0], clean_up_tokenization_spaces = False)
print(output)
