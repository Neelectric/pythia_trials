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

model_id_pythia = "EleutherAI/pythia-70m-deduped"
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


prompt = """
Answer the following two-digit addition tasks:
14 + 41 = 55
43 + 42 = 85
13 + 01 = 14
19 + 21 ="""

prompt = "What is 43+42?"

prompt = """
8+1=9
1+3=4
2+5=7
3+3=6
6+2="""

prompt = """1+1=2, 2+2=4, 3+3=6, 4+4=8, 5+5=10, 6+6=12, 7+7=14, 8+8=16, 9+9="""

prompt = "What is 43+42? 43+42="


inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(inputs)
input_ids = inputs["input_ids"].tolist()[0]
for elt in input_ids:
    print(tokenizer.decode(elt, skip_special_tokens=False))
tokens = model.generate(**inputs,
                        do_sample=False,
                        max_new_tokens=5,
                        )
output = tokenizer.decode(tokens[0], clean_up_tokenization_spaces = False)
print(output)




# for i in range(5):
#     message = ""
#     equation = input()
#     message += equation
#     inputs = tokenizer(message, return_tensors="pt").to(model.device)
#     response = model.generate(**inputs, 
#                             max_new_tokens=15, 
#                             do_sample=False, 
#                             )
#     print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
#     print("-"*100)

