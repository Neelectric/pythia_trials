# This is inspired by the paper "Interpreting Context Look-ups in Transformers: Investigating Attention-MLP Interactions" By Clement Neo, Shay B. Cohen and Fazl Barez
# A central hypothesis they use is that there are neurons in late-layer MLPs, who's input weights are similar to embeddings of tokens in the vocabulary

# system imports
import time

# external imports
from transformers import GPTNeoXForCausalLM, AutoModel, AutoTokenizer, OlmoForCausalLM
import torch
from tqdm import tqdm

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

# Model initialisation
model_id_pythia = "EleutherAI/pythia-70m-deduped"
cache_dir_pythia = "./models/pythia-70m-deduped/"

model_id = "allenai/OLMo-1B-hf"
cache_dir = "./models/allenai/OLMo-1B-hf"
model = OlmoForCausalLM.from_pretrained(
  model_id,
  cache_dir=cache_dir,
  device_map=device,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=cache_dir,
)

prompt = "Kristin and her son Justin went to visit her mother Carol on a nice Sunday afternoon. They went out for a movie together and had a good time. If Justin is Kristin's son, and Carol is Kristin's mom, it follows that Carol is Justin's "

lm_head = model.lm_head
embed_tokens = model.model.embed_tokens

print(embed_tokens.weight)
print(embed_tokens.weight.shape)
print(lm_head.weight)
print(lm_head.weight.shape)

is_same = torch.equal(lm_head.weight, embed_tokens.weight)
print(is_same)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
tokens = model.generate(**inputs,
                        do_sample=False,
                        max_new_tokens=10,
                        # repetition_penalty=1.0008,
                        )
tokenizer.decode(tokens[0])
print(tokenizer.decode(tokens[0]))