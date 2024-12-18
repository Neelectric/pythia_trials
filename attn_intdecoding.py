# system imports
import time
import json
import itertools
import random

# external imports
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM, AutoTokenizer, OlmoForCausalLM
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# local imports
from src.pythia_intermediate_decoder import PythiaIntermediateDecoder
from src.olmo2_intermediate_decoder import Olmo2IntermediateDecoder

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


# model_id = "allenai/OLMo-7B-0724-hf"
# model_id = "EleutherAI/pythia-160m-deduped"
model_id = "allenai/OLMo-2-1124-7B-Instruct"
print(f"Loading {model_id}...")
if "pythia" in model_id:
    int_decoder = PythiaIntermediateDecoder(model_id=model_id)
elif "OLMo" in model_id:
    int_decoder = Olmo2IntermediateDecoder(model_id=model_id)
else:
    raise TypeError("Could not recognise model type and associated intermediate decoder")
dataset = load_dataset("LightEval/MATH")["train"]

print(dataset)

n = 100
first_n = dataset[:n]
num_layers = len(int_decoder.model.base_model.layers)

# colors = itertools.cycle(sns.color_palette("tab10"))
# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))

# questions = [elt[0] for elt in first_n]
# answers = [elt[1] for elt in first_n]
# all_probabilities = []
# avg_probabilities = {i: [] for i in range(num_layers)}
# plotted_counter = 0

# for i in tqdm(range(len(questions)), dynamic_ncols=True):
#     question = questions[i]
#     answer = answers[i]
#     prompt = f"Question: What is {question}? Answer: {question}="
#     int_decoder.reset_all()
#     block_activations = int_decoder.decode_all_layers(prompt, 
#                             topk=5,
#                             printing=False,
#                             print_attn_mech=False, 
#                             print_intermediate_res=False, 
#                             print_mlp=False, 
#                             print_block=True
#                             )
#     block_numbers = []
#     probabilities = []

#     for block_activation in block_activations:
#         block_num = int(block_activation[0].split()[1])
#         token_probs = dict(block_activation[1])