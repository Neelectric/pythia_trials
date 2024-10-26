# system imports
import time
import json
import itertools
import random

# external imports
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM, AutoTokenizer, OlmoForCausalLM
import torch
from tqdm import tqdm
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# local imports
from src.intermediate_decoder import IntermediateDecoder

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
model_id = "EleutherAI/pythia-6.9B-deduped"
model = IntermediateDecoder(model_id=model_id)

with open("datasets/2digit_sum_dataset.json") as f:
    two_digit_dataset = json.load(f)

random.shuffle(two_digit_dataset)
first_10 = two_digit_dataset[0:10]

questions = [elt[0] for elt in first_10]
answers = [elt[1] for elt in first_10]
all_probabilities = []

for question, answer in zip(questions, answers):
    prompt = f"Question: What is {question}? Answer: {question}="
    model.reset_all()
    block_activations = model.decode_all_layers(prompt, 
                            topk=5,
                            printing=False,
                            print_attn_mech=False, 
                            print_intermediate_res=False, 
                            print_mlp=False, 
                            print_block=True
                            )
    block_numbers = []
    probabilities = []

    for block_activation in block_activations:
        block_num = int(block_activation[0].split()[1])
        token_probs = dict(block_activation[1])
        # print(block_num)
        # print(token_probs)
        probability_correct = token_probs.get(answer, 0)
        block_numbers.append(block_num)
        probabilities.append(probability_correct)
    all_probabilities.append(probabilities)

colors = itertools.cycle(sns.color_palette("tab10"))

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
for probabilities, answer in zip(all_probabilities, answers):
    sns.lineplot(x=block_numbers, y=probabilities, marker='o', label=f"Probability of '{answer}'", color=next(colors))
    print("PRINTING")
    print(probabilities)
    print("-"*100)
plt.xlabel("Block Number")
plt.ylabel("Probability (%)")
plt.title("Probability of Tokens Across Decoder Blocks")
plt.ylim(0, 100)
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.legend()
plt.show()

