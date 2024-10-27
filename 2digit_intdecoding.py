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
from src.olmo_intermediate_decoder import OlmoIntermediateDecoder

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


model_id = "allenai/OLMo-7B-0724-hf"
# model_id = "EleutherAI/pythia-14m"
print(f"Loading {model_id}...")
if "pythia" in model_id:
    model = PythiaIntermediateDecoder(model_id=model_id)
elif "OLMo" in model_id:
    model = OlmoIntermediateDecoder(model_id=model_id)
else:
    raise TypeError("Could not recognise model type and associated intermediate decoder")

with open("datasets/2digit_sum_dataset.json") as f:
    two_digit_dataset = json.load(f)
random.shuffle(two_digit_dataset)
n = len(two_digit_dataset)
n = 100
first_n = two_digit_dataset[:n]

colors = itertools.cycle(sns.color_palette("tab10"))
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

questions = [elt[0] for elt in first_n]
answers = [elt[1] for elt in first_n]
all_probabilities = []
avg_probabilities = {i: [] for i in range(32)}
plotted_counter = 0

for i in tqdm(range(len(questions)), dynamic_ncols=True):
    question = questions[i]
    answer = answers[i]
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
        probability_correct = token_probs.get(answer, 0)
        block_numbers.append(block_num)
        probabilities.append(probability_correct)
        avg_probabilities[block_num].append(probability_correct)
    if plotted_counter < 10:
        sns.lineplot(x=block_numbers, y=probabilities, marker='o', label=f"Probability of token '{answer}' for question {question}", color=next(colors), alpha=0.3)
    plotted_counter += 1

avg_prob_values = [np.mean(avg_probabilities[block]) for block in block_numbers]
sns.lineplot(x=block_numbers, y=avg_prob_values, marker='o', color="black", label=f"Average Probability Across {n} prompts", linewidth=3)


plt.xlabel("Block Number", fontsize=17)
plt.ylabel("Probability (%)", fontsize=17)
plt.title(f"Probability of Correct Token Across Decoder Blocks of {model_id}", fontsize=18)

plt.ylim(0, 100)
num_layers = len(model.model.base_model.layers)
x_min = num_layers // 5
plt.xlim(x_min, num_layers)  
plt.xticks(block_numbers[x_min:], fontsize=14)  
plt.yticks(fontsize=14)
plt.gca().yaxis.set_major_formatter(PercentFormatter())

plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.7)
plt.legend(fontsize=13, loc='upper left',)
plt.savefig(f"figures/2digit_accuracy_intdecoding/{model_id}.pdf")
plt.show()