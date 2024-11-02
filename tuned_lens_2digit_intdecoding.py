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
from tuned_lens.nn.lenses import TunedLens, LogitLens
from tuned_lens import TunedLens
from tuned_lens.plotting import PredictionTrajectory
import ipywidgets as widgets
from plotly import graph_objects as go

# local imports
# from src.pythia_intermediate_decoder import PythiaIntermediateDecoder
# from src.olmo_intermediate_decoder import OlmoIntermediateDecoder

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


# model_id = 'gpt2-large'
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "allenai/OLMo-1B-0724-hf"
model_id = "EleutherAI/pythia-2.8b-deduped"
if "llama" not in model_id: cache_dir = "./models/" + model_id
# To try a diffrent modle / lens check if the lens is avalible then modify this code
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    cache_dir=cache_dir,
    device_map=device,
    )
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_id,
    cache_dir=cache_dir,
    )
tuned_lens = TunedLens.from_model_and_pretrained(model)
tuned_lens = tuned_lens.to(device)
logit_lens = LogitLens.from_model(model)

lens = tuned_lens
layer_stride = 1
top_k = 10

# question = "19+21"
# answer = "40"

with open("datasets/2digit_sum_dataset.json") as f:
    two_digit_dataset = json.load(f)
random.shuffle(two_digit_dataset)
n = len(two_digit_dataset)
# n = 10
first_n = two_digit_dataset[:n]
num_layers = len(model.base_model.layers)

colors = itertools.cycle(sns.color_palette("tab10"))
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

questions = [elt[0] for elt in first_n]
answers = [elt[1] for elt in first_n]

all_probabilities = []
avg_probabilities = {i: [] for i in range(num_layers+1)}
plotted_counter = 0
block_numbers = list(range(num_layers+1))

for i in tqdm(range(len(questions)), dynamic_ncols=True):
    # prepare the tokenized prompt and expected answer
    question = questions[i]
    prompt = f"Question: What is {question}? Answer: {question}="
    answer = answers[i]
    input_ids = tokenizer.encode(prompt)
    tokenized_answer = tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt").squeeze()
    targets = input_ids[1:] + [tokenizer.eos_token_id]

    #collect probs from lens
    pred_traj = PredictionTrajectory.from_lens_and_model(
        lens=lens,
        model=model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        targets=targets,
    )
    probs = torch.from_numpy(pred_traj.probs)

    # for every layer in lens output, collect prob of correct token id
    probabilities = []
    for i, layer_probabilities in enumerate(probs):
        probs_at_last_token = layer_probabilities[-1]
        # top_probabilities, token_ids = torch.topk(probs_at_last_token, top_k)
        # probs_percent = [(v * 100) for v in top_probabilities.tolist()]
        # tokens = tokenizer.batch_decode(token_ids.unsqueeze(-1))
        prob_of_correct_token = float(probs_at_last_token[tokenized_answer]) * 100
        probabilities.append(prob_of_correct_token)
        avg_probabilities[i].append(prob_of_correct_token)
    if plotted_counter < 10:
        sns.lineplot(x=block_numbers, y=probabilities, marker='o', label=f"Probability of token '{answer}' for question {question}", color=next(colors), alpha=0.3)
    plotted_counter += 1


avg_prob_values = [np.mean(avg_probabilities[block]) for block in block_numbers]
sns.lineplot(x=block_numbers, y=avg_prob_values, marker='o', color="black", label=f"Average Probability Across {n} prompts", linewidth=3)


plt.xlabel("Block Number", fontsize=17)
plt.ylabel("Probability (%)", fontsize=17)
plt.title(f"Probability of Correct Token Across Decoder Blocks of {model_id}", fontsize=18)

plt.ylim(0, 100)
# x_min = num_layers // 5
x_min = 0
plt.xlim(x_min, num_layers+1)  
plt.xticks(block_numbers[x_min:], fontsize=14)  
plt.yticks(fontsize=14)
plt.gca().yaxis.set_major_formatter(PercentFormatter())

plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.7)
plt.legend(fontsize=13, loc='upper left',)
plt.savefig(f"figures/2digit_accuracy_tuned_lens/{model_id}.pdf")
plt.show()
