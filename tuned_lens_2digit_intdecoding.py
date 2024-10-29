import torch
from tuned_lens.nn.lenses import TunedLens, LogitLens
from tuned_lens import TunedLens
# from tuned_lens_example import TunedLens, LogitLens
from transformers import AutoModelForCausalLM, AutoTokenizer

from tuned_lens.plotting import PredictionTrajectory
import ipywidgets as widgets
from plotly import graph_objects as go

# temp = torch.load("/Users/s2011847/.cache/huggingface/hub/spaces--AlignmentResearch--tuned-lens/snapshots/1ac7285852a22309f571c2555efc37375d0c4cda/lens/EleutherAI/pythia-410m-deduped/params.pt", map_location=torch.device("cpu"))

device = torch.device('mps')
# model_id_pythia = 'gpt2-large'
# model_id_pythia = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id_pythia = "EleutherAI/pythia-1.4b-deduped"
if "llama" not in model_id_pythia: cache_dir_pythia = "./models/" + model_id_pythia
# To try a diffrent modle / lens check if the lens is avalible then modify this code
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id_pythia,
    cache_dir=cache_dir_pythia,
    device_map=device,
    )
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_id_pythia,
    cache_dir=cache_dir_pythia,
    )
tuned_lens = TunedLens.from_model_and_pretrained(model)
tuned_lens = tuned_lens.to(device)
logit_lens = LogitLens.from_model(model)


# def make_plot(lens, text, layer_stride, statistic, token_range):
lens = tuned_lens
layer_stride = 1


text = "Question: What is 19+21? Answer: 19+21="
input_ids = tokenizer.encode(text)
targets = input_ids[1:] + [tokenizer.eos_token_id]
statistic = "entropy"

pred_traj = PredictionTrajectory.from_lens_and_model(
    lens=lens,
    model=model,
    input_ids=input_ids,
    tokenizer=tokenizer,
    targets=targets,
)
# .slice_sequence(slice(*token_range))

getattr(pred_traj, statistic)().stride(layer_stride).figure(
    title=f"{lens.__class__.__name__} ({model.name_or_path}) {statistic}",
)

