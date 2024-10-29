import torch
from tuned_lens.nn.lenses import TunedLens, LogitLens
from transformers import AutoModelForCausalLM, AutoTokenizer

from tuned_lens.plotting import PredictionTrajectory
import ipywidgets as widgets
from plotly import graph_objects as go

# temp = torch.load("/Users/s2011847/.cache/huggingface/hub/spaces--AlignmentResearch--tuned-lens/snapshots/1ac7285852a22309f571c2555efc37375d0c4cda/lens/EleutherAI/pythia-410m-deduped/params.pt", map_location=torch.device("cpu"))

device = torch.device('mps')
# model_id_pythia = 'gpt2-large'
# model_id_pythia = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id_pythia = "EleutherAI/pythia-410m-deduped"
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


def make_plot(lens, text, layer_stride, statistic, token_range):
    input_ids = tokenizer.encode(text)
    targets = input_ids[1:] + [tokenizer.eos_token_id]

    if len(input_ids) == 0:
        return widgets.Text("Please enter some text.")
    
    if (token_range[0] == token_range[1]):
        return widgets.Text("Please provide valid token range.")
    pred_traj = PredictionTrajectory.from_lens_and_model(
        lens=lens,
        model=model,
        input_ids=input_ids,
        tokenizer=tokenizer,
        targets=targets,
    ).slice_sequence(slice(*token_range))

    return getattr(pred_traj, statistic)().stride(layer_stride).figure(
        title=f"{lens.__class__.__name__} ({model.name_or_path}) {statistic}",
    )

style = {'description_width': 'initial'}
statistic_wdg = widgets.Dropdown(
    options=[
        ('Entropy', 'entropy'),
        ('Cross Entropy', 'cross_entropy'),
        ('Forward KL', 'forward_kl'),
    ],
    description='Select Statistic:',
    style=style,
)
text_wdg = widgets.Textarea(
    description="Input Text",
    value="it was the best of times, it was the worst of times",
)
lens_wdg = widgets.Dropdown(
    options=[('Tuned Lens', tuned_lens), ('Logit Lens', logit_lens)],
    description='Select Lens:',
    style=style,
)

layer_stride_wdg = widgets.BoundedIntText(
    value=2,
    min=1,
    max=10,
    step=1,
    description='Layer Stride:',
    disabled=False
)

token_range_wdg = widgets.IntRangeSlider(
    description='Token Range',
    min=0,
    max=1,
    step=1,
    style=style,
)


def update_token_range(*args):
    token_range_wdg.max = len(tokenizer.encode(text_wdg.value))

update_token_range()

token_range_wdg.value = [0, token_range_wdg.max]
text_wdg.observe(update_token_range, 'value')

interact = widgets.interact.options(manual_name='Run Lens', manual=True)

plot = interact(
    make_plot,
    text=text_wdg,
    statistic=statistic_wdg,
    lens=lens_wdg,
    layer_stride=layer_stride_wdg,
    token_range=token_range_wdg,
)