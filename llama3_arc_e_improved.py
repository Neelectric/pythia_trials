import subprocess
from huggingface_hub import HfApi
from tqdm import tqdm
import lm_eval
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# attempt to auto recognize the device!
device = "cpu"
if torch.cuda.is_available(): 
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
    device = "mps"
print(f"using device {device}")

api = HfApi()
model_id = "meta-llama/Llama-3.1-8B-Instruct"


model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id,
    # cache_dir = "./pythia-14m/" + revision,
    device_map=device,
    )
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_id,
)
# wrapped_model = lm_eval.models.huggingface.HFLM(pretrained=model)

# results = lm_eval.simple_evaluate( # call simple_evaluate
#     model=wrapped_model,
#     tasks=["arc_easy"],
#     num_fewshot=0,
#     batch_size="auto"
# )
# del model

# print("All evaluations completed")


prompt = "19 + 21 ="


inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(inputs)
input_ids = inputs["input_ids"].tolist()[0]
for elt in input_ids:
    print(tokenizer.decode(elt, skip_special_tokens=False))
tokens = model.generate(**inputs,
                        do_sample=False,
                        max_new_tokens=10,
                        temperature=None,
                        top_p=None,
                        # repetition_penalty=1.0008,
                        )
output = tokenizer.decode(tokens[0], clean_up_tokenization_spaces = False)
print(output)
