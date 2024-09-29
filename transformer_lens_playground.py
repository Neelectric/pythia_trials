# Playground for playing with Neel Nanda's transformer lens

# system imports
import time

# external imports
import torch
from tqdm import tqdm
import torch
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix
import circuitsvis as cv
# Testing that the library works
cv.examples.hello("Neel")



# local imports

# enivornment setup
torch.set_grad_enabled(False)
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

# NBVAL_IGNORE_OUTPUT
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = model.to_tokens(gpt2_text)
print(gpt2_tokens.device)
gpt2_logits, gpt2_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True)
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0, "attn"]
print(attention_pattern.shape)
gpt2_str_tokens = model.to_str_tokens(gpt2_text)
print("Layer 0 Head Attention Patterns:")
cv.attention.attention_patterns(tokens=gpt2_str_tokens, attention=attention_pattern)