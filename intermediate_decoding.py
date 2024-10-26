import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None
        self.add_tensor = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,)+output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, unembed_matrix, norm, layer_id):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.layer_id = layer_id
        if hasattr(self.block, "self_attn"):
            attention = self.block.self_attn
        elif hasattr(self.block, "attention"):
            attention = self.block.attention
        else:
            raise TypeError("The attention modules of the decoder layers could not be recognised.")
        self.block.attention = AttnWrapper(attention)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None


    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))
        attn_output = self.block.attention.activations
        if attn_output is None:
            attn_output = output[1]
        self.attn_mech_output_unembedded = self.unembed_matrix(self.norm(attn_output))
        attn_output += args[0]
        self.intermediate_res_unembedded = self.unembed_matrix(self.norm(attn_output))
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_output_unembedded = self.unembed_matrix(self.norm(mlp_output))
        return output

    def attn_add_tensor(self, tensor):
        self.block.attention.add_tensor = tensor

    def reset(self):
        self.block.attention.reset()

    def get_attn_activations(self):
        return self.block.attention.activations

class IntermediateDecoder:
    def __init__(self, model_id="EleutherAI/pythia-14m"):
        cache_dir = "./models/"
        if torch.cuda.is_available(): self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): self.device = "mps"
        else: self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir + model_id,
            device_map=self.device,
            attn_implementation="eager",
            )
        if "OLMo" in model_id:
            embed_out = self.model.lm_head
            final_layer_norm = self.model.base_model.norm
            self.model.config.output_attentions = True
            self.model.config.output_hidden_states = True
        elif "pythia" in model_id:
            embed_out = self.model.embed_out
            final_layer_norm = self.model.base_model.final_layer_norm
        else:
            raise TypeError("The passed model id was not parsed as OLMo or pythia and so wasn't recognised.")
        for i, layer in enumerate(self.model.base_model.layers):
            self.model.base_model.layers[i] = BlockOutputWrapper(layer, embed_out, final_layer_norm, i)

    def generate_text(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            # repetition_penalty=1.0008,
            temperature=None
            )
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        with torch.no_grad():
          logits = self.model(input_ids).logits
          return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.model.layers[layer].get_attn_activations()

    def reset_all(self):
        for layer in self.model.base_model.layers:
            layer.reset()

    def return_decoded_activations(self, decoded_activations, label, topk):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return (label, list(zip(tokens, probs_percent)))


    def decode_all_layers(self, text, topk=2, printing=True, print_attn_mech=True, print_intermediate_res=True, print_mlp=True, print_block=True):
        self.get_logits(text)
        block_activations = []
        for i, layer in enumerate(self.model.base_model.layers):
            if printing: print(f'Layer {i}: Decoded intermediate outputs')
            if print_attn_mech:
                decoded_attn_mech = self.return_decoded_activations(layer.attn_mech_output_unembedded, 'Attention mechanism', topk)
                if printing: print(decoded_attn_mech)
            if print_intermediate_res:
                decoded_intermediate_res_stream = self.return_decoded_activations(layer.intermediate_res_unembedded, 'Intermediate residual stream', topk)
                if printing: print(decoded_intermediate_res_stream)
            if print_mlp:
                decoded_mlp_activations = self.return_decoded_activations(layer.mlp_output_unembedded, 'MLP output', topk)
                if printing: print(decoded_mlp_activations)
            if print_block:
                decoded_block_activations = self.return_decoded_activations(layer.block_output_unembedded, 'Block output', topk)
                if printing: print(decoded_block_activations)
                block_activations.append(decoded_block_activations)
            print("\n")
        return block_activations

# model_id = "allenai/OLMo-7B-0724-hf"
model_id = "EleutherAI/pythia-6.9B-deduped"
model = IntermediateDecoder(model_id=model_id)


question = "54+45"
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

print(block_activations)

# output = model.generate_text(prompt, max_new_tokens=7,)
# print(output)
