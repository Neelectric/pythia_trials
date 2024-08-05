from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  # revision="step3000",
  # cache_dir="./pythia-70m-deduped/step3000",
  device_map="cuda:0",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  # revision="step3000",
  # cache_dir="./pythia-70m-deduped/step3000",
)

# prompt = "The most important political question in the world is"
prompt = "Kristin and her son Justin went to visit her mother Carol on a nice Sunday afternoon. They went out for a movie together and had a good time. If Justin is Kristin's son, and Carol is Kristin's mom, it follows that Carol is Justin's "

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
tokens = model.generate(**inputs,
                        do_sample=False,
                        max_new_tokens=10,
                        # repetition_penalty=1.0008,
                        )
tokenizer.decode(tokens[0])
print(tokenizer.decode(tokens[0]))
