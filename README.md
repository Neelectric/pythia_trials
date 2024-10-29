# The Pythia Trials

## Intro
Public repository to perform some research on Pythia and OLMo and how they behave when varying sizes/checkpoints in training. The hope is to reverse engineer some algorithms!

### 2-digit Addition
I have started by looking at 2-digit addition because it is a super simple task that is easy to understand, reason about, and to implement. With 2-digit addition, I am referring to the task

question = “19+21”
prompt = f"Question: What is {question}? Answer: {question}="

Where the response has to be the correct summation. I only iterate through questions where the result is still two digits, ie. I stop iterating before 50+50=100. 

The following table shows the performance of OLMo and Pythia along various sizes

<!-- table -->
| Model                                | Size   | 2-sum    | temp      |
|:-------------------------------------|:-------|:---------|:----------|
| EleutherAI/pythia-14m                |  14m   |   0.00%  |           |
| EleutherAI/pythia-70m-deduped        |  70m   |   0.00%  |           |
| EleutherAI/pythia-160m-deduped       |  70m   |   0.16%  |           |
| EleutherAI/pythia-410m-deduped       | 410m   |   4.56%  |           |
| EleutherAI/pythia-1B-deduped         |   1B   |   5.08%  |           |
| EleutherAI/pythia-1.4B-deduped       | 1.4B   |  20.69%  |           |
| EleutherAI/pythia-2.8B-deduped       | 2.8B   |  86.32%  |           |
| EleutherAI/pythia-6.9B-deduped       | 6.9B   |  91.88%  |           |
| EleutherAI/pythia-12B-deduped        |  12B   |  88.96%  |           |

| allenai/OLMo-1B-0724-hf              |   1B   |          |           |
| hamishivi/OLMo-1B-0724-SFT-hf        |   1B   |          |           |
| hamishivi/OLMo-1B-0724-Instruct-hf   |   1B   |          |           |

| allenai/OLMo-7B-0724-hf              |   7B   |          |           |
| allenai/OLMo-7B-0724-SFT-hf          |   7B   |          |           |
| allenai/OLMo-7B-0724-Instruct-hf     |   7B   |          |           |
<!-- table -->