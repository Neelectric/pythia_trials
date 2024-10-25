import json

output_list = []
for summand_a in range(1,51):
    for summand_b in range(1,51):
        sum = summand_a + summand_b
        question = f"{summand_a}+{summand_b}"
        answer = str(sum)
        output_list.append((question, answer))
print(len(output_list))
print(output_list)
with open('datasets/2digit_sum_dataset.json', 'w') as f:
    json.dump(output_list, f)