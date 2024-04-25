from multiprocessing import Pool
import argparse
import os
import json
import random
import math
from transformers import LongformerTokenizer
from tqdm import tqdm

def map_to_small_files(prompt_file, file_nums, shuffle=False):
    with open(prompt_file, 'r') as f:
        data = json.load(f)

    conversation_list = []
    for entry in data:
        prefix_prompt = ""
        conversations = []
        for conversation in entry.get("conversations", []):
            prompt = conversation.get("value").replace("\n", " ")
            if prompt:
                conversations.append(prefix_prompt + prompt + "\n")
                prefix_prompt += prompt + " "
        if conversations:
            conversation_list.append(conversations)

    avg = math.ceil(len(conversation_list) / file_nums)
    sub_conversation_list = [conversation_list[i:i+avg] for i in range(0, len(conversation_list), avg)]

    for i in range(file_nums):
        with open(f'./small_files/prompts_{i}.txt', 'w') as f:
            prompts_list = []
            # shuffle the conversation list
            if shuffle:
                while sub_conversation_list[i]:
                    random_index = random.randrange(len(sub_conversation_list[i]))
                    selected_conversation = sub_conversation_list[i][random_index]
                    if selected_conversation:
                        tokens = selected_conversation[0].split(' ')
                        tokens = [token for token in tokens]
                        prompts_list.append(selected_conversation.pop(0))
                    if not selected_conversation:
                        sub_conversation_list[i].pop(random_index)
            else:
                for conversation in sub_conversation_list[i]:
                    for prompt in conversation:
                        tokens = prompt.split(' ')
                        tokens = [token for token in tokens]
                        prompts_list.append(prompt)
            f.writelines(prompts_list)

def process_small_files(file_name):
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    with open(f'./small_files/{file_name}', 'r') as f:
        prompts = f.read().strip().split("\n")

    num_prompts = len(prompts)

    file_index = int(file_name.split('_')[-1].split('.')[0])

    with open(f'./small_files/tokens_{file_index}', 'w') as tf:
        for prompt in tqdm(prompts, desc=f'Processing {file_name}', total=num_prompts, position=file_index, leave=True):
            index = 0
            tokens_set = []
            full_prompt = prompt.strip()
            while index < len(full_prompt):
                if index + 4000 < len(full_prompt):
                    index_end = index + 4000
                    for i in range(index_end-1, index-1, -1):
                        if full_prompt[i] == " ":
                            index_end = i
                            break
                else:
                    index_end = len(full_prompt)
                if index_end - index <= 0:
                    break
                tokens = tokenizer.encode(full_prompt[index:index_end], add_special_tokens=True)
                tokens_set.append(tokens)
                index = index_end

            token_str = ""
            for tokens in tokens_set:
                token_str += " ".join([str(token) for token in tokens])
            tf.write(token_str + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-num', type=int, default=os.cpu_count())
    parser.add_argument('--dir', type=str, default='./small_files')
    parser.add_argument('--prompt-file', type=str, default='./prompt-samples.txt')
    parser.add_argument('--shuffle', type=bool, default=False)
    args = parser.parse_args()
    if not os.path.exists(args.dir):
        os.mkdir(args.dir)
    map_to_small_files(args.prompt_file, args.file_num, args.shuffle)
    file_names = [f'prompts_{i}.txt' for i in range(args.file_num)]
    with Pool(args.file_num) as p:
        p.map(process_small_files, file_names)
