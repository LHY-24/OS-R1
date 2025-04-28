import argparse
import datasets
import json
import os

from verl.utils.hdfs_io import copy, makedirs

def download_file(url, local_path):
    print('download function is not implemented')
    exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/byos')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=12800,
                        help='Number of training samples to use')
    parser.add_argument('--val_size', type=int, default=128,
                        help='Number of validation samples to use')
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_url = ""
    validate_url = ""

    train_file = os.path.join(local_dir, "byos_train.json")
    validate_file = os.path.join(local_dir, "byos_validate.json")

    if not os.path.exists(train_file):
        print(f"Downloading training data to {train_file}...")
        download_file(train_url, train_file)

    if not os.path.exists(validate_file):
        print(f"Downloading training data to {validate_file}...")
        download_file(validate_url, validate_file)

    print("Loading downloaded files...")
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(validate_file, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    train_dataset = datasets.Dataset.from_dict({
        'question': [item['query'] for item in train_data],
        'answer': [item['result'] for item in train_data],
    })
    validation_dataset = datasets.Dataset.from_dict({
        'question': [item['query'] for item in validation_data],
        'answer': [item['result'] for item in validation_data],
    })

    instruction_following = """Answer the given question. You can use the tools provided to you to answer the question. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>
"""

    def make_map_fn():
        def process_fn(content):
            question_raw = content.pop('question')
            question = instruction_following + "Question: " + question_raw

            answer = content.pop('answer')
            
            data = {
                "data_source": 'byos',
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "linux_config_autogen",
                "reward_model": {
                    "ground_truth": answer
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn())
    validation_dataset = validation_dataset.map(function=make_map_fn())

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)