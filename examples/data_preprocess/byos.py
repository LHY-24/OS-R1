import argparse
import datasets
import json
import os

from verl.utils.hdfs_io import copy, makedirs

def download_file(url, local_path):
    print('download function is not implemented')
    exit()

def process_logs(source_dir):
    # Process QA logs to training datas. Returns a dict
    question_list = []
    answer_list = []
    type_list = []
    task_list = []

    for files in os.listdir(source_dir):
        if not files.endswith(".log"):
            continue
        with open(source_dir + "/" + files, "r") as f:
            task = f.readline()
            qa_instances = f.readlines()
        for qa_json_instance in qa_instances:
            qa_instance = json.loads(qa_json_instance)
            qa_type, question = qa_instance["question"].split("\t")
            answer = qa_instance["answer"]
            question_list.append(question)
            answer_list.append(qa_type + "\t" + json.dumps(answer))
            type_list.append(qa_type)
            task_list.append(task)

    return {
        'question': question_list,
        'answer': answer_list,
        'type': type_list,
        'task': task_list
    }

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

    # train_url = ""
    # validate_url = ""

    train_dir = os.path.join(local_dir, "byos_train")
    validate_dir = os.path.join(local_dir, "byos_validate")

    # train_file = os.path.join(local_dir, "byos_train.json")
    # validate_file = os.path.join(local_dir, "byos_validate.json")

    # if not os.path.exists(train_file):
    #     print(f"Downloading training data to {train_file}...")
    #     download_file(train_url, train_file)

    # if not os.path.exists(validate_file):
    #     print(f"Downloading training data to {validate_file}...")
    #     download_file(validate_url, validate_file)

    # print("Loading downloaded files...")
    # with open(train_file, 'r', encoding='utf-8') as f:
    #     train_data = json.load(f)
    
    # with open(validate_file, 'r', encoding='utf-8') as f:
    #     validation_data = json.load(f)

    train_dataset = datasets.Dataset.from_dict(process_logs(train_dir))
    validation_dataset = datasets.Dataset.from_dict(process_logs(validate_dir))

    # train_dataset = datasets.Dataset.from_dict({
    #     'question': [item['query'] for item in train_data],
    #     'answer': [item['result'] for item in train_data],
    # })
    # validation_dataset = datasets.Dataset.from_dict({
    #     'question': [item['query'] for item in validation_data],
    #     'answer': [item['result'] for item in validation_data],
    # })

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

    qa_bool_prompt = "I want to explore the config options related to TARGET in the Linux kernel configurations. Please choose the configs concerned with TARGET in the CONFIGS as much as possible. For each concerned config related to TARGET, you should determine whether it will increase or decrease TARGET. If it increases TARGET, output [CONFIG increase]. If it decreases TARGET, output [CONFIG decrease]. If a config is not related to TARGET, output [CONFIG - cannot determine impact without specific context]. I also have to gurantee the success boot of the OS after selecting. Answer the in the following form, without any explanation, just answer in pure text form, give me the config names in my given CONFIGS, each line represents a single config like this:\n[config_name_1 increase]\n[config_name_2 decrease]\n....\n[config_name_n increase]\nBelow are the numeric config options for your recommendations: \n"
    qa_menu_prompt = "I want to explore the config options related to TARGET in the Linux kernel configurations. Please choose the directories concerned with TARGET in the DIRECTORIES as much as possible. I also have to gurantee the success boot of the OS after selecting. Answer the in the following form, without any explanation, just answer in pure text form, give me directory names with index in my given DIRECTORIES, each line represents a single directory like this:\n[directory_name_1]\n[directory_name_2]\n....\n[directory_name_n]\nBelow are the numeric config options for your recommendations: \n"
    qa_choice_prompt = "I want to explore the config options related to TARGET in the Linux kernel configurations. The CONFIGS I gave you are choices of a config, and you need to choose which config is most likely related to TARGET. I also have to gurantee the success boot of the OS after selecting. Answer the in the following form, without any explanation, just answer in pure text form, each line represents a single config like this:\n[config_name]\nBelow are the numeric config options for your recommendations: \n"
    qa_value_prompt = "I want to explore the config options related to TARGET in the Linux kernel configurations. I have listed some numeric config options listed in menuconfig, along with their corresponding value ranges. For each option, please select a value that may help improve TARGET. If the option is not related to TARGET, reset it to the defalut value. Config input format: [option name] (default value). Value output format: [option name] (recommended  value). For instance, if you are given: 'maximum CPU number(1=>2 2=>4)  (cpunum) (1), Your response would be: 'maximum CPU number(1=>2 2=>4)  (cpunum) (2). Because when the CPU number is more, the speed is usually better. Below are the numeric config options for your recommendations:\n"
    qa_type_prompt = {
        "Bool": qa_bool_prompt,
        "Menu": qa_menu_prompt,
        "Choice": qa_choice_prompt,
        "Value": qa_value_prompt,
    }

    def make_map_fn():
        def process_fn(content):
            task = content.pop('task')
            qa_type = content.pop('type')
            question_raw = content.pop('question')
            question = instruction_following + \
                       "Question: Given TARGET = " + \
                       task + \
                       qa_type_prompt[qa_type] + \
                       question_raw

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
                },
                "extra_info": {
                    "type": qa_type
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