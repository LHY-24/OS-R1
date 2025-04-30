import os
import json

def main(json_dir, data_log_dir):
    datas = []

    # find log files in data log directory
    for files in os.listdir(data_log_dir):
        if not files.endswith(".log"):
            continue
        with open(data_log_dir + "/" + files, "r") as f:
            query = f.readline()
            ground_truth = "\n".join(f.readlines())
            datas.append({"query": query, "result": ground_truth})
    
    with open(json_dir, "w") as f:
        json.dump(datas, f)

if __name__ == "__main__":
    main(json_dir="/root/data/byos/byos.json", data_log_dir="/root/Agent-R1/byos_train_data/")