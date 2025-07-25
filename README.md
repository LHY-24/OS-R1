## Get Started
- [Environment Setup](https://github.com/0russwest0/Agent-R1/blob/main/docs/getting_started/installation.md)
- [Quick Start: Try Default Search Tool on HotpotQA](https://github.com/0russwest0/Agent-R1/blob/main/docs/getting_started/quickstart.md)

### Install LightRAG

```bash
git clone git@github.com:HKUDS/LightRAG.git
cd LightRAG
git checkout v1.1.1
pip3 install -e .
```

### Install kconfiglib

```bash
pip3 install kconfiglib
```

### Training

```bash
# create data
mkdir ~/data
cd ~/data
mkdir os_r1
cd os_r1
mkdir os_r1_train
mkdir os_r1_validate
# unzip OS-R1/os_r1_data/train.zip to os_r1_train
# unzip OS-R1/os_r1_data/validate.zip to os_r1_validate
cd /path/to/OS-R1
python3 examples/data_preprocess/os_r1.py

# start training
bash run_grpo_os_r1.sh
```

### Inference

```bash
# download linux kernel source code, for example, linux-6.8
# place an origin config to linux-6.8/.config
cd /path/to/OS-R1/Inference
python3 Inference.py /path/to/linux/kernel -t "Tuning target" --config-path "/path/to/OS-R1/agent_r1/src/config" --config-name "agent_trainer_inference"
```