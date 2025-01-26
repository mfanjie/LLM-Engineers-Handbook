# checkout source code
```
mkdir /root/llmeng/source
cd /root/llmeng/source
git clone https://github.com/mfanjie/LLM-Engineers-Handbook.git
```

# setup dependencies
## feature store: mongo
todo
## vector db: qdrant
todo
# setup local pipeline env
## get the docker image
### build docker image 
```
docker build -t mfanjie/llmeng-lite -f Dockerfile .
```
or you can use the pre-built image of mine
docker pull mfanjie/llmeng-lite
## start and enter container
### run docker container 
parameters:
- `-p 18237:18237`: to map host 18237 to zenml port 8237
- `-p 18000:8000`: to map host port 18000 to container inference port 8000
- `--gpus all`: to enable all gpus on the host
- `-v /usr/local/cuda:/usr/local/cuda`: to mount cuda to - container
- `-v /root/llmeng/source/LLM-Engineers-Handbook/:/app`: to mount source from host path `/root/llmeng/source/LLM-Engineers-Handbook/:/app` to container patch `/app`
- `-d`: to put container to daemon
- `mfanjie/llmeng-lite`: image tag
- `sleep infinity`: keep the container process
```
docker run -p 18237:8237 -p 18000:8000 --name llmeng --gpus all -v /usr/local/cuda:/usr/local/cuda -v /root/llmeng/source/LLM-Engineers-Handbook/:/app -d mfanjie/llmeng-lite sleep infinity
```
### then you can enter container by
```
docker exec -it llmeng bash
```
## setup env
### start the poetry shell
```
poetry shell
```
### install training requirements, unsloth requires specific driver and cuda version so its not built in in the image
```
pip install unsloth comet-ml
```
if your GPU supprt flash attention you can run
```
pip install -r llm_engineering/model/finetuning/requirements.txt
```
### set env variable
```
export SM_NUM_GPUS=1
export SM_MODEL_DIR=models
export SM_OUTPUT_DATA_DIR=output
```
### run training pipeline to do sft and dpo
- make sure finetuning_type: sft for first train and set finetuning_type: pdo after sft completed
- set is_dummy: true for pipeline testing and set it to false if you want to do real training
```
cat configs/training.yaml
```
```
settings:
  docker:
    parent_image: 992382797823.dkr.ecr.eu-central-1.amazonaws.com/zenml-rlwlcs:latest
    skip_build: True
  orchestrator.sagemaker:
    synchronous: false

parameters:
  finetuning_type: sft
  num_train_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 3e-4
  dataset_huggingface_workspace: mlabonne
  is_dummy: false # Change this to 'false' to run the training with the full dataset and epochs.
```
- run traing pipeline once for sft and once for dpo
```
poetry poe run-training-pipeline
```
### start inference service
```
poetry poe run-inference-ml-service
```

