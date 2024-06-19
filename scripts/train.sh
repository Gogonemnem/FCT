#!/bin/sh
# CUDA_VISIBLE_DEVICES=0,1,2,3

config_name="pvtv2-retinanet"
dataset="dota"
dataset_config="configs/fsod/${dataset}.yaml"
num_gpus=1
effective_batch_size=8
batch_size=1

### Few-shot Finetuning (Two-branch)
few_shot_iterations="10" # (1 2 3 5 10 3-)

# Calculate the number of accumulation steps
accumulation_steps=$((effective_batch_size / batch_size))

export TORCH_DISTRIBUTED_DEBUG=DETAIL
## Pretraining (Single-Branch)
python3 train_net.py --num-gpus $num_gpus --resume --dist-url  auto \
    --config-file "$dataset_config" \
    --additional-configs "configs/fsod/pretraining-${config_name}.yaml" \
        -- SOLVER.IMS_PER_BATCH $batch_size \
        SOLVER.ACCUMULATION_STEPS $accumulation_steps \
        2>&1 | tee "logs/pretraining-${config_name}.txt"

### Training (Two-branch)
python3 train_net.py --num-gpus $num_gpus --resume --dist-url auto \
    --config-file "$dataset_config" \
    --additional-configs configs/fsod/training-${config_name}.yaml \
        -- SOLVER.IMS_PER_BATCH $batch_size \
        SOLVER.ACCUMULATION_STEPS $accumulation_steps \
        2>&1 | tee "logs/training-${config_name}.txt"

### Fine-tuning (Two-branch)
for i in "$few_shot_iterations"; do
    dataset_name="${dataset}_2014_train_full_${i}_shot"
    
    datasets="('${dataset_name}',)"
    test_shots="('${i}',)"
    
    # Execute the training command with the adjusted dataset name
    python3 train_net.py --num-gpus $num_gpus --dist-url auto \
        --additional-configs configs/fsod/finetuning-${config_name}.yaml \
        --config-file "$dataset_config" \
        -- SOLVER.IMS_PER_BATCH $batch_size \
        SOLVER.ACCUMULATION_STEPS $accumulation_steps \
        INPUT.FS.SUPPORT_SHOT $i \
        DATASETS.TRAIN "$datasets" \
        DATASETS.TEST_SHOTS "$test_shots" \
        2>&1 | tee "logs/finetuning-${config_name}_${i}_shot.txt"
done
