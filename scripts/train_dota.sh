#!/bin/sh
# CUDA_VISIBLE_DEVICES=0,1,2,3

### Pretraining (Single-Branch)
python3 train_net.py --num-gpus 1 --resume --dist-url  auto \
    --config-file configs/fsod/pretraining-pvtv2-retinanet.yaml \
    --additional-configs configs/fsod/dota.yaml \
        -- SOLVER.IMS_PER_BATCH 1 \
        SOLVER.ACCUMULATION_STEPS 8 \
        2>&1 | tee logs/pretraining-pvtv2-retinanet.txt

### Training (Two-branch)
python3 train_net.py --num-gpus 1 --resume --dist-url auto \
    --config-file configs/fsod/training-pvtv2-retinanet.yaml \
    --additional-configs configs/fsod/dota.yaml \
        -- SOLVER.IMS_PER_BATCH 1 \
        SOLVER.ACCUMULATION_STEPS 8 \
        2>&1 | tee logs/training-pvtv2-retinanet.txt

### Few-shot Finetuning (Two-branch)
iterations=(10) # (1 2 3 5 10 3-)

# Loop through the iterations
for i in "${iterations[@]}"; do
    dataset_name="dota_2014_train_full_${i}_shot"
    
    datasets="('${dataset_name}',)"
    test_shots="('${i}',)"
    
    # Execute the training command with the adjusted dataset name
    python3 train_net.py --num-gpus 1 --resume --dist-url auto \
        --config-file configs/fsod/finetuning-pvtv2-retinanet.yaml \
        --additional-configs configs/fsod/dota.yaml \
        -- SOLVER.IMS_PER_BATCH 1 \
        SOLVER.ACCUMULATION_STEPS 8 \
        INPUT.FS.SUPPORT_SHOT $i \
        DATASETS.TRAIN "$datasets" \
        DATASETS.TEST_SHOTS "$test_shots" \
    2>&1 | tee logs/finetuning-pvtv2-retinanet.txt
done
