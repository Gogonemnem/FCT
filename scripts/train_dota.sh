python3 train_net.py --num-gpus 1 --resume --dist-url  auto \
    --config-file configs/fsod/pretraining-pvtv2-retinanet.yaml \
    --additional-configs configs/fsod/dota.yaml \
    -- SOLVER.IMS_PER_BATCH 1 2>&1 | tee log/pretraining-pvtv2-retinanet.txt
