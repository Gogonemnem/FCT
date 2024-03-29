CUDA_VISIBLE_DEVICES=4,5,6,7 python3 fsod_train_net.py --num-gpus 4 --dist-url auto \
         --config-file configs/fsod/two_branch_5shot_finetuning_pascalcoco_pvt_v2_b2_li.yaml SOLVER.IMS_PER_BATCH 8 2>&1 | tee log_pascalcoco/two_branch_5shot_finetuning_coco_pvt_v2_b2_li.txt
#