CUDA_VISIBLE_DEVICES=0 python3 faster_rcnn_train_net.py --num-gpus 1 --resume --dist-url auto \
	--config-file configs/fsod/single_branch_pretraining_dota_pvt_v2_b2_li.yaml SOLVER.IMS_PER_BATCH 1 2>&1 | tee log/single_branch_pretraining_dota_pvt_v2_b2_li.txt
