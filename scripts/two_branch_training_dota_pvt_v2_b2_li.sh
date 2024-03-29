CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 fsod_train_net.py --num-gpus 8 --dist-url auto \
	--config-file configs/fsod/two_branch_training_dota_pvt_v2_b2_li.yaml SOLVER.IMS_PER_BATCH 8 2>&1 | tee log/two_branch_training_dota_pvt_v2_b2_li.txt
