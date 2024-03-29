CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 faster_rcnn_train_net.py --num-gpus 8 --resume --dist-url auto \
	--config-file configs/fsod/single_branch_pretraining_dior_pvt_v2_b2_li.yaml 2>&1 | tee log_dior/single_branch_pretraining_coco_pvt_v2_b2_li.txt
