CUDA_VISIBLE_DEVICES=0,1,2,3 python3 faster_rcnn_train_net.py --num-gpus 4 --resume --dist-url auto \
	--config-file configs/fsod/single_branch_pretraining_pascalcoco_pvt_v2_b2_li.yaml 2>&1 | tee log_pascalcoco/single_branch_pretraining_coco_pvt_v2_b2_li.txt
