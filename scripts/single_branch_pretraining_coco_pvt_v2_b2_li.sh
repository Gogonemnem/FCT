CUDA_VISIBLE_DEVICES=0,1 python3 faster_rcnn_train_net.py --num-gpus 2  --num-machines 2 --machine-rank 0 --dist-url tcp://172.16.36.7:8686 \
	--config-file configs/fsod/single_branch_pretraining_coco_pvt_v2_b2_li.yaml 2>&1 | tee log_coco/single_branch_pretraining_coco_pvt_v2_b2_li.txt
