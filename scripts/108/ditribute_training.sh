## distributed

## nohup run
nohup python -m torch.distributed.run --nnodes=1 --nproc_per_node=8  main.py  \
--data_path ./protein \
--output-dir /private2_data/lwj/code/protein_result_dir \
--dataset_name esm \
--model esm \
--lr 0.001 \
--epochs 25 \
--batch-size 8 \
--pool_mode attention2 \
--workers 4 > results.log 2>&1 &

#####  af2 node
## run
python -m torch.distributed.run --nnodes=2 --nproc_per_node=4  --master_port 29501 main.py  \
--data_path /data/xbiome/protein_classification \
--output-dir /home/af2/xbiome/DeepFold/work_dir \
--lr 0.0001 \
--epochs 20 \
--batch-size 1 \
--log_wandb \
--pool_mode pooler \
--workers 4

## single gpu
nohup  python main.py  \
--data_path /data/xbiome/protein_classification \
--output-dir /home/af2/xbiome/DeepFold/work_dir \
--dataset_name esm \
--model esm \
--pool_mode pooler \
--lr 0.001 \
--epochs 20 \
--batch-size 4 \
--workers 4

### af2
python main.py  \
--data_path /data/xbiome/protein_classification \
--output-dir /home/af2/xbiome/DeepFold/work_dir \
--dataset_name bert_embedding \
--model bert_embedding \
--lr 0.001 \
--epochs 20 \
--batch-size 64 \
--workers 4 \
--log_wandb


python main_esm_embedding.py  \
--data_path /data/xbiome/protein_classification \
--output-dir /home/af2/xbiome/DeepFold/work_dir \
--model esm_embedding \
--lr 0.001 \
--epochs 20 \
--batch-size 64 \
--workers 4

###  108 node
python -m torch.distributed.run  --nnodes=1 --nproc_per_node=2 --master_port 29501 main_esm_embedding.py  \
--data_path /data/xbiome/protein_classification \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--model esm_embedding \
--lr 0.001 \
--epochs 20 \
--batch-size 256 \
--workers 4 \
--log_wandb


python main.py  \
--data_path /home/niejianzheng/xbiome/datasets/protein  \
--output-dir /home/niejianzheng/xbiome/DeepFold/work_dir \
--dataset_name esm \
--model esm \
--pool_mode pooler \
--lr 0.001 \
--epochs 20 \
--batch-size 4 \
--workers 4
