## evaluate diamond
python  evaluate_diamondscore.py \
    --train-data-file /Users/robin/xbiome/datasets/protein/train_data.pkl \
    --test-data-file /Users/robin/xbiome/datasets/protein/test_data.pkl \
    --diamond-scores-file /Users/robin/xbiome/datasets/protein/test_diamond.res \
    --ontology-obo-file /Users/robin/xbiome/datasets/protein/go.obo

## 108
## evaluate diamond
python  evaluate_diamondscore.py \
    --train-data-file /home/niejianzheng/xbiome/datasets/protein/train_data.pkl \
    --test-data-file /home/niejianzheng/xbiome/datasets/protein/test_data.pkl \
    --diamond-scores-file /home/niejianzheng/xbiome/datasets/protein/test_diamond.res \
    --ontology-obo-file /home/niejianzheng/xbiome/datasets/protein/go.obo \
    --output_dir /home/niejianzheng/xbiome/DeepFold/work_dir


## evaluate model
python  evaluate_deepmodel.py \
    --train-data-file /home/niejianzheng/xbiome/datasets/protein/train_data.pkl \
    --test-data-file /home/niejianzheng/xbiome/datasets/protein/predictions.pkl \
    --terms-file /home/niejianzheng/xbiome/datasets/protein/terms.pkl \
    --ontology-obo-file /home/niejianzheng/xbiome/datasets/protein/go.obo \
    --output_dir /home/niejianzheng/xbiome/DeepFold/work_dir

## evaluate go_sim
python  evaluate_gosim.py \
    --train-data-file /home/niejianzheng/xbiome/datasets/protein/train_data.pkl \
    --test-data-file /home/niejianzheng/xbiome/datasets/protein/cco_predictions.pkl \
    --terms-file /home/niejianzheng/xbiome/datasets/protein/cco_terms.pkl \
    --ontology-obo-file /home/niejianzheng/xbiome/datasets/protein/go_cafa3.obo \
    --output_dir /home/niejianzheng/xbiome/DeepFold/work_dir

## evaluate model
python  evaluate_deepmodel.py \
    --train-data-file /home/niejianzheng/xbiome/DeepFold/protein_data/cafa3/process/cco/cco_train_data.pkl \
    --test-data-file /home/niejianzheng/xbiome/DeepFold/protein_data/cafa3/process/cco_predictions.pkl \
    --terms-file /home/niejianzheng/xbiome/DeepFold/protein_data/cafa3/process/cco_terms.pkl \
    --ontology-obo-file /home/niejianzheng/xbiome/DeepFold/protein_data/cafa3/process/go_cafa3.obo \
    --output_dir /home/niejianzheng/xbiome/DeepFold/work_dir

## evaluate multi modal
python tools/evaluate_multimodal.py 
    --data_path data/cafa3 
    --train-data-file data/cafa3/mfo_esm1b_t33_650M_UR50S_embeddings_mean_train.pkl 
    --test-data-file data/cafa3/mfo_predictions1.pkl 
    --ontology-obo-file data/cafa3/go_cafa3.obo 
    --namespace 'mfo' 
    --output_dir ./work_dir
    --ont mf
