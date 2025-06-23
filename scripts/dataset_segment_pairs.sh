mkdir -p ./logs
python src/datasets/dataset_segment_pairs.py 2>&1 | tee ./logs/dataset_segment_pairs.log
