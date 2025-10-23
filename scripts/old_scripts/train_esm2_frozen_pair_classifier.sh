mkdir -p ./logs
python src/models/train_esm2_frozen_pair_classifier.py 2>&1 | tee ./logs/train_esm2_frozen_pair_classifier.log
