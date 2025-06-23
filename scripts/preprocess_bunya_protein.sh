mkdir -p ./logs
python src/preprocess/preprocess_bunya_protein.py 2>&1 | tee ./logs/preprocess_bunya_protein.log
