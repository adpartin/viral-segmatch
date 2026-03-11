P1:

Experiments:
1. No metadata filters, 5K isolates, CV kfold=10.
2. H3N2-only, 5K isolates, CV kfold=10.
3. Temporal holdout (train 2021–2023, test 2024).

Models:
ESM-2 + MLP Classifier, slot_norm, concat.
6-mers + MLP classifier, slot_norm, concat.

Before running these. There are few things we may need to address in the codebase.
1. Currently, data/.../runs/data_...<ts> models/.../runs/training_...<ts> includes ts, but results/ does not have runs folder nor ts in the filename. Thus, results are overwritten. How do you suggest addressing this?
2. Currently, when CV kfold>1, I don't know if this is somehow shown in the results plots. For example, we need to add a compute of spread (e.g., std) and add std as error bars (e.g., )

P2:

...

