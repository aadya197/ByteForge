# ByteForge — S1 to S2 Entity Resolution

This component matches Source 1 reference businesses to Source 2 records.

## Expected local data

The dataset is intentionally NOT part of the code package.

Place it at:

dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    └── test_source2.tsv

All TSV files must use tab separators.

## Install

From `code/business_entity_resolution/`:

```bash
pip install -r requirements.txt
```

## Train and validate

From `code/business_entity_resolution/`:

```bash
python src/pipeline.py --mode train
```

This:
1. normalizes names/addresses;
2. generates TF-IDF blocking candidates;
3. injects known positives into the training candidates;
4. trains an XGBoost binary classifier;
5. evaluates a held-out Source-1 validation split;
6. tunes the decision threshold for macro F0.5;
7. saves the model under `models/`.

## Predict test data

```bash
python src/pipeline.py --mode predict
```

This generates:

```text
output/candidate_pairs.tsv
output/matching_results.tsv
```

## Full run

```bash
python src/pipeline.py --mode full
```

## Important

This component is only the S1→S2 half of the team pipeline. The final team integration should combine its predictions with the S1→S3 component and then produce the single final submission files.
