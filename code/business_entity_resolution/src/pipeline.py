import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from config import (
    TRAIN_S1, TRAIN_S2, GROUND_TRUTH, TEST_S1, TEST_S2,
    OUTPUT_DIR, MODEL_DIR, RANDOM_STATE, TOP_K, MIN_BLOCK_SCORE,
    MATCH_THRESHOLD, VALIDATION_FRACTION
)
from preprocess import preprocess_dataframe
from blocking import TfidfBlocker
from features import build_feature_frame
from model import MatchModel
from metrics import macro_f05

def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

def load_truth(path):
    gt = read_tsv(path)
    truth = {}
    for _, row in gt.iterrows():
        ids = str(row["matched_entity_ids"]).strip()
        truth[row["source1_entity_id"]] = set(x for x in ids.split(",") if x)
    return truth

def candidate_truth_labels(candidates, truth):
    return np.array([
        int(c["candidate_entity_id"] in truth.get(c["source1_entity_id"], set()))
        for c in candidates
    ], dtype=int)

def inject_training_positives(candidates, source1, source2, truth):
    seen = {(c["source1_entity_id"], c["candidate_entity_id"]) for c in candidates}
    s2 = source2.set_index("entity_id")
    for s1_id, ids in truth.items():
        for s2_id in ids:
            if s2_id not in s2.index:
                continue
            key = (s1_id, s2_id)
            if key not in seen:
                candidates.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": s2_id,
                    "block_name_score": 1.0,
                })
                seen.add(key)
    return candidates

def split_by_source1(source1, fraction=0.2):
    rng = np.random.default_rng(RANDOM_STATE)
    ids = source1["entity_id"].to_numpy().copy()
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * fraction))
    return set(ids[n_val:]), set(ids[:n_val])

def train():
    s1 = preprocess_dataframe(read_tsv(TRAIN_S1))
    s2 = preprocess_dataframe(read_tsv(TRAIN_S2))
    truth = load_truth(GROUND_TRUTH)

    train_ids, val_ids = split_by_source1(s1, VALIDATION_FRACTION)
    s1_train = s1[s1.entity_id.isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1.entity_id.isin(val_ids)].reset_index(drop=True)

    # Build blocking index once over all S2 records.
    blocker = TfidfBlocker(TOP_K, MIN_BLOCK_SCORE).fit(s2)

    train_candidates = blocker.generate(s1_train)
    train_truth = {k: v for k, v in truth.items() if k in train_ids}
    train_candidates = inject_training_positives(
        train_candidates, s1_train, s2, train_truth
    )

    X_train, _ = build_feature_frame(s1_train, s2, train_candidates)
    y_train = candidate_truth_labels(train_candidates, train_truth)

    # Hard negatives are naturally supplied by the blocker.
    model = MatchModel(RANDOM_STATE).fit(X_train, y_train)

    # Validation uses only candidates produced by blocking; no positive injection.
    val_candidates = blocker.generate(s1_val)
    X_val, keys = build_feature_frame(s1_val, s2, val_candidates)
    probs = model.predict_proba(X_val)

    val_truth = {k: v for k, v in truth.items() if k in val_ids}
    best_threshold, best_score = 0.5, -1.0

    for threshold in np.arange(0.20, 0.951, 0.02):
        pred = {s1_id: set() for s1_id in val_ids}
        for (s1_id, s2_id), p in zip(keys, probs):
            if p >= threshold:
                pred[s1_id].add(s2_id)
        score = macro_f05(pred, val_truth)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "s1_s2_xgb.joblib")
    (MODEL_DIR / "s1_s2_threshold.txt").write_text(str(best_threshold))

    print(f"Validation macro F0.5: {best_score:.6f}")
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Training candidates: {len(train_candidates):,}")
    print(f"Validation candidates: {len(val_candidates):,}")

def predict():
    if not TEST_S1.exists() or not TEST_S2.exists():
        raise FileNotFoundError(
            "Put test_source1.tsv and test_source2.tsv under dataset/test/."
        )

    s1 = preprocess_dataframe(read_tsv(TEST_S1))
    s2 = preprocess_dataframe(read_tsv(TEST_S2))

    model = MatchModel.load(MODEL_DIR / "s1_s2_xgb.joblib")
    threshold = float((MODEL_DIR / "s1_s2_threshold.txt").read_text().strip())

    blocker = TfidfBlocker(TOP_K, MIN_BLOCK_SCORE).fit(s2)
    candidates = blocker.generate(s1)

    X, keys = build_feature_frame(s1, s2, candidates)
    probs = model.predict_proba(X)

    candidate_map = {s1_id: [] for s1_id in s1["entity_id"]}
    prediction_map = {s1_id: [] for s1_id in s1["entity_id"]}

    for (s1_id, s2_id), p in zip(keys, probs):
        candidate_map[s1_id].append(s2_id)
        if p >= threshold:
            prediction_map[s1_id].append(s2_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    match_rows = []
    for s1_id in s1["entity_id"]:
        # Remove duplicates while preserving order.
        cands = list(dict.fromkeys(candidate_map[s1_id]))
        preds = list(dict.fromkeys(prediction_map[s1_id]))
        candidate_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands)
        })
        match_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(preds)
        })

    pd.DataFrame(candidate_rows).to_csv(
        OUTPUT_DIR / "candidate_pairs.tsv", sep="\t", index=False
    )
    pd.DataFrame(match_rows).to_csv(
        OUTPUT_DIR / "matching_results.tsv", sep="\t", index=False
    )

    print("Wrote output/candidate_pairs.tsv")
    print("Wrote output/matching_results.tsv")
    print(f"Threshold: {threshold:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict", "full"], default="train")
    args = parser.parse_args()

    if args.mode in {"train", "full"}:
        train()
    if args.mode in {"predict", "full"}:
        predict()

if __name__ == "__main__":
    main()
