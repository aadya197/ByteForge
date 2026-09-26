import re
import numpy as np
from rapidfuzz import fuzz, distance

def _tokens(x):
    return set(str(x).split()) if x else set()

def _jaccard(a, b):
    a, b = _tokens(a), _tokens(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _numeric_tokens(x):
    return set(re.findall(r"\d+", str(x)))

def pair_features(a, b, block_name_score=0.0):
    name_a, name_b = a["name_norm"], b["name_norm"]
    addr_a, addr_b = a["address_norm"], b["address_norm"]

    name_ratio = fuzz.ratio(name_a, name_b) / 100.0
    name_token = fuzz.token_set_ratio(name_a, name_b) / 100.0
    name_sort = fuzz.token_sort_ratio(name_a, name_b) / 100.0
    name_jw = distance.JaroWinkler.normalized_similarity(name_a, name_b)

    addr_ratio = fuzz.ratio(addr_a, addr_b) / 100.0
    addr_token = fuzz.token_set_ratio(addr_a, addr_b) / 100.0
    addr_sort = fuzz.token_sort_ratio(addr_a, addr_b) / 100.0
    addr_jw = distance.JaroWinkler.normalized_similarity(addr_a, addr_b)

    addr_jaccard = _jaccard(addr_a, addr_b)
    nums_a, nums_b = _numeric_tokens(addr_a), _numeric_tokens(addr_b)
    numeric_jaccard = (
        len(nums_a & nums_b) / len(nums_a | nums_b)
        if nums_a or nums_b else 0.0
    )

    country_match = float(a["country_norm"] == b["country_norm"])
    length_ratio = (
        min(len(name_a), len(name_b)) / max(len(name_a), len(name_b))
        if max(len(name_a), len(name_b)) else 1.0
    )

    return {
        "block_name_score": float(block_name_score),
        "name_ratio": name_ratio,
        "name_token_set_ratio": name_token,
        "name_token_sort_ratio": name_sort,
        "name_jaro_winkler": name_jw,
        "name_jaccard": _jaccard(name_a, name_b),
        "name_length_ratio": length_ratio,
        "address_ratio": addr_ratio,
        "address_token_set_ratio": addr_token,
        "address_token_sort_ratio": addr_sort,
        "address_jaro_winkler": addr_jw,
        "address_jaccard": addr_jaccard,
        "numeric_address_jaccard": numeric_jaccard,
        "country_match": country_match,
        "name_x_address": name_token * addr_token,
        "max_name_similarity": max(name_ratio, name_token, name_sort, name_jw),
    }

def build_feature_frame(source1, source2, candidates):
    s1 = source1.set_index("entity_id")
    s2 = source2.set_index("entity_id")
    rows = []
    keys = []

    for c in candidates:
        a = s1.loc[c["source1_entity_id"]]
        b = s2.loc[c["candidate_entity_id"]]
        rows.append(pair_features(a, b, c.get("block_name_score", 0.0)))
        keys.append((c["source1_entity_id"], c["candidate_entity_id"]))

    import pandas as pd
    X = pd.DataFrame(rows).fillna(0.0)
    return X, keys
