import re
import unicodedata
import pandas as pd

LEGAL_SUFFIXES = {
    "private", "pvt", "limited", "ltd", "llp", "inc", "incorporated",
    "corp", "corporation", "co", "company", "llc", "plc"
}

ABBREVIATIONS = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "apartment": "apt",
    "building": "bldg", "floor": "fl"
}

def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def normalize_name(value):
    text = normalize_text(value)
    tokens = [ABBREVIATIONS.get(t, t) for t in text.split()]
    tokens = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)

def normalize_address(value):
    text = normalize_text(value)
    tokens = [ABBREVIATIONS.get(t, t) for t in text.split()]
    return " ".join(tokens)

def preprocess_dataframe(df):
    df = df.copy()
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["name_norm"] = df["business_name"].map(normalize_name)
    df["address_norm"] = df["business_address"].map(normalize_address)
    df["country_norm"] = df["country"].map(normalize_text)
    df["name_tokens"] = df["name_norm"].map(lambda x: set(x.split()))
    df["address_tokens"] = df["address_norm"].map(lambda x: set(x.split()))
    return df
