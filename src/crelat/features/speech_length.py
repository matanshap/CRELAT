"""Speech-pair length and lexical-similarity features."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_apostrophes(value: object) -> object:
    return value.replace("\u2018", "'").replace("\u2019", "'") if isinstance(value, str) else value


def count_words(value: object) -> int:
    if not isinstance(value, str):
        return 0
    return len(re.findall(r"[A-Za-z']+", normalize_apostrophes(value)))


def add_tfidf_similarity(interactions: pd.DataFrame) -> pd.DataFrame:
    required = {"text1", "text2"}
    if missing := required - set(interactions.columns):
        raise ValueError(f"Missing text columns: {', '.join(sorted(missing))}")
    result = interactions.copy()
    for column in result.select_dtypes(include="object").columns:
        result[column] = result[column].map(normalize_apostrophes)
    documents = pd.concat([result["text1"], result["text2"]], ignore_index=True).fillna("")
    matrix = TfidfVectorizer(lowercase=True).fit_transform(documents)
    count = len(result)
    numerators = matrix[:count].multiply(matrix[count:]).sum(axis=1).A1
    denominator = np.sqrt(matrix[:count].multiply(matrix[:count]).sum(axis=1).A1) * np.sqrt(
        matrix[count:].multiply(matrix[count:]).sum(axis=1).A1
    )
    result["tfidf_cosine_similarity"] = np.divide(
        numerators, denominator, out=np.zeros(count), where=denominator > 0
    )
    return result


def add_length_metrics(
    interactions: pd.DataFrame,
    tokenizer: object,
    max_sequence_length: Optional[int] = None,
) -> pd.DataFrame:
    result = interactions.copy()
    for column in result.select_dtypes(include="object").columns:
        result[column] = result[column].map(normalize_apostrophes)
    texts = pd.unique(
        pd.concat([result["text1"], result["text2"]], ignore_index=True).fillna("").astype(str)
    )
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    content_limit = min(int(tokenizer.model_max_length), max_sequence_length or 512) - special_tokens
    lengths = {
        text: len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
        for text in texts
    }
    result["length1_tokens_raw"] = result["text1"].map(lengths)
    result["length2_tokens_raw"] = result["text2"].map(lengths)
    result["length1_tokens"] = result["length1_tokens_raw"].clip(upper=content_limit)
    result["length2_tokens"] = result["length2_tokens_raw"].clip(upper=content_limit)
    result["length1_truncated"] = result["length1_tokens_raw"] > content_limit
    result["length2_truncated"] = result["length2_tokens_raw"] > content_limit
    result["length_difference"] = (
        result["length1_tokens"] - result["length2_tokens"]
    ).abs()
    result["length_sum"] = result["length1_tokens"] + result["length2_tokens"]
    result["length_min"] = result[["length1_tokens", "length2_tokens"]].min(axis=1)
    result["length_max"] = result[["length1_tokens", "length2_tokens"]].max(axis=1)
    result["length_ratio"] = result["length_min"] / result["length_max"].replace(0, np.nan)
    return result


def add_word_count_metrics(interactions: pd.DataFrame) -> pd.DataFrame:
    required = {"text1", "text2"}
    if missing := required - set(interactions.columns):
        raise ValueError(f"Missing text columns: {', '.join(sorted(missing))}")
    result = interactions.copy()
    result["word_count1"] = result["text1"].map(count_words)
    result["word_count2"] = result["text2"].map(count_words)
    result["word_count_difference_nominal"] = (
        result["word_count1"] - result["word_count2"]
    ).abs()
    result["word_count_change_percent"] = np.divide(
        result["word_count2"] - result["word_count1"],
        result["word_count1"],
        out=np.full(len(result), np.nan, dtype=float),
        where=result["word_count1"].to_numpy() > 0,
    ) * 100
    return result


def speech_pair_export(interactions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "play_title",
        "scene_id",
        "speaker1",
        "speaker2",
        "text1",
        "text2",
        "cosine_similarity",
        "tfidf_cosine_similarity",
        "word_count_change_percent",
        "word_count_difference_nominal",
    ]
    missing = [column for column in columns if column not in interactions.columns]
    if missing:
        raise ValueError(f"Missing export columns: {', '.join(missing)}")
    return interactions.loc[:, columns].rename(
        columns={
            "play_title": "play_name",
            "scene_id": "act_scene",
        }
    )
