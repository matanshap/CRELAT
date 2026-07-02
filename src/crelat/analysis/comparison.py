"""Cross-representation comparison using stable play IDs."""

from __future__ import annotations

import pandas as pd

from crelat.analysis.isolation import mean_absolute_isolation


def compare_representations(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_feature: str,
    right_feature: str,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    metadata = [column for column in ("title", "genre", "year") if column in left.columns]
    left_rows = left.loc[
        left["feature"] == left_feature, ["play_id", *metadata, "value"]
    ].rename(
        columns={"value": left_name}
    )
    right_rows = right.loc[right["feature"] == right_feature, ["play_id", "value"]].rename(
        columns={"value": right_name}
    )
    joined = left_rows.merge(right_rows, on="play_id", how="inner", validate="one_to_one")
    if len(joined) < 2:
        raise ValueError("At least two overlapping plays are required")
    joined[f"{left_name}_isolation"] = mean_absolute_isolation(joined[left_name].to_numpy())
    joined[f"{right_name}_isolation"] = mean_absolute_isolation(joined[right_name].to_numpy())
    return joined
