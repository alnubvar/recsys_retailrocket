import pandas as pd
from typing import Tuple
from pathlib import Path  # <- важно


def load_events(path: str) -> pd.DataFrame:
    events = pd.read_csv(path)
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms")
    return events


def filter_users_items(
    events: pd.DataFrame,
    min_user_events: int = 3,
    min_item_events: int = 5,
) -> pd.DataFrame:
    user_counts = events.groupby("visitorid").size()
    item_counts = events.groupby("itemid").size()

    valid_users = user_counts[user_counts >= min_user_events].index
    valid_items = item_counts[item_counts >= min_item_events].index

    filtered = events[
        events["visitorid"].isin(valid_users) & events["itemid"].isin(valid_items)
    ]

    return filtered


def temporal_split(
    events: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = events.sort_values("timestamp")

    n = len(events)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = events.iloc[:train_end]
    val = events.iloc[train_end:val_end]
    test = events.iloc[val_end:]

    return train, val, test


def prepare_datasets(
    input_path: str,
    output_dir: str,
    min_user_events: int = 3,
    min_item_events: int = 5,
) -> None:
    events = load_events(input_path)
    events = filter_users_items(
        events,
        min_user_events=min_user_events,
        min_item_events=min_item_events,
    )

    train, val, test = temporal_split(events)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_path / "train.csv", index=False)
    val.to_csv(output_path / "val.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)
