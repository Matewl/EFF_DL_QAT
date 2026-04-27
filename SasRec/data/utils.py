import os
import pandas as pd
from collections import defaultdict


def data_partition(df):
    """
    Split data into train/val/test using leave-one-out strategy.
    """
    usernum = df["user_id"].max()
    itemnum = df["item_id"].max()
    
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)
    
    # Group by user
    grouped = df.groupby("user_id")
    
    for user_id, group in grouped:
        # sort by timestamp
        interactions = group.sort_values("timestamp")
        items = interactions["item_id"].tolist()
        
        if len(items) < 3:
            # If less than 3 interactions, put all in train (or ignore)
            user_train[user_id] = items
            continue
            
        # Last item -> test
        user_test[user_id] = [items[-1]]
        # Second last -> valid
        user_valid[user_id] = [items[-2]]
        # Rest -> train
        user_train[user_id] = items[:-2]
        
    return [user_train, user_valid, user_test, usernum, itemnum]


def load_movielens(data_path: str, dataset_name='ml-1m.txt'):
    """
    Load MovieLens 1M dataset.
    Expects ratings.dat in data_path.
    Returns a DataFrame with [user_id, item_id, rating, timestamp].
    """
    # Check for ratings.dat directly or inside ml-1m subdirectory
    ratings_file = os.path.join(data_path, dataset_name)
    
    print(f"Loading data from {ratings_file}...")
    # ML-1M: UserID::MovieID::Rating::Timestamp
    df = pd.read_csv(
        ratings_file,
        sep="::",
        header=None,
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    return df
