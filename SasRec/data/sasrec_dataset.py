import torch
import numpy as np
from torch.utils.data import Dataset

class SASRecDataset(Dataset):
    def __init__(self, user_train, usernum, itemnum, maxlen, train=True):
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.train = train
        
        # Filter users who have training data
        self.users = [u for u in range(1, int(usernum) + 1) if len(user_train[u]) > 0]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        train_items = self.user_train[user]
        
        seq_len = len(train_items)
        
        if seq_len <= self.maxlen:
            padding_len = self.maxlen - seq_len
            input_seq = [0] * padding_len + train_items

        if seq_len < 2:
            return self.__getitem__((idx + 1) % len(self))

        if seq_len > self.maxlen:
            
            input_ids = train_items[:-1]
            target_ids = train_items[1:]

            if len(input_ids) > self.maxlen:
                input_ids = input_ids[-self.maxlen:]
                target_ids = target_ids[-self.maxlen:]
            else:
                pad_len = self.maxlen - len(input_ids)
                input_ids = [0] * pad_len + input_ids
                target_ids = [0] * pad_len + target_ids
                
        else: 
            input_ids = train_items[:-1]
            target_ids = train_items[1:]
            
            pad_len = self.maxlen - len(input_ids)
            input_ids = [0] * pad_len + input_ids
            target_ids = [0] * pad_len + target_ids

        neg_ids = []
        ts = set(train_items)
        for _ in range(self.maxlen):
            t = np.random.randint(1, self.itemnum + 1)
            while t in ts:
                t = np.random.randint(1, self.itemnum + 1)
            neg_ids.append(t)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(neg_ids, dtype=torch.long)
        )


class SASRecEvalDataset(Dataset):
    """
    Dataset for validation / test evaluation.
    Each item corresponds to one user, its interaction history (from user_train)
    and the target item from user_valid or user_test.
    """

    def __init__(self, user_train, user_split, usernum, itemnum, maxlen):
        self.user_train = user_train
        self.user_split = user_split
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen

        # users that have both train history and at least one item in the split
        self.users = [
            u
            for u in range(1, int(usernum) + 1)
            if len(user_train[u]) > 0 and len(user_split[u]) > 0
        ]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_train[user]
        target_items = self.user_split[user]
        target_item = target_items[0]

        seq_len = len(seq)
        if seq_len >= self.maxlen:
            seq = seq[-self.maxlen :]
        else:
            seq = [0] * (self.maxlen - seq_len) + seq

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(target_item, dtype=torch.long),
        )
