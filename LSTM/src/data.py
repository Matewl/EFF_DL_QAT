import re
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter

def basic_english_tokenizer(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

class yelp_polarity_dataset(Dataset):
    def __init__(self, data, vocab, max_len=256):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']

        tokens = basic_english_tokenizer(text)
        
        unk_idx = self.vocab['<unk>']
        indices = [self.vocab.get(token, unk_idx) for token in tokens]

        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        
        length = len(indices) if len(indices) > 0 else 1

        if len(indices) < self.max_len:
            indices += [self.vocab['<pad>']] * (self.max_len - len(indices))
            
        if not indices:
            indices = [self.vocab['<pad>']] * self.max_len

        return torch.tensor(indices, dtype=torch.long), length, torch.tensor(label, dtype=torch.float)

def build_vocab(dataset, max_tokens=20000):
    counter = Counter()
    for item in dataset:
        tokens = basic_english_tokenizer(item['text'])
        counter.update(tokens)
    
    vocab = {'<pad>': 0, '<unk>': 1}
    
    most_common = counter.most_common(max_tokens - 2)
    for i, (word, _) in enumerate(most_common):
        vocab[word] = i + 2
        
    return vocab

def get_dataloaders(batch_size=64, max_vocab_size=20000, max_len=256):
    print("Загрузка датасета Yelp Polarity")
    dataset = load_dataset('yelp_polarity')
    
    # В Yelp 560 000 примеров

    full_train = dataset['train'].shuffle(seed=42).select(range(150000))
    test_data = dataset['test'].shuffle(seed=42).select(range(15000))
    
    # Откусываем 10% на валидацию
    
    train_val_split = full_train.train_test_split(test_size=0.1, seed=42)
    train_data = train_val_split['train']
    val_data = train_val_split['test']

    print("Построение словаря...")
    vocab = build_vocab(train_data, max_tokens=max_vocab_size)
    print(f"Размер словаря: {len(vocab)} токенов.")

    train_dataset = yelp_polarity_dataset(train_data, vocab, max_len)
    val_dataset = yelp_polarity_dataset(val_data, vocab, max_len)
    test_dataset = yelp_polarity_dataset(test_data, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(vocab)