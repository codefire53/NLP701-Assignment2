from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import numpy as np
import torch
import pandas as pd
import random
from dataclasses import dataclass
from collections import Counter

def get_device():
    if torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"
    return torch.device(device_type)
device = get_device()

train_file = '../subtaskC_train.jsonl'
val_file = '../subtaskC_dev.jsonl'

#init seed
seed=42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def count_tokens_and_characters(text):
        # Tokenize the text into words (assuming words are separated by spaces)
        tokens = text.lower().split(" ")

        # Count token occurrences
        token_counts = Counter(tokens)

        # Count character occurrences
        character_counts = Counter(text)

        # Sort counts in descending order
        sorted_token_counts = token_counts.most_common()
        sorted_character_counts = character_counts.most_common()

        return sorted_token_counts, sorted_character_counts

def load_data(data_file):
    with open(data_file, 'r') as f:
        data =  [json.loads(line) for line in f]
    return data

def load_vocab(train_file):
    train_data = load_data(train_file)
    chars_cnt = dict()
    tokens_cnt = dict()
    for row in train_data:
        text = row['text']
        sorted_tokens, sorted_characters = count_tokens_and_characters(text)
        for token, cnt in sorted_tokens:
            if token not in tokens_cnt:
                tokens_cnt[token] = cnt
            else:
                tokens_cnt[token] += cnt
        for ch, cnt in sorted_characters:
            if ch not in chars_cnt:
                chars_cnt[ch] = cnt
            else:
                chars_cnt[ch] += cnt 
    chars_sorted = sorted(chars_cnt.items(), key=lambda x:x[1], reverse=True)
    chars_sorted = [ch for ch,  _ in chars_sorted]
    tokens_sorted = sorted(tokens_cnt.items(), key=lambda x:x[1], reverse=True)
    tokens_sorted = [tok for tok,  _ in tokens_sorted]

    #token mapping
    tok2id = {"<pad>": 0, "<unk>":1, "<s>":2, "</s>": 3}
    tok2id.update({tok: tok_id for tok_id, tok in enumerate(tokens_sorted)})
    id2tok = {tok_id: tok for tok_id, tok in tok2id.items()}

    #char mapping
    char2id = {"<pad>": 0, "<unk>":1, "<s>":2, "</s>": 3}
    char2id.update({char: char_id for char_id, char in enumerate(chars_sorted)})
    id2char = {char_id: char for char_id, char in char2id.items()}

    return tok2id, id2tok, char2id, id2char

@dataclass
class Data:
 

    tok2id, id2tok, char2id, id2char = load_vocab(train_file)

    #size
    char_vocab_size = len(char2id)
    token_vocab_size = len(tok2id)
    label_size = 2

    #unknown id
    unknown_ch_id = char2id["<unk>"]
    unknown_token_id = tok2id["<unk>"]

    #start id
    start_ch_id = char2id["<s>"]
    start_token_id = tok2id["<s>"]

    #end id
    end_ch_id = char2id["</s>"]
    end_token_id = tok2id["</s>"]

    #pad id
    pad_ch_id = char2id["<pad>"]
    pad_token_id = tok2id["<pad>"]

    @staticmethod
    def convert_to_char_ids(mapping, data, max_length, unk_id, pad_id, start_id, end_id):
        input_ids = []

        for sent in data:
            sent_input_ids = []
            tokens = sent
            for token in tokens:
                char_input_ids = [start_id]
                max_ch = max_length-2
                for ch in token[:max_ch]:
                    char_input_ids.append(mapping.get(ch, unk_id))
                char_input_ids.append(end_id)
                if len(char_input_ids) < max_length:
                    char_input_ids.extend([pad_id]*(max_length-len(char_input_ids)))
                sent_input_ids.append(char_input_ids)
            input_ids.append(sent_input_ids)
        return input_ids
    
    @staticmethod
    def convert_to_label_ids(txt, labels):
        sent_labels = []
        for sent, label_pos in zip(txt, labels):
            tokens = sent
            if label_pos > 0:
                token_labels = [0]*(label_pos) + [1]*(len(tokens)-label_pos)
            else:
                token_labels = [1]*(len(tokens))
            assert len(token_labels) == len(tokens)
            sent_labels.append(token_labels)
        return sent_labels
    
class LSTMDataset(Dataset):
    def __init__(self, data_file, max_length=64):
        data = pd.read_json(data_file, lines=True, orient='records')
        self.X = data["text"].apply(lambda txt: txt.lower().split(" "))
        self.y = data["label"]
        self.max_length = max_length
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.max_length



def collate_fn(batch):
    X,y, max_length = zip(*batch)

    sent_lens = torch.LongTensor([len(sent) for sent in X]).to(device)
    X = Data.convert_to_char_ids(Data.char2id, X, max_length[0], Data.unknown_ch_id, Data.pad_ch_id, Data.start_ch_id, Data.end_ch_id)
    y = Data.convert_to_label_ids(X, y)
    X = pad_sequence(
        [torch.LongTensor(i).to(device) for i in X],
        batch_first=True,
    )
    y = pad_sequence(
        [torch.LongTensor(i).to(device) for i in y],
        batch_first=True,
        padding_value=2,
    )
    return X.to(device), y.to(device), sent_lens



