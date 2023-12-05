import warnings
from collections import defaultdict
from datetime import datetime
import argparse
import os


import numpy as np
import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from rich import print
from rich.progress import track
from tqdm import tqdm

from model import BiLSTMCNNWordEmbed
from data import Data, LSTMDataset, collate_fn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        device_type = "cuda"
    else:
        device_type = "cpu"
    return torch.device(device_type)

device = get_device()
warnings.filterwarnings("ignore")

def flatten_labels(labels, sent_lengths):
    return torch.cat([labels[idx,:sent_length] for idx, sent_length in enumerate(sent_lengths)]).detach().cpu().numpy()

def evaluate_pred(preds, actual):
    f1 = f1_score(actual, preds, average="macro")
    p = precision_score(actual, preds, average="macro")
    r = recall_score(actual, preds, average="macro")
    return f1, p, r

def print_metrics_avg(metrics_map):
    for metric_name, scores_lst in metrics_map.items():
        print(f"Average {metric_name}: {sum(scores_lst)/len(scores_lst)}")

def plot_metric_scores(train_metrics, val_metrics):
    for metric_name, scores_lst in train_metrics.items():
        n_epochs = [i for i in range(len(scores_lst))]
        train_scores = train_metrics[metric_name]
        val_scores = val_metrics[metric_name]
        plt.plot(n_epochs, train_scores, label="train")
        plt.plot(n_epochs, val_scores, label="val")
        plt.title(f"{metric_name} scores over epoch")
        plt.xlabel("epoch")
        plt.ylabel("scores")
        plt.legend()
        plt.savefig(f"./figures/{metric_name}.png")
        plt.show()

# To make a reproducible output
def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_data_loader(train_file, val_file):
    train_dataset = LSTMDataset(train_file)
    val_dataset = LSTMDataset(val_file)

    kwargs = {
        "batch_size":32,
        "collate_fn":collate_fn,
        "num_workers":0,
    }



    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **kwargs
    )

    return train_loader, val_loader

def train(args):
    init_seed(args.seed)
    initial_embedding_dim, embedding_dim, hidden_dim, learning_rate, weight_decay = args.initial_embedding_dim, args.embedding_dim, args.hidden_dim, args.learning_rate, args.weight_decay 
    target_size = Data.label_size
    char_vocab_size = Data.char_vocab_size
    ignore_label = target_size
    num_epochs = args.num_epochs

    model = BiLSTMCNNWordEmbed(char_vocab_size, hidden_dim, embedding_dim, initial_embedding_dim,
                        target_size).to(device)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=ignore_label
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    train_file = '../subtaskC_train.jsonl'
    val_file = '../subtaskC_dev.jsonl'

    train_loader, val_loader = load_data_loader(train_file, val_file)

    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    checkpoints_dir = "./checkpoints"

    for epoch in (range(num_epochs)):
        model.train() 
        avg_loss = 0
        train_targets, train_preds = [], []
        if epoch != 0:
            print("Training...")
            for sentence, label, sent_lens in tqdm(train_loader):
                model.zero_grad()
                scores = model(sentence)
                loss = loss_fn(
                    scores.view(-1, scores.shape[-1]),
                    label.view(-1),
                )
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
                train_targets.extend(flatten_labels(label, sent_lens))
                train_preds.extend(flatten_labels(scores.argmax(axis=-1), sent_lens))
        train_f1, train_p, train_r = evaluate_pred(train_preds, train_targets)
        train_metrics["f1_score"].append(train_f1)
        train_metrics["p_score"].append(train_p)
        train_metrics["r_score"].append(train_r)

        model.eval()
        avg_val_loss = 0
        val_targets, val_preds = [], []
        for sentence, label, sent_lens in tqdm(val_loader):
            scores = model(sentence)
            avg_val_loss += loss_fn(
                scores.view(-1, scores.shape[-1]),
                label.view(-1),
                ).item() / len(val_loader)
            val_targets.extend(flatten_labels(label, sent_lens))
            val_preds.extend(flatten_labels(scores.argmax(axis=-1), sent_lens))
        
        val_f1, val_p, val_r = evaluate_pred(val_preds, val_targets)
        print(
                f"Epoch {epoch}/{num_epochs}: train_loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}",
                f",train_f1={train_f1:.4f}, val_f1={val_f1:.4f}",
                f",train_precision={train_p:.4f}, val_precision={val_p:.4f}",
                f",train_recall={train_r:.4f}, val_recall={val_r:.4f}",
            )
        val_metrics["f1_score"].append(val_f1)
        val_metrics["p_score"].append(val_p)
        val_metrics["r_score"].append(val_r)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        torch.save(model.state_dict(), os.path.join(checkpoints_dir, f"epoch-{epoch}.pt"))
    print("Training")
    print_metrics_avg(train_metrics)

    print("Validation")
    print_metrics_avg(val_metrics)

    plot_metric_scores(train_metrics, val_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial_embedding_dim", type=int, default=21)
    parser.add_argument("--embedding_dim", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)


    args = parser.parse_args() 
    
    train(args)


