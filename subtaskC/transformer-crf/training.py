import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.trainer_callback import TrainerState
from torch.utils.data import Dataset, DataLoader
import transformers
from tqdm import tqdm
import pandas as pd
import numpy as np
from model import AutoModelCRF
from trainer import CRFTrainer

import logging
import glob
import os
import sys
from tqdm import tqdm

sys.path.insert(0, '../baseline')
from transformer_baseline import ModelConfig, DatasetConfig, TrainingArgsConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
def evaluate_position_difference(actual_position, predicted_position):
    """
    Compute the absolute difference between the actual and predicted start positions.

    Args:
    - actual_position (int): Actual start position of machine-generated text.
    - predicted_position (int): Predicted start position of machine-generated text.

    Returns:
    - int: Absolute difference between the start positions.
    """
    return abs(actual_position - predicted_position)


def get_start_position(sequence, mapping=None, token_level=True):
    """
    Get the start position from a sequence of labels or predictions.

    Args:
    - sequence (np.array): A sequence of labels or predictions.
    - mapping (np.array): Mapping from index to word for the sequence.
    - token_level (bool): If True, return positional indices; else, return word mappings.

    Returns:
    - int or str: Start position in the sequence.
    """
    # Locate the position of label '1'

    if mapping is not None:
        mask = mapping != -100
        sequence = sequence[mask]
        mapping = mapping[mask]
    index = np.where(sequence == 1)[0]
    value = index[0] if index.size else (len(sequence) - 1)

    if not token_level:
        value = mapping[value]

    return value


def evaluate_machine_start_position(
    labels, predictions, idx2word=None, token_level=False
):
    """
    Evaluate the starting position of machine-generated text in both predicted and actual sequences.

    Args:
    - labels (np.array): Actual labels.
    - predictions (np.array): Predicted labels.
    - idx2word (np.array): Mapping from index to word for each sequence in the batch.
    - token_level (bool): Flag to determine if evaluation is at token level. If True, return positional indices; else, return word mappings.

    Returns:
    - float: Mean absolute difference between the start positions in predictions and actual labels.
    """

    actual_starts = []
    predicted_starts = []

    if not token_level and idx2word is None:
        raise ValueError(
            "idx2word must be provided if evaluation is at word level (token_level=False)"
        )

    for idx in range(labels.shape[0]):
        # Remove padding
        predict, label, mapping = (
            predictions[idx][1:len(labels[idx])],
            labels[idx][1:len(labels[idx])],
            idx2word[idx][1:len(labels[idx])] if not token_level else None,
        )

        # If token_level is True, just use the index; otherwise, map to word
        predicted_value = get_start_position(predict, mapping, token_level)
        actual_value = get_start_position(label, mapping, token_level)

        predicted_starts.append(predicted_value)
        actual_starts.append(actual_value)

    position_differences = [
        evaluate_position_difference(actual, predict)
        for actual, predict in zip(actual_starts, predicted_starts)
    ]
    mean_position_difference = np.mean(position_differences)

    return mean_position_difference


def compute_metrics(p):
    pred, labels = p
    mean_absolute_diff = evaluate_machine_start_position(labels, pred, token_level=True)

    return {
        "mean_absolute_diff": mean_absolute_diff,
    }

def training_loop(model, optimizer, train_dataloader, device):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        loss = model(input_ids, attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        logger.info(f"Step {step}: {loss.item():.4f}")
        total_loss += loss.item()

    avg_loss = total_loss/len(train_dataloader)
    print(f"Training loss: {avg_loss:.4f}")

def validation_loop(model, val_dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds = model(input_ids, attention_mask)

            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
    diff_loss = compute_metrics((np.array(all_preds), np.array(all_labels)))
    return diff_loss["mean_absolute_diff"]

def predict(model, test_dataloader, device):
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask)
            all_preds.extend(preds)
    out = np.array(all_preds)
    print(out.shape)
    return out

def save_model(model_name, model, optimizer, train_mae, val_mae, epoch, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint = {
        'train_mae': train_mae,
        'val_mae': val_mae,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    model_name = model_name.replace("/", "-")
    file_path = os.path.join(output_dir, f"{model_name}-epoch-{epoch}.pt")
    print(file_path)
    torch.save(checkpoint, file_path)
    logger.info(f"Model has been saved successfully to {file_path}")

def load_best_checkpoint(checkpoints_dir: str):
    ckp_paths = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
    min_mae = 1000000
    best_ckp = ""
    for ckp_path in tqdm(ckp_paths):
        checkpoint_info = torch.load(ckp_path)
        if checkpoint_info['val_mae'] < min_mae:
            best_ckp = ckp_path
            print(f"Best checkpoint: {best_ckp}")
            min_mae = checkpoint_info['val_mae']
    logger.info(f"best checkpoint: {best_ckp}")
    return best_ckp

class Semeval_Data(torch.utils.data.Dataset):
    def __init__(self, data_path, max_length=1024, inference=False, debug=False):
        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f]
        self.inference = inference
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.max_length = max_length
        self.debug = debug

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        id = self.data[idx]["id"]
        label = None
        labels_available = "label" in self.data[idx]

        if labels_available:
            label = self.data[idx]["label"]

        if self.debug and not self.inference:
            print("Orignal Human Position: ", label)

        labels = []
        corresponding_word = []
        tokens = []
        input_ids = []
        attention_mask = []

        for jdx, word in enumerate(text.split(" ")):
            word_encoded = self.tokenizer.tokenize(word)
            sub_words = len(word_encoded)

            if labels_available:
                is_machine_text = 1 if jdx >= label else 0
                labels.extend([is_machine_text] * sub_words)

            corresponding_word.extend([jdx] * sub_words)
            tokens.extend(word_encoded)
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(word_encoded))
            attention_mask.extend([1] * sub_words)

        ###Add padding to labels as -100
        if len(input_ids) < self.max_length - 2:
            input_ids = (
                [0] + input_ids + [2] + [1] * (self.max_length - len(input_ids) - 2)
            )
            if labels_available:
                labels = [0] + labels + [labels[-1]] * (self.max_length - len(labels) - 1)

            attention_mask = (
                [1]
                + attention_mask
                + [1]
                + [0] * (self.max_length - len(attention_mask) - 2)
            )
            corresponding_word = (
                [-100]
                + corresponding_word
                + [-100] * (self.max_length - len(corresponding_word) - 1)
            )
            tokens = (
                ["<s>"]
                + tokens
                + ["</s>"]
                + ["<pad>"] * (self.max_length - len(tokens) - 2)
            )
        else:
            # Add -100 for CLS and SEP tokens
            input_ids = [0] + input_ids[: self.max_length - 2] + [2]

            if labels_available:
                labels = [0] + labels[: self.max_length - 2] + [labels[self.max_length - 3]]

            corresponding_word = (
                [-100] + corresponding_word[: self.max_length - 2] + [-100]
            )
            attention_mask = [1] + attention_mask[: self.max_length - 2] + [1]
            tokens = ["<s>"] + tokens[: self.max_length - 2] + ["</s>"]

        encoded = {}
        if labels_available:
            encoded["labels"] = torch.tensor(labels)

        encoded["input_ids"] = torch.tensor(input_ids)
        encoded["attention_mask"] = torch.tensor(attention_mask)

        if labels_available:
            if encoded["input_ids"].shape != encoded["labels"].shape:
                print("Input IDs Shape: ", encoded["input_ids"].shape)
                print("Labels Shape: ", encoded["labels"].shape)
            assert encoded["input_ids"].shape == encoded["labels"].shape

        if self.debug and not self.inference:
            print("Tokenized Human Position: ", labels.index(1))
            print("Original Human Position: ", label)
            print("Full Human Text:", text)
            print("\n")
            print("Human Text Truncated:", text.split(" ")[:label])
            print("\n")
            encoded["partial_human_review"] = " ".join(text.split(" ")[:label])

        if self.inference:
            encoded["text"] = text
            encoded["id"] = id
            encoded["corresponding_word"] = corresponding_word
        return encoded


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelConfig, DatasetConfig, TrainingArgsConfig)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("Model Arguments: ", model_args)
    print("Data Arguments: ", data_args)
    print("Training Arguments: ", training_args)

    # Set seed
    transformers.set_seed(training_args.seed)

    model_path = model_args.model_path
    model_checkpoint_dir = model_args.model_checkpoint_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Load model
    model = AutoModelCRF(model_path).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_set = Semeval_Data(data_args.train_file)
    dev_set = Semeval_Data(data_args.dev_file)

    train_dataloader = DataLoader(train_set, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_set, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
    train_eval_dataloader = DataLoader(train_set, batch_size=training_args.per_device_eval_batch_size, shuffle=False) 

    if training_args.do_train:
        logger.info("Training...")
        logger.info("*** Train Dataset ***")
        logger.info(f"Number of samples: {len(train_set)}")
        logger.info("*** Dev Dataset ***")
        logger.info(f"Number of samples: {len(dev_set)}")
        num_train_epochs = training_args.num_train_epochs
        for epoch in tqdm(range(num_train_epochs)):
            training_loop(model, optimizer, train_dataloader, device)
            train_mse = validation_loop(model, train_dataloader, device)
            val_mse = validation_loop(model, dev_dataloader, device)
            logger.info(f"Validation MSE: {val_mse:.4f}, Training MSE: {train_mse:.4f}")
            save_model(model_path, model, optimizer, train_mse, val_mse, epoch, model_checkpoint_dir)   
        logger.info("Training completed!")


    if training_args.do_predict:
        test_sets = []
        checkpoint = load_best_checkpoint(model_checkpoint_dir)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        for test_file in data_args.test_files:
            test_set = Semeval_Data(test_file, inference=True)
            test_set = DataLoader(test_set, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
            test_sets.append(test_set)
        logger.info("Predicting...")
        logger.info("*** Test Datasets ***")
        logger.info(f"Number of sets: {len(test_sets)}")

        for idx, test_set in enumerate(tqdm(test_sets)):
            logger.info(f"Test Dataset {idx + 1}")
            logger.info(f"Number of samples: {len(test_set)}")
            predictions = predict(model, test_set, device)
            corresponding_words = []
            ids = []
            for batch in test_set:
                
                corr_word = np.transpose(np.array(batch['corresponding_word']), (1, 0))
                ids.extend(batch["id"])
                corresponding_words.extend(corr_word)
                
            corresponding_words = np.array(corresponding_words)

            logger.info("Predictions completed!")
            print(predictions.shape)
            df_ids = []
            df_labels = []
            for id, pred, corr_word in tqdm(zip(ids, predictions, corresponding_words)):
                df_ids.append(id)
                df_labels.append(get_start_position(
                            pred,
                            corr_word,
                            token_level=False,
                        ))
            df = pd.DataFrame(
                {
                    "id": df_ids,
                    "label": df_labels,
                }
            )
            import os

            file_name = os.path.basename(data_args.test_files[idx])
            file_dirs = os.path.join(training_args.output_dir, "predictions")
            os.makedirs(file_dirs, exist_ok=True)
            file_path = os.path.join(file_dirs, file_name)
            records = df.to_dict("records")
            with open(file_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")