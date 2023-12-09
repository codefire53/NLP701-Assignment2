import pandas as pd
import argparse
import pandas as pd
import sys
import os
import numpy as np
from tqdm import tqdm


def evaluate_position_difference(actual_position, predicted_position, text):
    tokens = text.split(' ')
    max_dist = len(tokens)
    min_dist = 0
    dist =  abs(actual_position - predicted_position)
    return (dist-min_dist)/(max_dist-min_dist)


def standardized_print(row):
    print(f"ID: {row['id']}")
    print(f"Pred label: {row['label_pred']}")
    print(f"Boundary token pred: {row['boundary_token_pred']}")
    print(f"Gold label: {row['label_gold']}")
    print(f"Boundary token gold: {row['boundary_token_gold']}")
    print(f"Text: {row['text']}")

def print_correct_example(df):
    df = df.sort_values('diff', ascending=True)
    print("Correct Example")
    standardized_print(df.iloc[0])

def print_absolutely_incorrect_example(df):
    df = df[df['diff'] > 0].sort_values('diff', ascending=False)
    print("Absolutely Incorrect Example")
    standardized_print(df.iloc[0]) 
        
def print_slightly_incorrect_example(df):
    df = df[df['diff'] > 0].sort_values('diff', ascending=True)
    print("Slightly Incorrect Example")
    standardized_print(df.iloc[0])

def print_examples_for_analysis(pred_fpath, gold_fpath):
    print(pred_fpath)
    pred_records = pd.read_json(pred_fpath, lines=True)
    gold_records = pd.read_json(gold_fpath, lines=True)
    merged_df = pred_records.merge(gold_records, on="id", suffixes=("_pred", "_gold")) 

    merged_df['diff'] = merged_df.apply(lambda row: evaluate_position_difference(row['label_gold'], row['label_pred'], row['text']), axis=1)
    merged_df['boundary_token_gold'] = merged_df.apply(lambda row: row['text'].split(" ")[row['label_gold']], axis=1)
    merged_df['boundary_token_pred'] = merged_df.apply(lambda row: row['text'].split(" ")[row['label_pred']], axis=1)
    max_diff = merged_df['diff']
    print_correct_example(merged_df)
    print_slightly_incorrect_example(merged_df)
    print_absolutely_incorrect_example(merged_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_file_path",
        "-g",
        type=str,
        required=True,
        help="Paths to the CSV file with gold annotations.",
    )
    parser.add_argument(
        "--pred_file_path",
        "-p",
        type=str,
        required=True,
        help="Path to the CSV file with predictions",
    )
    args = parser.parse_args()

    pred_file_path = args.pred_file_path
    gold_file_path = args.gold_file_path

    print_examples_for_analysis(pred_file_path, gold_file_path)
