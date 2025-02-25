#!/usr/bin/env python3
"""
contrastive_pair_generator.py

This script provides functionality to generate contrastive pairs from training,
validation, and test CSV datasets for a contrastive learning task. Two pairing
methods are supported:
    1. Label-based pairing: pairs are generated based solely on the 'binding_class' column.
    2. Composite pairing: pairs are generated using a composite of 'antigen' and 'binding_class',
       e.g., "3L95_X_HeroesExclusive", to define finer cohorts.

The script also provides a method to build a consolidated contrastive learning
DataFrame by merging two rows from the original dataset (using indices from a pairs DataFrame).

Usage (from the command line):
    python contrastive_pair_generator.py --train_csv <train_csv_path> --val_csv <val_csv_path> \
        --test_csv <test_csv_path> --output_dir <output_dir> [--n_pos_pairs 5000] \
        [--n_neg_pairs 20000] [--n_pos_pairs_val 1000] [--n_neg_pairs_val 2000] \
        [--pairing_method label|composite] [--seed 42] [--verbose]

Author: Brett Averso
Date: 2023-08-31
"""

import argparse
import os
import random
import pandas as pd


class ContrastivePairGenerator:
    """
    Class for generating contrastive pairs and building a consolidated contrastive dataset.
    """

    @staticmethod
    def generate_pairs(df, n_pos_pairs_per_class=5000, n_neg_pairs=20000, seed=42):
        """
        Generate positive and negative pairs based on the 'binding_class' column.

        Positive pairs: sample two different rows within the same binding_class.
        Negative pairs: sample rows from different binding_classes.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing at least the 'binding_class' column.
            n_pos_pairs_per_class (int): Number of positive pairs per binding_class.
            n_neg_pairs (int): Total number of negative pairs to generate.
            seed (int): Random seed for reproducibility.

        Returns:
            pd.DataFrame: DataFrame with columns ['ID1', 'ID2', 'pair_type'].
        """
        random.seed(seed)
        pairs = []
        for binding in df['binding_class'].unique():
            df_class = df[df['binding_class'] == binding]
            indices = df_class.index.tolist()
            if len(indices) < 2:
                continue
            for _ in range(n_pos_pairs_per_class):
                i, j = random.sample(indices, 2)
                pairs.append({'ID1': i, 'ID2': j, 'pair_type': 'positive'})
        all_indices = df.index.tolist()
        neg_generated = 0
        while neg_generated < n_neg_pairs:
            i, j = random.sample(all_indices, 2)
            if df.loc[i, 'binding_class'] != df.loc[j, 'binding_class']:
                pairs.append({'ID1': i, 'ID2': j, 'pair_type': 'negative'})
                neg_generated += 1
        return pd.DataFrame(pairs)

    @staticmethod
    def generate_composite_pairs(df, n_pos_pairs_per_cohort=5000, n_neg_pairs=20000, seed=42):
        """
        Generate positive and negative pairs using a composite class defined by concatenating
        'antigen' and 'binding_class' (e.g. "3L95_X_HeroesExclusive").

        Positive pairs are drawn from the same composite group; negative pairs are drawn from
        different composite groups.

        Parameters:
            df (pd.DataFrame): Input DataFrame that must contain 'antigen' and 'binding_class' columns.
            n_pos_pairs_per_cohort (int): Number of positive pairs per composite cohort.
            n_neg_pairs (int): Total number of negative pairs to generate.
            seed (int): Random seed for reproducibility.

        Returns:
            pd.DataFrame: DataFrame with columns ['ID1', 'ID2', 'cohort1', 'cohort2', 'pair_type'].
        """
        import random
        random.seed(seed)

        # Reset the index to ensure a unique integer index.
        df = df.copy().reset_index(drop=True)

        # Create a composite column by concatenating 'antigen' and 'binding_class'.
        df['cohort'] = df['antigen'].astype(str) + "_" + df['binding_class'].astype(str)
        pairs = []

        # Generate positive pairs within each composite cohort.
        for cohort in df['cohort'].unique():
            df_cohort = df[df['cohort'] == cohort]
            indices = df_cohort.index.tolist()
            if len(indices) < 2:
                continue  # Need at least two examples for a pair.
            for _ in range(n_pos_pairs_per_cohort):
                i, j = random.sample(indices, 2)
                pairs.append({
                    'ID1': i,
                    'ID2': j,
                    'cohort1': cohort,
                    'cohort2': cohort,
                    'pair_type': 'positive'
                })

        # Generate negative pairs from different composite cohorts.
        all_indices = df.index.tolist()
        neg_generated = 0
        while neg_generated < n_neg_pairs:
            i, j = random.sample(all_indices, 2)
            # Use .at to get a scalar value.
            cohort_i = df.at[i, 'cohort']
            cohort_j = df.at[j, 'cohort']
            if cohort_i != cohort_j:
                pairs.append({
                    'ID1': i,
                    'ID2': j,
                    'cohort1': cohort_i,
                    'cohort2': cohort_j,
                    'pair_type': 'negative'
                })
                neg_generated += 1

        pairs_df = pd.DataFrame(pairs)
        initial_count = len(pairs_df)
        pairs_df = pairs_df.drop_duplicates(
            subset=['ID1', 'ID2', 'cohort1', 'cohort2', 'pair_type']
        )
        final_count = len(pairs_df)
        print(f"Generated {initial_count} pairs; after deduplication: {final_count}")
        return pairs_df

    @staticmethod
    def build_contrastive_dataframe(train_df, pairs_df):
        """
        Build a contrastive learning DataFrame from a training DataFrame and a pairs DataFrame.

        Each row in the output DataFrame represents a contrastive pair by concatenating two rows
        from train_df based on the indices in pairs_df. Columns from the first row are suffixed with
        '_1' and those from the second row with '_2'. Pair metadata is also preserved.

        Parameters:
            train_df (pd.DataFrame): The original training data.
            pairs_df (pd.DataFrame): DataFrame with pairing info, including 'ID1', 'ID2',
                                     and optionally 'cohort1', 'cohort2', 'pair_type'.

        Returns:
            pd.DataFrame: A consolidated DataFrame for contrastive learning.
        """
        contrastive_rows = []
        for _, pair in pairs_df.iterrows():
            id1 = pair["ID1"]
            id2 = pair["ID2"]
            row1 = train_df.loc[id1].to_dict()
            row2 = train_df.loc[id2].to_dict()
            combined = {f"{k}_1": v for k, v in row1.items()}
            combined.update({f"{k}_2": v for k, v in row2.items()})
            combined["ID1"] = id1
            combined["ID2"] = id2
            if 'pair_type' in pair:
                combined["pair_type"] = pair["pair_type"]
            if 'cohort1' in pair:
                combined["cohort1"] = pair["cohort1"]
            if 'cohort2' in pair:
                combined["cohort2"] = pair["cohort2"]
            contrastive_rows.append(combined)
        return pd.DataFrame(contrastive_rows)


def main(args):
    """
    Main function to generate contrastive pairs and build contrastive learning datasets.

    This function loads train, validation, and test CSV files, generates contrastive pairs
    using the specified pairing method (label or composite), builds a contrastive DataFrame
    from the training pairs, and saves the output files to the specified output directory.

    Parameters:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    print("Loading datasets...")
    train_df = pd.read_csv(args.train_csv, index_col=None)
    val_df = pd.read_csv(args.val_csv, index_col=None)
    test_df = pd.read_csv(args.test_csv, index_col=None)

    if args.pairing_method.lower() == "composite":
        print("Using composite pairing method (antigen + binding_class).")
        pair_func = ContrastivePairGenerator.generate_composite_pairs
        # For composite pairing, use n_pos_pairs_per_cohort parameter.
        train_pairs = pair_func(train_df,
                                n_pos_pairs_per_cohort=args.n_pos_pairs,
                                n_neg_pairs=args.n_neg_pairs, seed=args.seed)
        val_pairs = pair_func(val_df,
                              n_pos_pairs_per_cohort=args.n_pos_pairs_val,
                              n_neg_pairs=args.n_neg_pairs_val, seed=args.seed)
        test_pairs = pair_func(test_df,
                               n_pos_pairs_per_cohort=args.n_pos_pairs_val,
                               n_neg_pairs=args.n_neg_pairs_val, seed=args.seed)
    else:
        print("Using label-class pairing method.")
        pair_func = ContrastivePairGenerator.generate_pairs
        train_pairs = pair_func(train_df,
                                n_pos_pairs_per_class=args.n_pos_pairs,
                                n_neg_pairs=args.n_neg_pairs, seed=args.seed)
        val_pairs = pair_func(val_df,
                              n_pos_pairs_per_class=args.n_pos_pairs_val,
                              n_neg_pairs=args.n_neg_pairs_val, seed=args.seed)
        test_pairs = pair_func(test_df,
                               n_pos_pairs_per_class=args.n_pos_pairs_val,
                               n_neg_pairs=args.n_neg_pairs_val, seed=args.seed)

    print("Number of training pairs:", train_pairs.shape[0])
    print("Number of validation pairs:", val_pairs.shape[0])
    print("Number of test pairs:", test_pairs.shape[0])

    print("Generating contrastive DataFrame for training...")
    contrastive_train = ContrastivePairGenerator.build_contrastive_dataframe(train_df, train_pairs)
    print("Generating contrastive DataFrame for validation...")
    contrastive_val = ContrastivePairGenerator.build_contrastive_dataframe(val_df, val_pairs)
    print("Generating contrastive DataFrame for test...")
    contrastive_test = ContrastivePairGenerator.build_contrastive_dataframe(test_df, test_pairs)

    os.makedirs(args.output_dir, exist_ok=True)
    train_pairs.to_csv(os.path.join(args.output_dir, "train_pairs.csv"), index=False)
    val_pairs.to_csv(os.path.join(args.output_dir, "val_pairs.csv"), index=False)
    test_pairs.to_csv(os.path.join(args.output_dir, "test_pairs.csv"), index=False)
    contrastive_train.to_csv(os.path.join(args.output_dir, "contrastive_train.csv"), index=False)
    contrastive_val.to_csv(os.path.join(args.output_dir, "contrastive_val.csv"), index=False)
    contrastive_test.to_csv(os.path.join(args.output_dir, "contrastive_test.csv"), index=False)

    print("Processing complete. Output files saved to:", args.output_dir)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate contrastive pairs from train, validation, and test CSV datasets."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="File path to the training CSV dataset."
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="File path to the validation CSV dataset."
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="File path to the test CSV dataset."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated pairs and contrastive DataFrame."
    )
    parser.add_argument(
        "--n_pos_pairs",
        type=int,
        default=5000,
        help="Number of positive pairs per class/cohort for training (default: 5000)."
    )
    parser.add_argument(
        "--n_neg_pairs",
        type=int,
        default=20000,
        help="Total number of negative pairs for training (default: 20000)."
    )
    parser.add_argument(
        "--n_pos_pairs_val",
        type=int,
        default=1000,
        help="Number of positive pairs per class/cohort for validation/test (default: 1000)."
    )
    parser.add_argument(
        "--n_neg_pairs_val",
        type=int,
        default=2000,
        help="Total number of negative pairs for validation/test (default: 2000)."
    )
    parser.add_argument(
        "--pairing_method",
        type=str,
        default="label",
        help="Pairing method to use: 'label' for label-class pairing or 'composite' for composite (antigen+binding_class) pairing (default: 'label')."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for detailed logging."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)