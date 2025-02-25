#!/usr/bin/env python3
"""
prepare_dataset.py

This script ingests, validates, splits, and (optionally) saves a large processed dataset for a
contrastive learning task. It leverages utility modules (data_ingestion.py and data_validator.py) to:
    - Load processed CSV files from a specified input directory.
    - Add additional features (e.g., sequence lengths) and validate the dataset,
      reporting missing values, duplicates, and (if verbose) data types.
    - Split the data into train, validation, and test sets using stratified sampling
      by the 'binding_class' column.
    - Optionally save the resulting datasets to CSV files in an output directory.

When run from the command line, the script writes out files.
When imported as a module, the process_dataset() function returns the DataFrames.

Usage (from the command line):
    python prepare_dataset.py --input_dir <input_directory> --output_dir <output_directory>
          [--test_size 0.20] [--val_size 0.25] [--selection 3] [--verbose]

Author: Brett Averso
Date: 2023-08-31
"""

import os
import argparse
from sklearn.model_selection import train_test_split
import data_ingestion
import data_validator


def process_dataset(input_dir, output_dir, test_size, val_size, selection, verbose, write_files=True):
    """
    Process the dataset by loading, validating, splitting, and optionally saving it.

    Parameters:
        input_dir (str): Path to the input directory containing processed CSV files.
        output_dir (str): Path to the output directory where CSV files will be saved (if write_files is True).
        test_size (float): Proportion of the full dataset to reserve for testing.
        val_size (float): Proportion of the remaining dataset to reserve for validation.
        selection (str or int): Parameter for data_ingestion.load_processed_csvs (e.g., "all" or a number).
        verbose (bool): If True, prints detailed validation reports (e.g., data types).
        write_files (bool): If True, writes out the train/val/test CSV files. If False, returns the DataFrames.

    Returns:
        If write_files is False, returns a tuple (train_df, val_df, test_df). Otherwise, returns None.
    """
    # Load processed CSV files using the data_ingestion utility.
    print("Loading processed CSV files from:", input_dir)
    df = data_ingestion.load_processed_csvs(input_dir, selection)
    print(f"Loaded DataFrame shape: {df.shape}")

    # Add sequence length features and validate the dataset.
    df = data_validator.add_sequence_lengths(df)
    df = data_validator.validate_dataframe(df, verbose=verbose)

    # Report overall binding_class distribution.
    print("\nOverall binding_class distribution:")
    print(df['binding_class'].value_counts())

    # -----------------------------------------------------------------------------
    # Split the dataset into train, validation, and test sets.
    # First, split off the test set (test_size) stratified by binding_class.
    train_val, test_df = train_test_split(
        df, test_size=test_size, stratify=df['binding_class'], random_state=42
    )
    # Then split train_val into training and validation sets.
    train_df, val_df = train_test_split(
        train_val, test_size=val_size, stratify=train_val['binding_class'], random_state=42
    )

    print(f"\nTrain set: {train_df.shape[0]} observations")
    train_df = data_validator.validate_dataframe(train_df, verbose=verbose)
    print(f"Validation set: {val_df.shape[0]} observations")
    val_df = data_validator.validate_dataframe(val_df, verbose=verbose)
    print(f"Test set: {test_df.shape[0]} observations")
    test_df = data_validator.validate_dataframe(test_df, verbose=verbose)

    if write_files:
        # Ensure the output directory exists.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the datasets.
        train_csv = os.path.join(output_dir, "train.csv")
        val_csv = os.path.join(output_dir, "val.csv")
        test_csv = os.path.join(output_dir, "test.csv")

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        print("\nDatasets saved successfully:")
        print(f"  Train: {train_csv}")
        print(f"  Validation: {val_csv}")
        print(f"  Test: {test_csv}")
        return None
    else:
        # Return the DataFrames for use in another script.
        return train_df, val_df, test_df


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Ingest, validate, and split dataset for contrastive learning."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing processed CSV files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for saving train/val/test CSV files."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Proportion of the full dataset to reserve for testing (default: 0.20)."
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.25,
        help="Proportion of the remaining dataset to reserve for validation (default: 0.25)."
    )
    parser.add_argument(
        "--selection",
        type=lambda x: x if x.lower() == "all" else int(x),
        default=1,
        help="Selection parameter for data_ingestion.load_processed_csvs (default: 1). "
             "Can be 'all' as a string or a number as an int."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for detailed validation reports."
    )
    # An optional argument to force file writing. When omitted in CLI, default is True.
    parser.add_argument(
        "--write_files",
        action="store_true",
        help="Flag indicating that datasets should be written out to CSV files. If omitted, datasets will be returned."
    )
    # Add a flag to explicitly avoid writing files (so that when imported, one can set this flag to False)
    parser.add_argument(
        "--no_write_files",
        dest="write_files",
        action="store_false",
        help="Flag to avoid writing out CSV files and return the DataFrames instead."
    )
    parser.set_defaults(write_files=True)
    return parser.parse_args()


def main():
    """
    Main function to parse arguments and process the dataset.
    """
    args = parse_args()
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        selection=args.selection,
        verbose=args.verbose,
        write_files=args.write_files
    )


if __name__ == "__main__":
    main()