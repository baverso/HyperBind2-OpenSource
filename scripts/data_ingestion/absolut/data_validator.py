#!/usr/bin/env python3
"""
data_validator.py

This module provides utility functions for validating and processing a pandas DataFrame.
It reports on missing (null) values, duplicate rows, and (in verbose mode) the dtypes of feature columns.
It also performs deduplication based on a given set of columns and can add sequence length features.

Usage in a Jupyter Notebook:
    from data_validator import validate_dataframe, add_sequence_lengths

    # Load your DataFrame (e.g., from a CSV)
    df = pd.read_csv("your_file.csv")

    # Optionally, add sequence length columns.
    df = add_sequence_lengths(df)

    # Validate the DataFrame and deduplicate.
    df_validated = validate_dataframe(df, verbose=True)
"""

import numpy as np


def validate_dataframe(df, verbose=False, dedup_columns=None):
    """
    Validate and check a pandas DataFrame.

    This function performs the following:
      - Prints the initial DataFrame shape.
      - Checks for missing values and prints a report of columns with nulls.
      - (In verbose mode) Prints the data types of all columns.
      - Reports the number of rows before and after deduplication based on the provided columns.

    Parameters:
        df (pd.DataFrame): The DataFrame to validate.
        verbose (bool): If True, prints additional details (e.g., data types).
        dedup_columns (list, optional): List of column names used for deduplication.
            Defaults to ['ID_slide_Variant', 'CDR3'] if not provided.

    Returns:
        pd.DataFrame: The deduplicated DataFrame.
    """
    if dedup_columns is None:
        dedup_columns = ['ID_slide_Variant', 'CDR3','antigen']

    # Report initial DataFrame shape
    initial_shape = df.shape
    print(f"Initial DataFrame shape: {initial_shape}")

    # Check for missing values in each column
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("Missing values found:")
        print(null_counts[null_counts > 0])
    else:
        print("No missing values found.")

    # Verbose report: Print data types of feature columns
    if verbose:
        print("\nData types of columns:")
        print(df.dtypes)

    # Check for missing classes
    antigen_cohorts = df['antigen'].unique().tolist()
    print(f'{len(antigen_cohorts)} antigen cohorts: {antigen_cohorts}')
    for cohort in antigen_cohorts:
        classes = df[df['antigen']==cohort]['binding_class'].unique().tolist()
        class_count=len(classes)
        if class_count < 4:
            print(f'Warning: less than 4 classes for {cohort}: {classes}')    

    # Report deduplication statistics
    print(f"\nRows before deduplication on {dedup_columns}: {df.shape[0]}")
    df_dedup = df.drop_duplicates(subset=dedup_columns)
    print(f"Rows after deduplication on {dedup_columns}: {df_dedup.shape[0]}")

    return df_dedup


def add_sequence_lengths(df):
    """
    Add sequence length columns for 'CDR3' and 'antigen_sequence' to the DataFrame.

    Creates two new columns:
      - 'CDR3_length': Length of the sequence in the 'CDR3' column.
      - 'antigen_seq_length': Length of the sequence in the 'antigen_sequence' column (if it's a string).

    Parameters:
        df (pd.DataFrame): The DataFrame that contains 'CDR3' and 'antigen_sequence' columns.

    Returns:
        pd.DataFrame: The DataFrame with the new length columns.
    """
    if 'CDR3' in df.columns:
        df['CDR3_length'] = df['CDR3'].apply(len)
    else:
        print("Warning: 'CDR3' column not found.")

    if 'antigen_sequence' in df.columns:
        df['antigen_seq_length'] = df['antigen_sequence'].apply(
            lambda x: len(x) if isinstance(x, str) else np.nan)
    else:
        print("Warning: 'antigen_sequence' column not found.")

    return df