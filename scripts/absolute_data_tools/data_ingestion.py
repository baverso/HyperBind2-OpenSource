#!/usr/bin/env python3
"""
data_ingestion.py

This module provides functionality to ingest and consolidate multiple processed CSV
files into a single pandas DataFrame. The processed CSV filenames are expected to follow
a naming convention such as:

    PDB_CHAIN_SUFFIX_processed.csv

where:
    - PDB is the PDB identifier (e.g., "1ADQ", "1NCB"),
    - CHAIN is the chain identifier,
    - SUFFIX is a binding class (e.g., "500kNonMascotte", "HeroesExclusive", etc.).

The main function, load_processed_csvs(), allows the user to specify the selection
criteria in one of three ways:
    - Passing 'all' to load all available CSV files.
    - Passing a list of PDB IDs (e.g., ['1NCB', '1MHP', '1ADQ']) to load only files
      corresponding to those PDB IDs.
    - Passing an integer to randomly select that many unique PDB IDs and load all files
      corresponding to them.

The selected files are read with pandas and concatenated into a single DataFrame.
"""

import os
import random
import pandas as pd


def load_processed_csvs(directory, selection):
    """
    Load processed CSV files from a specified directory and concatenate them into one DataFrame.

    The CSV files are expected to follow a naming convention like:
        PDB_CHAIN_SUFFIX_processed.csv
    where:
        - "PDB" is the PDB identifier (e.g., "1ADQ", "1NCB"),
        - "CHAIN" is the chain identifier,
        - "SUFFIX" is the binding class (e.g., "500kNonMascotte", "HeroesExclusive", etc.),
        - The filename ends with "_processed.csv".

    Parameters:
        directory (str): The path to the directory containing the processed CSV files.
        selection (str | list | int):
            - If 'all' (case insensitive), all files in the directory will be loaded.
            - If a list of strings (e.g., ['1NCB', '1MHP', '1ADQ']), only files whose filename's
              PDB identifier (the part before the first underscore) is in the list will be loaded.
            - If an integer, that many unique PDB IDs will be randomly selected from the available
              files, and all files corresponding to those PDB IDs will be loaded.

    Returns:
        pd.DataFrame: A single DataFrame that is the concatenation of all the loaded CSV files.
                      If no files are loaded, an empty DataFrame is returned.

    Raises:
        ValueError: If the selection parameter is not one of the supported types.

    Example usage:
        # Load files for specific PDB IDs:
        df_subset = load_processed_csvs("/path/to/csvs", ['1NCB', '1MHP', '1ADQ'])

        # Load all CSV files:
        df_all = load_processed_csvs("/path/to/csvs", 'all')

        # Load a random selection of 3 unique PDB ID sets:
        df_random = load_processed_csvs("/path/to/csvs", 3)
    """
    # List all files in the directory that end with "_processed.csv"
    all_files = [f for f in os.listdir(directory) if f.endswith('_processed.csv')]

    # Extract unique PDB IDs from the filenames. Assumes the filename format is "PDB_CHAIN_SUFFIX_processed.csv"
    available_pdb_ids = set()
    for f in all_files:
        tokens = f.split('_')
        if tokens:
            available_pdb_ids.add(tokens[0].upper())  # Normalize to uppercase
    available_pdb_ids = sorted(list(available_pdb_ids))

    # Determine which files to load based on the selection criteria.
    selected_files = []
    if isinstance(selection, str) and selection.lower() == 'all':
        # Select all files.
        selected_files = all_files
    elif isinstance(selection, list):
        # Assume selection is a list of PDB IDs; normalize them to uppercase.
        selection_upper = {x.upper() for x in selection}
        selected_files = [f for f in all_files if f.split('_')[0].upper() in selection_upper]
    elif isinstance(selection, int):
        # Randomly select a given number of unique PDB IDs.
        if selection > len(available_pdb_ids):
            print(f"Selection integer {selection} exceeds the available unique PDB IDs "
                  f"({len(available_pdb_ids)}). Using all available PDB IDs.")
            selected_pdb_ids = available_pdb_ids
        else:
            selected_pdb_ids = random.sample(available_pdb_ids, selection)
        selected_files = [f for f in all_files if f.split('_')[0].upper() in selected_pdb_ids]
    else:
        raise ValueError("Selection must be 'all', a list of PDB IDs, or an integer.")

    # Read each selected file into a DataFrame.
    dataframes = []
    for file_name in selected_files:
        file_path = os.path.join(directory, file_name)
        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    # Concatenate all the DataFrames into one, if any files were loaded.
    if dataframes:
        concatenated_df = pd.concat(dataframes, ignore_index=True)
    else:
        concatenated_df = pd.DataFrame()

    return concatenated_df


def main():
    """
    Main function to demonstrate the usage of the load_processed_csvs function.
    """
    # Example directory containing the processed CSV files.
    directory = "/path/to/your/csv_files"

    # Option 1: Load files for a specific set of PDB IDs.
    pdb_ids = ['1NCB', '1MHP', '1ADQ']
    df_subset = load_processed_csvs(directory, pdb_ids)
    print("Loaded DataFrame for specific PDB IDs:", df_subset.shape)

    # Option 2: Load all processed files.
    df_all = load_processed_csvs(directory, 'all')
    print("Loaded DataFrame for all files:", df_all.shape)

    # Option 3: Load a random selection of 3 unique PDB ID sets.
    df_random = load_processed_csvs(directory, 3)
    print("Loaded DataFrame for random selection:", df_random.shape)


if __name__ == "__main__":
    main()