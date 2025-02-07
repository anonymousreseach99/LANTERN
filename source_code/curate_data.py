import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

def curateDAVIS(train_path, val_path, test_path, updated_train_path, updated_val_path, updated_test_path,
                ligand_id_path, protein_id_path) :
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    with open(ligand_id_path, 'r') as f :
        id2ligand = json.load(f)
    with open(protein_id_path, 'r') as f :
        id2protein = json.load(f)
    ligand2ids = {value: key for key, value in id2ligand.items()}
    protein2ids = {value : key for key, value in id2protein.items()}
    
    def addColumnCSV(df, csv_out_path) :
        df['ligand_id'] = df['SMILES'].map(ligand2ids)
        df['protein_id'] = df['Target Sequence'].map(protein2ids)
        df.to_csv(csv_out_path, index=False)
        print(f"Save the updated csv file to {csv_out_path}")
    
    addColumnCSV(train_df, updated_train_path)
    addColumnCSV(val_df, updated_val_path)
    addColumnCSV(test_df, updated_test_path)

def do_statistics_with_kiba(path2settings) :
    try:
        # Open and read the file
        with open(path2settings, 'r') as file:
            content = file.read()
        
        # Parse the array (assuming indices are comma-separated or space-separated)
        indices = [int(i) for i in content.split()]
        
        # Return the count of elements
        return len(indices)
    except Exception as e:
        print(f"Error processing the file: {e}")
        return 0

def kiba_add_label_column(kiba_path, output_path):
    # Load the KIBA dataset
    kiba_df = pd.read_csv(kiba_path)
    
    # Add a new column named 'Label' based on the value in the KIBA Score column
    kiba_df['Label'] = (kiba_df['Ki , Kd and IC50  (KIBA Score)'] >= 12.1).astype(int)
    
    # Save the updated dataset
    kiba_df.to_csv(output_path, index=False)
    print(f"Updated KIBA dataset with 'Label' column saved to {output_path}")

def split_kiba_dataset(kiba_path, test_indices_path, train_indices_path, output_dir):
    """
    Splits the KIBA dataset into train, validation, and test CSV files with a random 7:1 train/validation split.
    
    :param kiba_path: Path to the full KIBA dataset CSV file.
    :param test_indices_path: Path to the file containing test indices.
    :param train_indices_path: Path to the file containing train indices.
    :param output_dir: Directory to save the split datasets.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the full KIBA dataset
    kiba_df = pd.read_csv(kiba_path)

    # Parse indices from the settings files
    def parse_indices(file_path):
        with open(file_path, 'r') as f:
            content = f.read().strip()
        indices = content.replace('[', '').replace(']', '').split(',')
        return [int(idx.strip()) for idx in indices]

    test_indices = parse_indices(test_indices_path)
    train_indices = parse_indices(train_indices_path)

    # Print lengths of test and train fold settings
    print(f"Length of test indices: {len(test_indices)}")
    print(f"Length of train indices: {len(train_indices)}")

    # Split the train indices into train and validation sets (7:1 random split)
    train_split, val_split = train_test_split(train_indices, test_size=0.125, random_state=42)

    # Generate the splits
    test_df = kiba_df.iloc[test_indices]
    train_df = kiba_df.iloc[train_split]
    val_df = kiba_df.iloc[val_split]

    
    # Print lengths of test and train fold settings
    print(f"Length of test df: {len(test_df)}")
    print(f"Length of train df: {len(train_df)}")
    print(f"Length of valid df: {len(val_df)}")


    # Save the splits to CSV files
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)

    print(f"Splits saved to {output_dir}: train.csv, val.csv, test.csv")

def kiba_generate_mappings(kiba_path, output_dir):
    # Load the KIBA dataset
    kiba_df = pd.read_csv(kiba_path)
    
    # Create mappings
    ligand_to_smiles = kiba_df.set_index('CHEMBLID')['compound_iso_smiles'].to_dict()
    protein_to_sequence = kiba_df.set_index('ProteinID')['target_sequence'].to_dict()
    
    # Save the mappings to JSON files
    os.makedirs(output_dir, exist_ok=True)
    ligand_path = os.path.join(output_dir, 'ligand_to_smiles.json')
    protein_path = os.path.join(output_dir, 'protein_to_sequence.json')
    
    with open(ligand_path, 'w') as f:
        json.dump(ligand_to_smiles, f, indent=4)
    
    with open(protein_path, 'w') as f:
        json.dump(protein_to_sequence, f, indent=4)
    
    print(f"Ligand and protein mappings saved to {output_dir}")

def get_column_names(file_path):
    """
    Prints and returns all column names of the dataset.

    :param file_path: Path to the KIBA dataset CSV file.
    :return: List of column names.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        # Get the column names
        column_names = df.columns.tolist()
        print("Column Names:", column_names)
        return column_names
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def count_unique_relations(csv_path):
    """
    Counts the number of unique relations in a CSV file with columns:
    ligand_id, protein_id, Label.

    :param csv_path: Path to the CSV file.
    :return: Number of unique relations.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)

        # Ensure the required columns are present
        required_columns = {"DrugBank ID", "Gene", "Label"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain the columns: {required_columns}")

        # Count unique relations (rows are considered unique by default)
        unique_relations = df.drop_duplicates(subset=["DrugBank ID", "Gene", "Label"])
        num_unique_relations = len(unique_relations)

        print(f"Number of unique relations: {num_unique_relations}")
        return num_unique_relations

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def find_repeated_triples(csv_path):
    """
    Finds and prints the row numbers of repeated triples (duplicate rows) 
    in a dataset containing columns: ligand_id, protein_id, Label.

    :param csv_path: Path to the CSV file.
    """
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)

        # Ensure the required columns are present
        required_columns = {"ligand_id", "protein_id", "Label"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file must contain the columns: {required_columns}")

        # Find duplicate rows and include all occurrences of duplicates
        duplicates = df[df.duplicated(subset=["ligand_id", "protein_id", "Label"], keep=False)]

        if duplicates.empty:
            print("No repeated triples found.")
            return

        # Print row numbers and the repeated rows
        print("Repeated triples and their row numbers:")
        for index, row in duplicates.iterrows():
            print(f"Row {index}: {row.to_dict()}")

    except Exception as e:
        print(f"An error occurred: {e}")

def curate_deepddi() :
    train_df = pd.read_csv(deepddi_train_csv_path)
    val_df = pd.read_csv(deepddi_val_csv_path)
    test_df = pd.read_csv(deepddi_test_csv_path)

    drug_deep_list = pd.read_csv(deepddi_list)
    smiles2id_dict = drug_deep_list.set_index('smiles')['drugbank_id'].to_dict()

    os.makedirs(deepddi_dir, exist_ok=True)

    def add2Columns(df, smiles2id_dict, output_file) :
        df['smiles_1_id'] = df['smiles_1'].map(smiles2id_dict)
        df['smiles_2_id'] = df['smiles_2'].map(smiles2id_dict)
        df.to_csv(output_file, index=False)
        print(f"Updated file saved to: {output_file}")
    
    add2Columns(train_df, smiles2id_dict, deepddi_train_csv_path)
    add2Columns(val_df, smiles2id_dict, deepddi_val_csv_path)
    add2Columns(test_df, smiles2id_dict, deepddi_test_csv_path)

def check_unique_triples_highppi(file_path) :
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure consistent ordering of item_id_a and item_id_b
    df['entity_1'] = df[['item_id_a', 'item_id_b']].min(axis=1)
    df['entity_2'] = df[['item_id_a', 'item_id_b']].max(axis=1)
    
    # Create normalized triples
    df_normalized = df[['entity_1', 'entity_2', 'score']].drop_duplicates()
    
    # Count unique triples
    unique_count = len(df_normalized)
    print("The number of unique triples in the file is : ", unique_count)
    return unique_count

def yeast_tsv_to_csv() :
    import csv

    # Load the TSV file
    tsv_file = "./yeast/protein_dictionary.tsv"

    # Save it as a CSV file
    csv_file = "./yeast/protein_dictionary.csv"

    # Convert TSV to CSV
    with open(tsv_file, 'r') as tsv_in, open(csv_file, 'w', newline='') as csv_out:
        tsv_reader = csv.reader(tsv_in, delimiter='\t')
        csv_writer = csv.writer(csv_out, delimiter=',')
        
        for row in tsv_reader:
            csv_writer.writerow(row)

    print(f"Converted {tsv_file} to {csv_file}")

def create_yeast_val_set() :
    # Load the train set
    train_path = "./yeast/train.csv"
    train_old_path = "./yeast/train_old.csv"
    val_path = "./yeast/val.csv"
    train_df = pd.read_csv(train_path)
    
    # Split validation set (12.5% of train)
    val_size = 0.125
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42)
    
    # Save the old train set
    train_df.to_csv(train_old_path, index=False)
    
    # Save the new train and validation sets
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Validation set created with {len(val_df)} samples.")
    print(f"Old train set saved to {train_old_path}.")
    print(f"Updated train set saved to {train_path}.")
    print(f"Validation set saved to {val_path}.")




if __name__ == '__main__' :
    #curateDAVIS(davis_train_path, davis_val_path, davis_test_path, davis_updated_train, 
    #            davis_updated_val, davis_updated_test, davis_ligands_path, davis_proteins_path)
    #do_statistics_with_kiba(kiba_test_fold_settings_path)
    #kiba_add_label_column(kiba_csv_path, kiba_updated_csv_path)
    #get_column_names(kiba_csv_path)
    #split_kiba_dataset(kiba_csv_path, kiba_test_fold_settings_path, kiba_train_fold_settings_path, kiba_folder_path)
    #kiba_generate_mappings(kiba_csv_path, kiba_folder_path)
    #count_unique_relations(biosnap_csv_path)
    #curate_deepddi()
    #check_unique_triples_highppi(highppi_csv_triples_path)
    #yeast_tsv_to_csv()
    create_yeast_val_set()