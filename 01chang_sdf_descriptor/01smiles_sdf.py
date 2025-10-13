import pandas as pd
import re
import os
from rdkit import Chem
from rdkit.Chem import AllChem


def clean_filename(filename):
    """
    Clean special characters (e.g., zero-width spaces) from a file path or name.
    :param filename: Original filename or path
    :return: Cleaned filename or path
    """
    # Remove all non-ASCII characters, including invisible or special ones
    cleaned_filename = re.sub(r'[^\x00-\x7F]+', '', filename)
    return cleaned_filename


def smiles_to_sdf(smiles, output_file, num_confs=1, optimize=True):
    """
    Convert a SMILES string to an SDF file.
    :param smiles: SMILES string
    :param output_file: Path to save the output SDF file
    :param num_confs: Number of conformers to generate (default = 1)
    :param optimize: Whether to perform energy minimization (default = True)
    :return: None
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate conformers
    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=num_confs,
        params=AllChem.ETKDGv3()
    )

    # Optional energy minimization
    if optimize:
        for conf_id in range(mol.GetNumConformers()):
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)

    # Write to SDF
    writer = Chem.SDWriter(output_file)
    for conf_id in range(mol.GetNumConformers()):
        writer.write(mol, confId=conf_id)
    writer.close()

    print(f"SDF file saved to: {output_file}")


def process_excel_to_sdf(excel_path, sheet_name='Sheet1', output_dir='output_sdf'):
    """
    Process an Excel file and generate SDF files for each compound.
    :param excel_path: Path to the Excel file
    :param sheet_name: Worksheet name (default = 'Sheet1')
    :param output_dir: Output directory (default = 'output_sdf')
    :return: None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Read Excel file
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return

    # Iterate through each row
    for index, row in df.iterrows():
        compound = row.get('Compound', '')
        cas = row.get('CAS number', '')
        smiles = row.get('Smiles number', '')

        # Skip rows with missing CAS or SMILES
        if pd.isnull(cas) or pd.isnull(smiles):
            print(f"Skipping row {index + 1}: missing CAS number or SMILES")
            continue

        # Sanitize CAS number for filenames
        safe_cas = str(cas).strip()
        illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in illegal_chars:
            safe_cas = safe_cas.replace(char, '_')

        # Clean any remaining special characters
        safe_cas = clean_filename(safe_cas)

        # Construct output path
        output_path = os.path.join(output_dir, f"{safe_cas}.sdf")

        try:
            smiles_to_sdf(smiles, output_path, num_confs=5, optimize=True)
        except Exception as e:
            print(f"Failed to process {compound} (CAS: {cas}): {str(e)}")


# Example usage
if __name__ == "__main__":
    input_file = "# Path to your input Excel file"
    process_excel_to_sdf(input_file)
