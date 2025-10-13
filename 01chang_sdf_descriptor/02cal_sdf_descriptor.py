import numpy as np
# Compatibility shim for older Mordred code that expects np.float
np.float = float  # Do not remove if your Mordred version relies on it

import os
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

# Initialize Mordred descriptor calculator (2D descriptors only)
calc = Calculator(descriptors, ignore_3D=True)

# Directory containing SDF files (each file is named by its CAS number, e.g., "123-45-6.sdf")
sdf_dir = "output_sdf"
rows = []

# Iterate over all SDF files in the directory
for sdf_file in os.listdir(sdf_dir):
    if not sdf_file.lower().endswith(".sdf"):
        continue

    # Extract CAS number from filename (assumes filename == CAS)
    cas = os.path.splitext(sdf_file)[0]
    sdf_path = os.path.join(sdf_dir, sdf_file)

    # Read SDF file; take the first valid molecule if multiple conformers/entries exist
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = None
    for m in suppl:
        if m is not None:
            mol = m
            break

    if mol is None:
        print(f"Failed to read a valid molecule from {sdf_file}")
        continue

    try:
        # Compute descriptors; result is a MordredDescriptorValues object
        descs = calc(mol)
        # Convert to a Python dict
        desc_dict = descs.asdict()
    except Exception as e:
        print(f"Descriptor calculation failed for {cas}: {e}")
        continue

    # Attach CAS number
    desc_dict["CAS"] = cas
    rows.append(desc_dict)

# Combine all rows into a single DataFrame and write to Excel
if rows:
    df = pd.DataFrame(rows)

    # Move CAS column to the first position if present
    if "CAS" in df.columns:
        cols = ["CAS"] + [c for c in df.columns if c != "CAS"]
        df = df[cols]

    # Replace non-finite values commonly produced by Mordred with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Write a single sheet where the first row is descriptor names and the first column is CAS
    output_excel = "molecular_descriptors.xlsx"
    df.to_excel(output_excel, index=False, sheet_name="Descriptors")
    print(f"All molecular descriptors were saved to {output_excel}")
else:
    print("No descriptor data were computed.")
