import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Select
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from joblib import Parallel, delayed
import itertools
import yaml
import argparse

class FMNSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname() == self.ligand_residue

def extract_ligands(pdb_file, ligand_residue):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('ligand', pdb_file)
    ligands = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == ligand_residue:
                    ligands.append(residue)

    return ligands

def calculate_rmsd(ligand1, ligand2):
    atoms1 = list(ligand1.get_atoms())
    atoms2 = list(ligand2.get_atoms())

    if len(atoms1) != len(atoms2):
        raise ValueError("The ligands do not have the same number of atoms.")

    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    return sup.rms

def compute_rmsd_matrix(ligands, n_jobs):
    num_ligands = len(ligands)
    rmsd_matrix = np.zeros((num_ligands, num_ligands))

    ligand_coords = [ligand for ligand in ligands]  # Removed unnecessary list comprehension

    indices = list(itertools.combinations(range(num_ligands), 2))

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(lambda i, j: calculate_rmsd(ligand_coords[i], ligand_coords[j]))(i, j)
        for i, j in indices
    )

    for k, (i, j) in enumerate(indices):
        rmsd_matrix[i, j] = results[k]
        rmsd_matrix[j, i] = results[k]

    return rmsd_matrix

def cluster_ligands(rmsd_matrix, threshold):
    condensed_dist_matrix = squareform(rmsd_matrix)
    Z = linkage(condensed_dist_matrix, 'average')
    cluster_labels = fcluster(Z, threshold, criterion='distance')
    return cluster_labels

def get_representative_files(cluster_labels, rmsd_matrix, pdb_file_mapping):
    representative_files = []
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        avg_rmsd = [np.mean([rmsd_matrix[i][j] for j in cluster_indices]) for i in cluster_indices]
        representative_index = cluster_indices[np.argmin(avg_rmsd)]
        representative_files.append(pdb_file_mapping[representative_index])

    return representative_files

def main(config):
    input_directory = config['input_directory']
    output_file = config['output_file']
    rmsd_csv_file = config['rmsd_csv_file']
    ligand_residue = config['ligand_residue']
    n_jobs = config.get('n_jobs', -1)
    threshold = config.get('clustering_threshold', 2.0)

    pdb_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.pdb')]
    ligands = []

    for pdb_file in pdb_files:
        ligands.extend(extract_ligands(pdb_file, ligand_residue))

    print(f"Found {len(ligands)} FMN ligands in {len(pdb_files)} PDB files.")
    print("Starting RMSD matrix computation...")

    rmsd_matrix = compute_rmsd_matrix(ligands, n_jobs)
    cluster_labels = cluster_ligands(rmsd_matrix, threshold)
    representative_files = get_representative_files(cluster_labels, rmsd_matrix, pdb_files)

    # Create and export the RMSD matrix as a DataFrame
    rmsd_df = pd.DataFrame(rmsd_matrix, columns=[f"Ligand {i}" for i in range(len(ligands))],
                           index=[f"Ligand {i}" for i in range(len(ligands))])
    print("RMSD Matrix:")
    print(rmsd_df)
    rmsd_df.to_csv(rmsd_csv_file)
    print(f"RMSD matrix has been saved to {rmsd_csv_file}")

    with open(output_file, 'w') as out_f:
        for pdb_file in representative_files:
            out_f.write(f"{os.path.basename(pdb_file)}\n")

    print(f"Representative PDB files have been saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run processing based on provided configuration.")
    parser.add_argument('config_path', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
