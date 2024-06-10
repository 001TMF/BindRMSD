import os
import numpy as np
import pandas as pd
import yaml
import argparse
from Bio.PDB import PDBParser
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from joblib import Parallel, delayed
import itertools
import matplotlib.pyplot as plt
from numba import jit

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

@jit(nopython=True)
def calculate_rmsd(ligand1_coords, ligand2_coords):
    diffs = ligand1_coords - ligand2_coords
    sq_diffs = np.sum(diffs ** 2, axis=1)
    rmsd = np.sqrt(np.mean(sq_diffs))
    return rmsd

def compute_rmsd_matrix(ligands, n_jobs=-1):
    num_ligands = len(ligands)
    rmsd_matrix = np.zeros((num_ligands, num_ligands))

    ligand_coords = [np.array([atom.get_coord() for atom in ligand.get_atoms()]) for ligand in ligands]
    indices = list(itertools.combinations(range(num_ligands), 2))
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(calculate_rmsd)(ligand_coords[i], ligand_coords[j])
        for i, j in indices
    )

    k = 0
    for i, j in indices:
        rmsd_matrix[i, j] = results[k]
        rmsd_matrix[j, i] = results[k]
        k += 1

    return rmsd_matrix

def cluster_ligands(rmsd_matrix, threshold):
    condensed_dist_matrix = squareform(rmsd_matrix)
    Z = linkage(condensed_dist_matrix, 'complete')
    cluster_labels = fcluster(Z, threshold, criterion='distance')
    return Z, cluster_labels

def get_representative_files(cluster_labels, rmsd_matrix, pdb_file_mapping):
    representative_files = []
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        avg_rmsd = [np.mean([rmsd_matrix[i][j] for j in cluster_indices]) for i in cluster_indices]
        representative_index = cluster_indices[np.argmin(avg_rmsd)]
        representative_files.append(pdb_file_mapping[representative_index])
        print(f"Cluster {cluster}: {cluster_indices}")
        for idx in cluster_indices:
            print(f"Ligand {idx + 1} from file {pdb_file_mapping[idx]}")

    return representative_files

def plot_dendrogram(Z, output_path, title="Hierarchical Clustering Dendrogram"):
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title(title)
    plt.xlabel("Ligand Index")
    plt.ylabel("Distance")
    plt.savefig(os.path.join(output_path, "dendrogram.png"))  # Save the figure
    plt.close()  # Close the plot to free up memory

def plot_clusters(rmsd_matrix, cluster_labels, output_path):
    plt.figure(figsize=(10, 7))
    plt.scatter(rmsd_matrix[:, 0], rmsd_matrix[:, 1], c=cluster_labels, cmap='viridis')
    plt.title("Cluster Scatter Plot")
    plt.xlabel("RMSD Dimension 1")
    plt.ylabel("RMSD Dimension 2")
    plt.colorbar(label='Cluster')
    plt.savefig(os.path.join(output_path, "scatter_plot.png"))  # Save the figure
    plt.close()  # Close the plot to free up memory

def main(config):
    input_directory = config['input_directory']
    threshold = config.get('clustering_threshold', 2.0)
    output_file = config['output_file']
    rmsd_csv_file = config['rmsd_csv_file']
    ligand_residue = config['ligand_residue']
    graph_output_path = config['graph_output_path']
    n_jobs = config.get('n_jobs', -1)
    enable_plots = config.get('enable_plots', False)

    pdb_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.pdb')]
    ligands = []
    pdb_file_mapping = []

    for pdb_file in pdb_files:
        ligand_list = extract_ligands(pdb_file, ligand_residue)
        ligands.extend(ligand_list)
        pdb_file_mapping.extend([pdb_file] * len(ligand_list))

    rmsd_matrix = compute_rmsd_matrix(ligands, n_jobs=n_jobs)

    # Create and export the RMSD matrix as a DataFrame
    rmsd_df = pd.DataFrame(rmsd_matrix, columns=[f"Ligand {i+1}" for i in range(len(ligands))],
                           index=[f"Ligand {i+1}" for i in range(len(ligands))])
    print("RMSD Matrix:")
    print(rmsd_df)
    rmsd_df.to_csv(rmsd_csv_file)
    print(f"RMSD matrix has been saved to {rmsd_csv_file}")

    Z, cluster_labels = cluster_ligands(rmsd_matrix, threshold)

    # Save plots if enabled
    if config.get('enable_plots', False):
        plot_dendrogram(Z, graph_output_path, title="Hierarchical Clustering Dendrogram")
        plot_clusters(rmsd_matrix, cluster_labels, graph_output_path)

    # Get representative files and save output
    representative_files = get_representative_files(cluster_labels, rmsd_matrix, pdb_file_mapping)
    with open(output_file, 'w') as out_f:
        for pdb_file in representative_files:
            out_f.write(f"{os.path.basename(pdb_file)}\n")

    print("Finished processing. Representative PDB files have been saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run processing based on provided configuration.")
    parser.add_argument('config_path', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    main(config)
