import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer, Select, PDBIO
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from joblib import Parallel, delayed
import itertools
from numba import cuda, float32


class FMNSelect(Select):
    def accept_residue(self, residue):
        return residue.get_resname() == 'FMN'


def extract_ligands(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('ligand', pdb_file)
    fmn_ligands = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == 'FMN':
                    fmn_ligands.append(residue)

    return fmn_ligands


@cuda.jit
def calculate_rmsd_gpu(atoms1, atoms2, rmsd):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tid < atoms1.shape[0]:
        diff = atoms1[tid] - atoms2[tid]
        rmsd[tid] = diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2


def calculate_rmsd(ligand1, ligand2):
    atoms1 = np.array([atom.get_coord() for atom in ligand1.get_atoms()])
    atoms2 = np.array([atom.get_coord() for atom in ligand2.get_atoms()])

    if atoms1.shape[0] != atoms2.shape[0]:
        raise ValueError("The ligands do not have the same number of atoms.")

    atoms1_device = cuda.to_device(atoms1)
    atoms2_device = cuda.to_device(atoms2)
    rmsd_device = cuda.device_array(atoms1.shape[0], dtype=float32)

    threads_per_block = 256
    blocks_per_grid = (atoms1.shape[0] + (threads_per_block - 1)) // threads_per_block
    calculate_rmsd_gpu[blocks_per_grid, threads_per_block](atoms1_device, atoms2_device, rmsd_device)

    rmsd = rmsd_device.copy_to_host()
    return np.sqrt(np.mean(rmsd))


def compute_rmsd_matrix(ligands, n_jobs=-1):
    num_ligands = len(ligands)
    rmsd_matrix = np.zeros((num_ligands, num_ligands))

    def compute_pairwise_rmsd(i, j):
        return calculate_rmsd(ligands[i], ligands[j])

    indices = list(itertools.combinations(range(num_ligands), 2))

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compute_pairwise_rmsd)(i, j)
        for i, j in indices
    )

    k = 0
    for i, j in indices:
        rmsd_matrix[i, j] = results[k]
        rmsd_matrix[j, i] = results[k]
        k += 1

    return rmsd_matrix


def cluster_ligands(rmsd_matrix, threshold=0.5):
    condensed_dist_matrix = squareform(rmsd_matrix)
    Z = linkage(condensed_dist_matrix, 'average')
    cluster_labels = fcluster(Z, threshold, criterion='distance')

    return cluster_labels


def get_representative_files(cluster_labels, rmsd_matrix, pdb_file_mapping):
    representative_files = []
    unique_clusters = np.unique(cluster_labels)

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        print(f"Cluster {cluster}: {cluster_indices}")
        for idx in cluster_indices:
            print(f"Ligand {idx} from file {pdb_file_mapping[idx]}")

        avg_rmsd = []
        for i in cluster_indices:
            avg_rmsd.append(np.mean([rmsd_matrix[i][j] for j in cluster_indices]))

        representative_index = cluster_indices[np.argmin(avg_rmsd)]
        representative_files.append(pdb_file_mapping[representative_index])

    return representative_files


def main(input_directory, output_file, rmsd_csv_file, n_jobs=-1):
    pdb_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.endswith('.pdb')]
    ligands = []
    pdb_file_mapping = []

    for pdb_file in pdb_files:
        ligand_list = extract_ligands(pdb_file)
        ligands.extend(ligand_list)
        pdb_file_mapping.extend([pdb_file] * len(ligand_list))

    print(f"Found {len(ligands)} FMN ligands in {len(pdb_files)} PDB files.")
    print("Starting RMSD matrix computation...")

    rmsd_matrix = compute_rmsd_matrix(ligands, n_jobs=n_jobs)
    print("Finished RMSD matrix computation. Starting clustering...")

    rmsd_df = pd.DataFrame(rmsd_matrix, columns=[f"Ligand {i}" for i in range(len(ligands))],
                           index=[f"Ligand {i}" for i in range(len(ligands))])
    print("RMSD Matrix:")
    print(rmsd_df)
    rmsd_df.to_csv(rmsd_csv_file)
    print(f"RMSD matrix has been saved to {rmsd_csv_file}")

    cluster_labels = cluster_ligands(rmsd_matrix)
    print(f"Cluster labels: {cluster_labels}")
    print("Finished clustering. Starting file representation selection...")

    representative_files = get_representative_files(cluster_labels, rmsd_matrix, pdb_file_mapping)

    with open(output_file, 'w') as out_f:
        for pdb_file in representative_files:
            out_f.write(f"{os.path.basename(pdb_file)}\n")

    print(f"Representative PDB files have been saved to {output_file}")


if __name__ == "__main__":
    input_directory = '/Users/tristanfarmer/Documents/PhotoenzymeDesignProject/data/FMN-aposteriori-pdb/protRestraint'  # Update this path
    output_file = 'test-threshold0.5-representative_ligands.txt'
    rmsd_csv_file = 'tes-threshold0.5-rmsd_matrix.csv'  # File to save the RMSD matrix
    n_jobs = -1  # Specify the number of cores you want to use
    main(input_directory, output_file, rmsd_csv_file, n_jobs)
