import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
from lxml import etree
import shutil


def run_plip(pdb_file, output_dir):
    """
    Run PLIP on a specified PDB file and save the results, including PyMOL session files, in the output directory.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    command = f"plip -f {pdb_file} -o {output_dir} -x -y"  # -y to generate PyMOL session files
    subprocess.run(command, shell=True, check=True)

def parse_plip_xml(xml_file):
    """
    Parse the XML output from PLIP and count different types of interactions.
    """
    tree = etree.parse(str(xml_file))
    root = tree.getroot()
    interaction_counts = {
        'hydrophobic': len(root.xpath('//hydrophobic_interaction')),
        'hydrogen_bond': len(root.xpath('//hydrogen_bond')),
        'halogen_bond': len(root.xpath('//halogen_bond')),
        'salt_bridge': len(root.xpath('//saltbridge')),
        'pi_stack': len(root.xpath('//pistacking')),
        'pi_cation': len(root.xpath('//pication'))
    }
    return interaction_counts

def score_interactions(interaction_counts):
    """
    Simple scoring function to rank the interactions. Modify weights as needed.
    """
    scores = {
        'hydrophobic': 1,
        'hydrogen_bond': 2,
        'halogen_bond': 2,
        'salt_bridge': 3,
        'pi_stack': 3,
        'pi_cation': 3
    }
    total_score = sum(interaction_counts[type] * scores[type] for type in interaction_counts)
    return total_score

def batch_process_plip(input_dir, output_dir):
    """
    Process all PDB files in the input directory with PLIP, score, and rank them.
    Skips PLIP if output files already exist.
    """
    results = []
    pdb_files = list(Path(input_dir).rglob('*.pdb'))
    print(f"Found {len(pdb_files)} PDB files to process.")

    for pdb_file in pdb_files:
        output_path = output_dir / pdb_file.stem
        xml_file = output_path / "report.xml"

        # Check if the output already exists and contains the necessary file
        if not xml_file.exists():
            print(f"No existing output found for {pdb_file.stem}, running PLIP...")
            run_plip(pdb_file, output_path)
        else:
            print(f"Existing output found for {pdb_file.stem}, skipping PLIP...")

        # Continue to parse and score
        if xml_file.exists():
            interactions = parse_plip_xml(xml_file)
            score = score_interactions(interactions)
            results.append((pdb_file.stem, score, interactions, output_path))
        else:
            print(f"Failed to find XML file for {pdb_file.stem} after processing.")

    # Rank results based on score
    results_df = pd.DataFrame(results, columns=['PDB', 'Score', 'Interactions', 'Output_Path'])
    results_df.sort_values(by='Score', ascending=False, inplace=True)

    top_10 = results_df.head(10)

    top_10_dir = output_dir / "top_10_sessions"
    top_10_dir.mkdir(exist_ok=True)

    for _, row in top_10.iterrows():
        # Find all PSE files in the directory
        pse_files = list(Path(row['Output_Path']).glob('*.pse'))
        if pse_files:
            for pse_file in pse_files:
                dst_pse_file = top_10_dir / pse_file.name
                shutil.copy(pse_file, dst_pse_file)
                print(f"Copied {pse_file} to {dst_pse_file}")
        else:
            print(f"No PyMOL session files found in {row['Output_Path']}.")

    return results_df, top_10

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plip_batch_analysis.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(exist_ok=True)

    ranked_results, top_10 = batch_process_plip(input_dir, output_dir)
    print("Ranked results:")
    print(ranked_results)
    print("Top 10 results:")
    print(top_10)

    # Now only save the DataFrame to CSV, ensuring ranked_results is the DataFrame
    ranked_results.to_csv(output_dir / 'ranked_interactions.csv', index=False)
