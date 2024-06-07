# BindRMSD

BindRMSD is a tool designed to streamline the analysis of protein-ligand interactions by reducing the number of docking output files through RMSD clustering. This project aims to assist researchers and scientists in the field of computational chemistry and bioinformatics by providing an efficient way to filter and identify key interactions from large datasets of docking results.

## Features

- **RMSD Calculation**: Compute the root-mean-square deviation (RMSD) for pairs of structures.
- **Clustering**: Cluster similar structures based on their RMSD values.
- **Output Reduction**: Reduce the number of output files by selecting representative structures from each cluster.
- **Visualization**: Generate dendrograms and scatter plots to visualize the clustering results.

## Visualization
![Dendrogram](https://github.com/001TMF/BindRMSD/blob/877637e086e35d39c2c57d33524d7ed315ff8176/outputs/dendrogram.png?raw=true)
![Scatter-Plot](https://github.com/001TMF/BindRMSD/blob/877637e086e35d39c2c57d33524d7ed315ff8176/outputs/scatter_plot.png)

## Prerequisites

Before you can use BindRMSD, you need to have the following installed on your machine:
- Python (3.7 or newer)
- Conda (Anaconda or Miniconda)




### Setting Up the Conda Environment

To create a Conda environment with all the necessary dependencies, follow these steps:

1. Clone the BindRMSD repository to your local machine:
   ```bash
   git clone https://github.com/001TMF/BindRMSD.git
   cd BindRMSD
   ```
2. Create the Conda environment from the environment.yml file:
   ```bash
   conda env create -f environment.yaml
   ```
3. Activate the environment:
   ```bash
   conda activate BindRMSD
   ```

### Installing Dependencies with Pip
If you prefer to use pip instead of Conda, ensure you have a virtual environment manager like venv:

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv bindrmsd-env
   source bindrmsd-env/bin/activate  # On Windows use `bindrmsd-env\Scripts\activate`
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage
To run BindRMSD, navigate to the BindRMSD scripts directory and execute the main script:

```bash
python main.py cluster-config.yaml
```
Ensure your config.yml file is set up correctly with paths to your input data and desired parameters.

## Contributing

Contributions to BindRMSD are welcome and appreciated. To contribute:

   1. Fork the repository.
   2. Create a new branch (git checkout -b feature-branch). 
   3. Make your changes and commit them (git commit -am 'Add some feature').
   4. Push to the branch (git push origin feature-branch).
   5. Create a new Pull Request.


## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) MIT License - see the LICENSE file for details.

