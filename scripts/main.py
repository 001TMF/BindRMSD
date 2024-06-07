import yaml
import argparse
import subprocess

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Decide which script to run based on the configuration
    if config['compute_mode'] == 'cpu':
        print("Running CPU mode...")
        subprocess.run(['python', 'process_cpu.py', config_path])
    elif config['compute_mode'] == 'GPU' and config['enable_plots']:
        print("Running GPU mode with graphing...")
        subprocess.run(['python', 'process_gpu_graph.py', config_path])
    elif config['compute_mode'] == 'cuda':  # Placeholder for future GPU processing
        print("Running GPU-cuda mode...")
        subprocess.run(['python', 'process_gpu_cuda.py', config_path])
    elif config['compute_mode'] == 'GPU':
        print("Running GPU mode without graphing...")
        subprocess.run(['python', 'process_gpu.py', config_path])
    else:
        print(f"Unsupported compute mode: {config['compute_mode']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run processing based on configuration.")
    parser.add_argument('config_path', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    main(args.config_path)
