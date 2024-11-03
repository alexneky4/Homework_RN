import json
import os
from pipeline import TrainingPipeline


def run_experiment(config_path):
    pipeline = TrainingPipeline(config_path)
    best_train_acc, best_val_acc = pipeline.train()

    output_file_path = "experiment_results.txt"

    with open(output_file_path, "a") as file:
        file.write("Training Results:\n")
        file.write(f"Best Training Accuracy: {best_train_acc:.4f}\n")
        file.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")


def main():
    with open("experiment_configs.json") as f:
        config_data = json.load(f)

    for config_path in config_data['configs']:
        if os.path.exists(config_path):
            print(f"Running experiment with config: {config_path}")
            run_experiment(config_path)
        else:
            print(f"Config path does not exist: {config_path}")


if __name__ == "__main__":
    main()
