import numpy as np
import matplotlib.pyplot as plt
import os


def plot_comparison(window=50):
    # ----------------------------
    # Configuration: Update paths here!
    # ----------------------------
    file_paths = {
        "DQN": "1_DQN/lunar_dqn_scores.npy",
        "Double DQN": "2_Double_DQN/double_dqn_scores.npy",
        "Dueling DQN": "3_Dueling_DQN/dueling_dqn_lunar_scores.npy",  # Or whatever your Dueling file is named
        "PER Dueling Double DQN": "4_PER/per_dqn_scores.npy"
    }

    # Colors for each line
    colors = {
        "DQN": "blue",
        "Double DQN": "orange",
        "Dueling DQN": "red",
        "PER Dueling Double DQN": "green"
    }

    plt.figure(figsize=(12, 8))

    # Plot loop
    for label, path in file_paths.items():
        if not os.path.exists(path):
            print(f"Warning: File not found for {label}: {path}. Skipping.")
            continue

        try:
            scores = np.load(path)
            # Calculate rolling average
            if len(scores) >= window:
                rolling_avg = np.convolve(scores, np.ones(window) / window, mode='valid')
                # Adjust x-axis to match the convolution shift
                x_axis = np.arange(len(rolling_avg)) + window - 1
                plt.plot(x_axis, rolling_avg, label=label, color=colors.get(label, 'black'), linewidth=2)
            else:
                print(f"Warning: Not enough data points in {label} to calculate rolling average.")

        except Exception as e:
            print(f"Error loading {label}: {e}")

    # Aesthetics
    plt.ylabel("Average Reward (Moving Avg 50)", fontsize=12)
    plt.xlabel("Episode", fontsize=12)
    plt.title("Discrete Control Algorithms Comparison (LunarLander-v3)", fontsize=14)
    plt.axhline(y=200, color='black', linestyle='--', alpha=0.5, label="Solved Benchmark (200)")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    output_file = "discrete_comparison_plot.png"
    plt.savefig(output_file, dpi=300)
    print(f"Comparison plot saved to {output_file}")
    # plt.show() # Uncomment if you want to see it interactively


if __name__ == "__main__":
    plot_comparison()