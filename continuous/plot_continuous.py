import numpy as np
import matplotlib.pyplot as plt
import os


def plot_comparison(window=50):
    # ----------------------------
    # Configuration: Update paths here!
    # ----------------------------
    file_paths = {
        "DDPG (Baseline)": "1_DDPG/ddpg_scores.npy",
        "TD3": "2_TD3/td3_scores.npy",
        "SAC": "3_SAC/sac_scores.npy",
        "TQC (Distributional)": "4_TQC/tqc_scores.npy",
    }

    # Colors for each line to ensure high contrast
    colors = {
        "DDPG (Baseline)": "gray",  # Often baselines are gray/black
        "TD3": "orange",  # Distinct warm color
        "SAC": "purple",  # Distinct cool color
        "TQC (Distributional)": "cyan",  # Bright, stands out as "best"
    }

    plt.figure(figsize=(12, 8))

    # Plot loop
    for label, path in file_paths.items():
        if not os.path.exists(path):
            print(f"Warning: File not found for {label}: {path}. Skipping.")
            continue

        try:
            scores = np.load(path)

            # Handle potential differences in training length
            # We plot whatever data is available

            # Calculate rolling average
            if len(scores) >= window:
                rolling_avg = np.convolve(scores, np.ones(window) / window, mode='valid')

                # Adjust x-axis to match the convolution shift
                x_axis = np.arange(len(rolling_avg)) + window - 1

                # Plot the rolling average line
                plt.plot(x_axis, rolling_avg, label=label, color=colors.get(label, 'black'), linewidth=2.5)

                # Optional: Plot faint raw data for context (can get messy with 4 algos)
                # plt.plot(scores, color=colors.get(label, 'black'), alpha=0.1)
            else:
                print(f"Warning: Not enough data points in {label} to calculate rolling average.")

        except Exception as e:
            print(f"Error loading {label}: {e}")

    # Aesthetics
    plt.ylabel("Average Reward (Moving Avg 50)", fontsize=12)
    plt.xlabel("Episode", fontsize=12)
    plt.title("Continuous Control Algorithms Comparison (LunarLanderContinuous-v3)", fontsize=14)

    # Add benchmark lines
    plt.axhline(y=200, color='green', linestyle='--', alpha=0.7, label="Solved (200)")
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)  # Zero line for reference

    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    output_file = "continuous_comparison_plot.png"
    plt.savefig(output_file, dpi=300)
    print(f"Comparison plot saved to {output_file}")
    # plt.show() # Uncomment if you want to see it interactively


if __name__ == "__main__":
    plot_comparison()