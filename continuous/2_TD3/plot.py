import numpy as np
import matplotlib

# Use 'Agg' backend for non-interactive environments (files only)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def plot_scores(filename="td3_scores.npy", window=50):
    if not os.path.exists(filename):
        print(f"Error: Could not find '{filename}'. Run training first.")
        return

    try:
        scores = np.load(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(scores, label="Episode Reward", alpha=0.3, color='gray')

    # Calculate rolling average
    if len(scores) >= window:
        rolling_avg = np.convolve(scores, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(len(rolling_avg)) + window - 1, rolling_avg,
                 label=f"{window}-Episode Moving Average", color='green', linewidth=2)

    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.title("TD3 Training Progress (LunarLanderContinuous-v3)")
    plt.axhline(y=200, color='r', linestyle='--', label="Solved Benchmark (200)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # SAVE INSTEAD OF SHOW
    output_file = filename.replace('.npy', '.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    plot_scores()