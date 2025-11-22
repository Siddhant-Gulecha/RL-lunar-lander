import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os


def plot_scores(filename="tqc_scores.npy", window=50):
    if not os.path.exists(filename):
        print(f"Error: Could not find '{filename}'. Run training first.")
        return

    scores = np.load(filename)
    if len(scores) == 0:
        print("No scores found in file. Run training for longer.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(scores, label="Episode Reward", alpha=0.3, color='cyan')

    if len(scores) >= window:
        rolling_avg = np.convolve(scores, np.ones(window) / window, mode='valid')
        x_axis = np.arange(len(rolling_avg)) + window - 1
        plt.plot(x_axis, rolling_avg,
                 label=f"{window}-Episode Moving Average", color='blue', linewidth=2)

    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.title("TQC Training Progress (LunarLanderContinuous-v3)")
    plt.axhline(y=200, color='r', linestyle='--', label="Solved Benchmark (200)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = filename.replace('.npy', '.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    plot_scores("tqc_scores.npy")