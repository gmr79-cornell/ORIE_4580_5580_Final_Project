import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_results(df: pd.DataFrame) -> None:
    """
    Block 3 – Visualization:
    1. Histograms of TTFT and total completion latency.
    2. Throughput over time using 2-second windows.
    """
    if df.empty:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Latency Distribution (TTFT vs Total Latency)
    ttft_vals = df["TTFT"].dropna()
    total_lat = df["completion_latency"].dropna()
    axes[0].hist(ttft_vals, bins=20, alpha=0.7, label="Time To First Token (TTFT)")
    axes[0].hist(total_lat, bins=20, alpha=0.5, label="Total Completion Latency")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Latencies")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Throughput (requests completed per 2-second window)
    max_time = df["completion_time"].max()
    time_bins = np.arange(0, max_time + 2, 2)  # 2-second windows
    throughput_counts, _ = np.histogram(df["completion_time"], bins=time_bins)

    throughput_rps = throughput_counts / 2.0  # requests per second
    axes[1].plot(time_bins[:-1], throughput_rps, marker="o", linestyle="-")
    axes[1].set_xlabel("Simulation Time (s)")
    axes[1].set_ylabel("Throughput (req/s)")
    axes[1].set_title("System Throughput Over Time")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary metrics
    print("\n--- Summary Metrics ---")
    print(f"Mean TTFT: {ttft_vals.mean():.4f} s")
    print(f"Mean total latency: {total_lat.mean():.4f} s")
    print(f"P95 total latency: {total_lat.quantile(0.95):.4f} s")
    print(f"Total throughput: {len(df) / max_time:.2f} req/s")
    if "avg_TBT" in df.columns:
        avg_tbt_vals = df["avg_TBT"].dropna()
        if not avg_tbt_vals.empty:
            print(f"Mean avg_TBT: {avg_tbt_vals.mean():.4f} s")


def plot_demo_metrics(df: pd.DataFrame) -> None:
    """
    Block 4 – Demo visualization:
    - Histograms of TTFT and total Latency.
    - Throughput using 5-second windows.
    """
    if df.empty:
        print("No data to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax1.hist(df["TTFT"].dropna(), bins=15, alpha=0.7, label="TTFT")
    # Use either 'Latency' or 'completion_latency' if available
    lat_col = "Latency" if "Latency" in df.columns else "completion_latency"
    ax1.hist(df[lat_col].dropna(), bins=15, alpha=0.5, label="Total Latency")
    ax1.set_title("Latency Distribution")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Count")
    ax1.legend()

    # Throughput over 5-second windows
    max_t = df["completion_time"].max()
    bins = np.arange(0, max_t + 5, 5)
    counts, _ = np.histogram(df["completion_time"], bins=bins)
    ax2.plot(bins[:-1], counts / 5.0, marker="o")
    ax2.set_title("Throughput (req/sec)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Throughput (req/s)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Console metrics (optional)
    ttft_vals = df["TTFT"].dropna()
    lat_vals = df[lat_col].dropna()
    print("\n--- Summary Metrics ---")
    print(f"Mean TTFT: {ttft_vals.mean():.4f} s")
    print(f"Mean total latency: {lat_vals.mean():.4f} s")
    print(f"P95 total latency: {lat_vals.quantile(0.95):.4f} s")
    print(f"Total throughput: {len(df) / max_t:.2f} req/s")
    if "avg_TBT" in df.columns:
        avg_tbt_vals = df["avg_TBT"].dropna()
        if not avg_tbt_vals.empty:
            print(f"Mean avg_TBT: {avg_tbt_vals.mean():.4f} s")
