import matplotlib.pyplot as plt

processes = [1, 2, 4, 8, 16, 22]
runtimes = [17794.87, 8975.80, 3710.67, 2300.04, 1929.20, 1955.66]
plt.figure(figsize=(10, 6))
plt.plot(
    processes, runtimes, marker="o", linestyle="-", linewidth=2, label="Runtime (s)"
)
plt.scatter(processes, runtimes, color="red", zorder=5)
for x, y in zip(processes, runtimes):
    plt.text(x, y + 200, f"{y:.2f}s", ha="center", fontsize=9)

plt.title("Parallel Processing Benchmark Results", fontsize=14)
plt.xlabel("Number of Processes", fontsize=12)
plt.ylabel("Runtime (seconds)", fontsize=12)
plt.xticks(processes)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
