import matplotlib.pyplot as plt

processes = [1, 2, 4, 8, 16, 22]
runtimes = [17794.87, 8975.80, 3710.67, 2300.04, 1929.20, 1955.66]
baseline = runtimes[0]
speedup = [baseline / t for t in runtimes]
efficiency = [s / p for s, p in zip(speedup, processes)]

plt.figure(figsize=(10, 5))
plt.plot(processes, speedup, marker='o', linestyle='-', label='Speedup')
plt.scatter(processes, speedup, color='green')
for x, y in zip(processes, speedup):
    plt.text(x, y + 0.1, f"{y:.2f}", ha='center', fontsize=9)
plt.title("Parallel Processing Speedup", fontsize=14)
plt.xlabel("Number of Processes", fontsize=12)
plt.ylabel("Speedup (T₁ / Tₙ)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(processes)
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(processes, efficiency, marker='o', linestyle='-', color='orange', label='Efficiency')
plt.scatter(processes, efficiency, color='red')
for x, y in zip(processes, efficiency):
    plt.text(x, y + 0.01, f"{y:.2f}", ha='center', fontsize=9)
plt.title("Parallel Processing Efficiency", fontsize=14)
plt.xlabel("Number of Processes", fontsize=12)
plt.ylabel("Efficiency (Speedup / Processes)", fontsize=12)
plt.ylim(0, 1.2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(processes)
plt.tight_layout()
plt.legend()
plt.show()
