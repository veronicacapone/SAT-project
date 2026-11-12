import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load serialized data
with open('Array_1.pkl', 'rb') as file:
    result_arrays = pickle.load(file)
with open('Array_2.pkl', 'rb') as file:
    log_arrays = pickle.load(file)

# Process results
result_n200 = result_arrays[0]
result_n200_median = np.median(result_n200, axis=1)

result_n300 = result_arrays[1]
result_n300_median = np.median(result_n300, axis=1)

result_n400 = result_arrays[2]
result_n400_median = np.median(result_n400, axis=1)

result_n500 = result_arrays[3]
result_n500_median = np.median(result_n500, axis=1)

result_n600 = result_arrays[4]
result_n600_median = np.median(result_n600, axis=1)

# Log arrays
log_n200 = log_arrays[0]
log_n300 = log_arrays[1]
log_n400 = log_arrays[2]
log_n500 = log_arrays[3]
log_n600 = log_arrays[4]

# Compute median-aligned values
median_aligned_200 = np.interp(0.5, result_n200_median[::-1], np.arange(400, 400 + 50 * len(result_n200), 50)[::-1])
median_aligned_300 = np.interp(0.5, result_n300_median[::-1], np.arange(950, 950 + 50 * len(result_n300), 50)[::-1])
median_aligned_400 = np.interp(0.5, result_n400_median[::-1], np.arange(1350, 1350 + 50 * len(result_n400), 50)[::-1])
median_aligned_500 = np.interp(0.5, result_n500_median[::-1], np.arange(1750, 1750 + 50 * len(result_n500), 50)[::-1])
median_aligned_600 = np.interp(0.5, result_n600_median[::-1], np.arange(2150, 2150 + 50 * len(result_n600), 50)[::-1])

# Combine results
median_aligned_values = np.array([
    median_aligned_200,
    median_aligned_300,
    median_aligned_400,
    median_aligned_500,
    median_aligned_600
])

n_values = np.arange(200, 700, 100)

# Print results
print(median_aligned_values / n_values)
print(median_aligned_values)

# Uncomment the plotting code below for visualization
# plt.clf()
# plt.figure(figsize=(12, 6))
# plt.xticks(ticks=np.arange(400, 2150 + 50 * len(result_n600) + 50, 100))

# plt.plot(np.arange(400, 400 + 50 * len(result_n200), 50), result_n200_median, marker="o", color="blue", label="N = 200")
# plt.plot(np.arange(950, 950 + 50 * len(result_n300), 50), result_n300_median, marker="o", color="red", label="N = 300")
# plt.plot(np.arange(1350, 1350 + 50 * len(result_n400), 50), result_n400_median, marker="o", color="cyan", label="N = 400")
# plt.plot(np.arange(1750, 1750 + 50 * len(result_n500), 50), result_n500_median, marker="o", color="orange", label="N = 500")
# plt.plot(np.arange(2150, 2150 + 50 * len(result_n600), 50), result_n600_median, marker="o", color="purple", label="N = 600")
# plt.xlabel("Values of M (Clauses)")
# plt.ylabel("Empirical probability of solving an instance")
# plt.axhline(y=0.5, color='red', linestyle='dashdot', linewidth=1)
# plt.legend()
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(400, 400 + 50 * len(result_n200), 50) / 200, result_n200_median, marker="o", color="blue", label="N = 200")
# plt.plot(np.arange(950, 950 + 50 * len(result_n300), 50) / 300, result_n300_median, marker="o", color="red", label="N = 300")
# plt.plot(np.arange(1350, 1350 + 50 * len(result_n400), 50) / 400, result_n400_median, marker="o", color="cyan", label="N = 400")
# plt.plot(np.arange(1750, 1750 + 50 * len(result_n500), 50) / 500, result_n500_median, marker="o", color="orange", label="N = 500")
# plt.plot(np.arange(2150, 2150 + 50 * len(result_n600), 50) / 600, result_n600_median, marker="o", color="purple", label="N = 600")
# plt.xlabel("Values of M/N (Clauses)")
# plt.ylabel("Empirical probability of solving an instance")
# plt.axhline(y=0.5, color='red', linestyle='dashdot', linewidth=1)
# plt.legend()
# plt.show()
