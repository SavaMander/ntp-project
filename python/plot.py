import numpy as np
import matplotlib.pyplot as plt

# --- Example data: wall-clock times from your experiments ---
# Replace these with your measured average times

# Strong scaling (Python)
threads = np.arange(1, 13)  # 1 to 12 threads
python_strong_times = np.array([
    1.2037, 2.7801, 2.6449, 2.6006, 2.7240, 2.7961, 2.9682, 3.0856, 3.2517, 3.4569, 3.6570, 3.7172
])
rust_strong_times = np.array([
    1.0908, 0.6628, 0.5238, 0.4671, 0.4338, 0.4330, 0.4215, 0.4012, 0.3791, 0.3652, 0.3538, 0.3452
])

# Weak scaling (Python)
python_weak_times = np.array([
    0.1220, 1.9647, 2.1016, 2.2464, 2.4114, 2.5791, 2.8554, 3.0761, 3.1409, 3.2780, 3.5748, 3.8160
])
rust_weak_times = np.array([
    1.0785, 2.1496, 2.3600, 2.4375, 3.1690, 3.8607, 4.2393, 4.1222, 4.4159, 4.9980, 5.0791, 5.6582
])

# --- Compute speedups ---
python_strong_speedup = python_strong_times[0] / python_strong_times
rust_strong_speedup = rust_strong_times[0] / rust_strong_times

python_weak_speedup = threads * python_weak_times[0] / python_weak_times
rust_weak_speedup = threads * rust_weak_times[0] / rust_weak_times

# --- Theoretical speedups (Amdahl) ---
def amdahl_speedup(p, f_s):
    """
    p: number of threads
    f_s: fraction of serial code
    """
    return 1 / (f_s + (1 - f_s) / p)

# Assume serial fraction based on first measurements
f_s_python = 0.0101  # adjust if needed
f_s_rust = 0.0475

amdahl_python = amdahl_speedup(threads, f_s_python)
amdahl_rust = amdahl_speedup(threads, f_s_rust)

# --- Theoretical speedups (Gustafson) ---
def gustafson_speedup(p, f_s):
    """
    p: number of threads
    f_s: fraction of serial code
    """
    return f_s + (1 - f_s) * p

gustafson_python = gustafson_speedup(threads, f_s_python)
gustafson_rust = gustafson_speedup(threads, f_s_rust)

# --- Plotting ---
plt.figure(figsize=(12, 10))

# 1. Strong scaling Python (Amdahl)
plt.subplot(2, 2, 1)
plt.plot(threads, python_strong_speedup, 'o-', label='Measured Python')
plt.plot(threads, amdahl_python, 'r--', label='Amdahl prediction')
plt.plot(threads, threads, 'g:', label='Ideal linear scaling')
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("Strong Scaling Python (Amdahl's Law)")
plt.legend()
plt.grid(True)

# 2. Strong scaling Rust (Amdahl)
plt.subplot(2, 2, 2)
plt.plot(threads, rust_strong_speedup, 'o-', label='Measured Rust')
plt.plot(threads, amdahl_rust, 'r--', label='Amdahl prediction')
plt.plot(threads, threads, 'g:', label='Ideal linear scaling')
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("Strong Scaling Rust (Amdahl's Law)")
plt.legend()
plt.grid(True)

# 3. Weak scaling Python (Gustafson)
plt.subplot(2, 2, 3)
plt.plot(threads, python_weak_speedup, 'o-', label='Measured Python')
plt.plot(threads, gustafson_python, 'r--', label='Gustafson prediction')
plt.plot(threads, threads, 'g:', label='Ideal linear scaling')
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("Weak Scaling Python (Gustafson's Law)")
plt.legend()
plt.grid(True)

# 4. Weak scaling Rust (Gustafson)
plt.subplot(2, 2, 4)
plt.plot(threads, rust_weak_speedup, 'o-', label='Measured Rust')
plt.plot(threads, gustafson_rust, 'r--', label='Gustafson prediction')
plt.plot(threads, threads, 'g:', label='Ideal linear scaling')
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("Weak Scaling Rust (Gustafson's Law)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
