import subprocess
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import os

# Configuration
CPP_EXECUTABLE = "./build/MatrixOperationsParallel"
SIZES = [100, 200, 400, 600, 800, 1000]
OPERATIONS_TO_TRACK = ["Matrix Addition", "Matrix Product", "LU Factorization"]

def run_cpp_benchmark(n):
    print(f"  Running C++ benchmark for N={n}...")
    try:
        # Pass N to stdin
        process = subprocess.Popen([CPP_EXECUTABLE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(input=str(n) + "\n")
        
        results = {}
        # Parse the table: "Matrix Addition                0.002900             0.006371"
        for line in stdout.splitlines():
            for op in OPERATIONS_TO_TRACK:
                if op in line:
                    parts = re.split(r'\s{2,}', line.strip())
                    if len(parts) >= 3:
                        results[f"{op}_Parallel"] = float(parts[1])
                        results[f"{op}_Serial"] = float(parts[2])
        return results
    except Exception as e:
        print(f"Error running C++: {e}")
        return None

def run_python_benchmark(n):
    print(f"  Running Python benchmark for N={n}...")
    results = {}
    
    # Generate matrices
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    
    # 1. Addition
    # NumPy
    start = time.perf_counter()
    _ = a + b
    results["Matrix Addition_NumPy"] = time.perf_counter() - start
    
    # Python Loops (only for small N because it's too slow)
    if n <= 400:
        a_list = a.tolist()
        b_list = b.tolist()
        start = time.perf_counter()
        c = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                c[i][j] = a_list[i][j] + b_list[i][j]
        results["Matrix Addition_Loops"] = time.perf_counter() - start
    else:
        results["Matrix Addition_Loops"] = None

    # 2. Multiplication
    # NumPy
    start = time.perf_counter()
    _ = np.dot(a, b)
    results["Matrix Product_NumPy"] = time.perf_counter() - start
    
    # Python Loops
    if n <= 200:
        a_list = a.tolist()
        b_list = b.tolist()
        start = time.perf_counter()
        c = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0
                for k in range(n):
                    s += a_list[i][k] * b_list[k][j]
                c[i][j] = s
        results["Matrix Product_Loops"] = time.perf_counter() - start
    else:
        results["Matrix Product_Loops"] = None

    # 3. LU Factorization
    import scipy.linalg
    start = time.perf_counter()
    _ = scipy.linalg.lu(a)
    results["LU Factorization_NumPy"] = time.perf_counter() - start
    
    return results

def main():
    if not os.path.exists(CPP_EXECUTABLE):
        print(f"Error: {CPP_EXECUTABLE} not found. Please build the C++ project first.")
        return

    data = {
        "Size": SIZES,
        "C++ Parallel": {op: [] for op in OPERATIONS_TO_TRACK},
        "C++ Serial": {op: [] for op in OPERATIONS_TO_TRACK},
        "Python NumPy": {op: [] for op in OPERATIONS_TO_TRACK},
        "Python Loops": {op: [] for op in OPERATIONS_TO_TRACK}
    }

    print("🚀 Starting Multi-Language Matrix Performance Benchmark...")
    
    for n in SIZES:
        cpp_res = run_cpp_benchmark(n)
        py_res = run_python_benchmark(n)
        
        for op in OPERATIONS_TO_TRACK:
            if cpp_res:
                data["C++ Parallel"][op].append(cpp_res.get(f"{op}_Parallel", 0))
                data["C++ Serial"][op].append(cpp_res.get(f"{op}_Serial", 0))
            if py_res:
                data["Python NumPy"][op].append(py_res.get(f"{op}_NumPy", 0))
                data["Python Loops"][op].append(py_res.get(f"{op}_Loops"))

    print("\n📊 Benchmarking complete! Generating graphs...")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Matrix Performance Comparison: C++ vs Python (NumPy/Loops)", fontsize=16)

    colors = {
        "C++ Parallel": "#2ecc71", # Green
        "C++ Serial": "#27ae60",   # Dark Green
        "Python NumPy": "#3498db", # Blue
        "Python Loops": "#e74c3c"  # Red
    }

    for i, op in enumerate(OPERATIONS_TO_TRACK):
        ax = axes[i]
        ax.plot(SIZES, data["C++ Parallel"][op], marker='o', label="C++ Parallel (OpenMP)", color=colors["C++ Parallel"], linewidth=2)
        ax.plot(SIZES, data["C++ Serial"][op], marker='s', label="C++ Serial", color=colors["C++ Serial"], linestyle='--')
        ax.plot(SIZES, data["Python NumPy"][op], marker='^', label="Python NumPy", color=colors["Python NumPy"], linewidth=2)
        
        # Filter out None for Loops
        loop_sizes = [s for s, v in zip(SIZES, data["Python Loops"][op]) if v is not None]
        loop_values = [v for v in data["Python Loops"][op] if v is not None]
        if loop_values:
            ax.plot(loop_sizes, loop_values, marker='x', label="Python Loops", color=colors["Python Loops"], linestyle=':')

        ax.set_title(f"{op}")
        ax.set_xlabel("Matrix Size (N x N)")
        ax.set_ylabel("Time (seconds)")
        ax.set_yscale('log')  # Use log scale because Python Loops are orders of magnitude slower
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png = "performance_comparison.png"
    plt.savefig(output_png, dpi=300)
    print(f"\n✅ Done! Comparison graph saved as '{output_png}'")
    
    # Summary of Matrix Product
    if data["C++ Parallel"]["Matrix Product"] and data["Python NumPy"]["Matrix Product"]:
        last_idx = -1
        cpp_p = data["C++ Parallel"]["Matrix Product"][last_idx]
        py_np = data["Python NumPy"]["Matrix Product"][last_idx]
        
        if py_np > 0:
            speed_ratio = cpp_p / py_np
            if speed_ratio > 1:
                print(f"\nQuick Comparison (N={SIZES[last_idx]} Matrix Multiplication):")
                print(f" - C++ Parallel: {cpp_p:.4f}s")
                print(f" - Python NumPy: {py_np:.4f}s")
                print(f" 💡 NumPy is ~{speed_ratio:.1f}x faster than our C++ implementation!")
                print("    (Reason: NumPy uses highly optimized BLAS/MKL libraries and SIMD instructions)")
            else:
                print(f"\nQuick Comparison (N={SIZES[last_idx]} Matrix Multiplication):")
                print(f" - C++ Parallel: {cpp_p:.4f}s")
                print(f" - Python NumPy: {py_np:.4f}s")
                print(f" 💡 C++ Parallel is ~{1/speed_ratio:.1f}x faster than NumPy!")

if __name__ == "__main__":
    main()
