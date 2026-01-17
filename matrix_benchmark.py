import numpy as np
import time
import sys

def matrix_addition_serial(a, b):
    N = len(a)
    c = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            c[i][j] = a[i][j] + b[i][j]
    return c

def matrix_subtraction_serial(a, b):
    N = len(a)
    c = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            c[i][j] = a[i][j] - b[i][j]
    return c

def matrix_multiply_serial(a, b):
    N = len(a)
    c = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            s = 0
            for k in range(N):
                s += a[i][k] * b[k][j]
            c[i][j] = s
    return c

def sum_all_serial(a, b):
    N = len(a)
    s = 0
    for i in range(N):
        for j in range(N):
            s += a[i][j] + b[i][j]
    return s

def lu_factorization_serial(a):
    N = len(a)
    lower = [[0.0 for _ in range(N)] for _ in range(N)]
    upper = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for k in range(i, N):
            s = 0
            for j in range(i):
                s += lower[i][j] * upper[j][k]
            upper[i][k] = a[i][k] - s
        for k in range(i, N):
            if i == k:
                lower[i][i] = 1.0
            else:
                s = 0
                for j in range(i):
                    s += lower[k][j] * upper[j][i]
                if upper[i][i] != 0:
                    lower[k][i] = (a[k][i] - s) / upper[i][i]
    return lower, upper

def run_benchmark():
    try:
        N = int(input("Enter the size of nxn matrix: "))
    except EOFError:
        N = 100
        print(f"Using default N={N}")
    except ValueError:
        print("Invalid input, using 100")
        N = 100

    print(f"\nGenerating {N} x {N} matrices...")
    a_list = [[float(np.random.randint(0, 50)) for _ in range(N)] for _ in range(N)]
    b_list = [[float(np.random.randint(0, 50)) for _ in range(N)] for _ in range(N)]
    
    a_np = np.array(a_list)
    b_np = np.array(b_list)

    print(f"\n{'Operation':<30} {'NumPy (s)':<20} {'Serial Loops (s)':<20}")
    print("-" * 70)

    # Addition
    start = time.perf_counter()
    _ = a_np + b_np
    t_np = time.perf_counter() - start
    
    start = time.perf_counter()
    _ = matrix_addition_serial(a_list, b_list)
    t_serial = time.perf_counter() - start
    print(f"{'Matrix Addition':<30} {t_np:<20.6f} {t_serial:<20.6f}")

    # Subtraction
    start = time.perf_counter()
    _ = a_np - b_np
    t_np = time.perf_counter() - start
    
    start = time.perf_counter()
    _ = matrix_subtraction_serial(a_list, b_list)
    t_serial = time.perf_counter() - start
    print(f"{'Matrix Subtraction':<30} {t_np:<20.6f} {t_serial:<20.6f}")

    # Multiplication
    start = time.perf_counter()
    _ = np.dot(a_np, b_np)
    t_np = time.perf_counter() - start
    
    start = time.perf_counter()
    if N > 250:
        print(f"{'Matrix Product':<30} {t_np:<20.6f} {'Skipped (>250)':<20}")
    else:
        _ = matrix_multiply_serial(a_list, b_list)
        t_serial = time.perf_counter() - start
        print(f"{'Matrix Product':<30} {t_np:<20.6f} {t_serial:<20.6f}")

    # Row/Col Sum (equivalent to sum all for comparison)
    start = time.perf_counter()
    _ = np.sum(a_np) + np.sum(b_np)
    t_np = time.perf_counter() - start
    
    start = time.perf_counter()
    _ = sum_all_serial(a_list, b_list)
    t_serial = time.perf_counter() - start
    print(f"{'Sum Elements':<30} {t_np:<20.6f} {t_serial:<20.6f}")

    # LU Factorization
    # NumPy doesn't have a direct LU, it has linalg.solve or scipy.linalg.lu
    try:
        import scipy.linalg
        start = time.perf_counter()
        _ = scipy.linalg.lu(a_np)
        t_np = time.perf_counter() - start
    except ImportError:
        t_np = 0.0 # Placeholder if scipy not installed

    start = time.perf_counter()
    if N > 200:
        print(f"{'LU Factorization':<30} {t_np:<20.6f} {'Skipped (>200)':<20}")
    else:
        _ = lu_factorization_serial(a_list)
        t_serial = time.perf_counter() - start
        print(f"{'LU Factorization':<30} {t_np:<20.6f} {t_serial:<20.6f}")

if __name__ == "__main__":
    run_benchmark()
