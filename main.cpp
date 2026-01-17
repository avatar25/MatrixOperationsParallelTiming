#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace std;

typedef vector<vector<double>> Matrix;

void MatAddition(const Matrix &a, const Matrix &b, Matrix &c, int N,
                 double *totalP, double *totalS);
void MatSubtraction(const Matrix &a, const Matrix &b, Matrix &c, int N,
                    double *totalP, double *totalS);
void MatMultiply(const Matrix &a, const Matrix &b, Matrix &c, int N,
                 double *totalP, double *totalS);
void SumRow(const Matrix &a, const Matrix &b, int N, double *totalP,
            double *totalS);
void SumColumn(const Matrix &a, const Matrix &b, int N, double *totalP,
               double *totalS);
void LUFactorization(const Matrix &a, int N, double *totalP, double *totalS);

int main() {
  int N;
  printf("Enter the size of nxn matrix: ");
  if (scanf("%d", &N) != 1)
    return 1;

  Matrix a(N, vector<double>(N));
  Matrix b(N, vector<double>(N));
  Matrix c(N, vector<double>(N));

  double totalP, totalS;
  omp_set_num_threads(8);

  printf("\nGenerating %d x %d matrices with random values...\n", N, N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i][j] = rand() % 50;
      b[i][j] = rand() % 50;
    }
  }

  printf("\n%-30s %-20s %-20s\n", "Operation", "Parallel (s)", "Serial (s)");
  printf("---------------------------------------------------------------------"
         "-\n");

  MatAddition(a, b, c, N, &totalP, &totalS);
  printf("%-30s %-20.6f %-20.6f\n", "Matrix Addition", totalP, totalS);

  MatSubtraction(a, b, c, N, &totalP, &totalS);
  printf("%-30s %-20.6f %-20.6f\n", "Matrix Subtraction", totalP, totalS);

  MatMultiply(a, b, c, N, &totalP, &totalS);
  printf("%-30s %-20.6f %-20.6f\n", "Matrix Product", totalP, totalS);

  SumRow(a, b, N, &totalP, &totalS);
  printf("%-30s %-20.6f %-20.6f\n", "Sum Row", totalP, totalS);

  SumColumn(a, b, N, &totalP, &totalS);
  printf("%-30s %-20.6f %-20.6f\n", "Sum Column", totalP, totalS);

  LUFactorization(a, N, &totalP, &totalS);
  printf("%-30s %-20.6f %-20.6f\n", "LU Factorization", totalP, totalS);

  return 0;
}

void MatAddition(const Matrix &a, const Matrix &b, Matrix &c, int N,
                 double *totalP, double *totalS) {
  double start, end;
  start = omp_get_wtime();
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      c[i][j] = a[i][j] + b[i][j];
    }
  }
  end = omp_get_wtime();
  *totalP = end - start;

  start = omp_get_wtime();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      c[i][j] = a[i][j] + b[i][j];
    }
  }
  end = omp_get_wtime();
  *totalS = end - start;
}

void MatSubtraction(const Matrix &a, const Matrix &b, Matrix &c, int N,
                    double *totalP, double *totalS) {
  double start, end;
  start = omp_get_wtime();
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      c[i][j] = a[i][j] - b[i][j];
    }
  }
  end = omp_get_wtime();
  *totalP = end - start;

  start = omp_get_wtime();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      c[i][j] = a[i][j] - b[i][j];
    }
  }
  end = omp_get_wtime();
  *totalS = end - start;
}

void MatMultiply(const Matrix &a, const Matrix &b, Matrix &c, int N,
                 double *totalP, double *totalS) {
  double start, end;
  start = omp_get_wtime();
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0;
      for (int k = 0; k < N; k++) {
        sum += a[i][k] * b[k][j];
      }
      c[i][j] = sum;
    }
  }
  end = omp_get_wtime();
  *totalP = end - start;

  start = omp_get_wtime();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0;
      for (int k = 0; k < N; k++) {
        sum += a[i][k] * b[k][j];
      }
      c[i][j] = sum;
    }
  }
  end = omp_get_wtime();
  *totalS = end - start;
}

void SumRow(const Matrix &a, const Matrix &b, int N, double *totalP,
            double *totalS) {
  double start, end;
  double total_sum = 0;
  start = omp_get_wtime();
#pragma omp parallel for reduction(+ : total_sum)
  for (int i = 0; i < N; i++) {
    double row_sum = 0;
    for (int j = 0; j < N; j++) {
      row_sum += a[i][j] + b[i][j];
    }
    total_sum += row_sum;
  }
  end = omp_get_wtime();
  *totalP = end - start;

  total_sum = 0;
  start = omp_get_wtime();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      total_sum += a[i][j] + b[i][j];
    }
  }
  end = omp_get_wtime();
  *totalS = end - start;
}

void SumColumn(const Matrix &a, const Matrix &b, int N, double *totalP,
               double *totalS) {
  double start, end;
  double total_sum = 0;
  start = omp_get_wtime();
#pragma omp parallel for reduction(+ : total_sum)
  for (int j = 0; j < N; j++) {
    double col_sum = 0;
    for (int i = 0; i < N; i++) {
      col_sum += a[i][j] + b[i][j];
    }
    total_sum += col_sum;
  }
  end = omp_get_wtime();
  *totalP = end - start;

  total_sum = 0;
  start = omp_get_wtime();
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      total_sum += a[i][j] + b[i][j];
    }
  }
  end = omp_get_wtime();
  *totalS = end - start;
}

void LUFactorization(const Matrix &a, int N, double *totalP, double *totalS) {
  Matrix lower(N, vector<double>(N, 0));
  Matrix upper(N, vector<double>(N, 0));

  double start, end;
  start = omp_get_wtime();
  // LU decomposition is hard to parallelize simply without blocking,
  // but we can parallelize the inner loops.
  for (int i = 0; i < N; i++) {
// Upper Triangular
#pragma omp parallel for
    for (int k = i; k < N; k++) {
      double sum = 0;
      for (int j = 0; j < i; j++)
        sum += (lower[i][j] * upper[j][k]);
      upper[i][k] = a[i][k] - sum;
    }

// Lower Triangular
#pragma omp parallel for
    for (int k = i; k < N; k++) {
      if (i == k)
        lower[i][i] = 1;
      else {
        double sum = 0;
        for (int j = 0; j < i; j++)
          sum += (lower[k][j] * upper[j][i]);
        if (upper[i][i] != 0)
          lower[k][i] = (a[k][i] - sum) / upper[i][i];
      }
    }
  }
  end = omp_get_wtime();
  *totalP = end - start;

  // Serial
  Matrix l2(N, vector<double>(N, 0));
  Matrix u2(N, vector<double>(N, 0));
  start = omp_get_wtime();
  for (int i = 0; i < N; i++) {
    for (int k = i; k < N; k++) {
      double sum = 0;
      for (int j = 0; j < i; j++)
        sum += (l2[i][j] * u2[j][k]);
      u2[i][k] = a[i][k] - sum;
    }
    for (int k = i; k < N; k++) {
      if (i == k)
        l2[i][i] = 1;
      else {
        double sum = 0;
        for (int j = 0; j < i; j++)
          sum += (l2[k][j] * u2[j][i]);
        if (u2[i][i] != 0)
          l2[k][i] = (a[k][i] - sum) / u2[i][i];
      }
    }
  }
  end = omp_get_wtime();
  *totalS = end - start;
}
