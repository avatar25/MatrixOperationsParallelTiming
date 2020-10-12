#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#define MAX_SIZE 100
using namespace std;

void MatAddition(int a[][MAX_SIZE], int b[][MAX_SIZE], int c[][MAX_SIZE], int N, double *totalP, double *totalS);
void MatSubtraction(int a[][MAX_SIZE], int b[][MAX_SIZE], int c[][MAX_SIZE], int N, double *totalP, double *totalS);
void MatMultiply(int a[][MAX_SIZE],int b[][MAX_SIZE],int c[][MAX_SIZE], int N, double *totalP, double *totalS);
void SumRow(int a[][MAX_SIZE], int b[][MAX_SIZE], int c[][MAX_SIZE], int N, double* totalP, double* totalS);
void SumColumn(int a[][MAX_SIZE], int b[][MAX_SIZE], int c[][MAX_SIZE], int N, double* totalP, double* totalS);
void LUFactorization(int a[][MAX_SIZE], int b[][MAX_SIZE], int c[][MAX_SIZE], int N, double* totalP, double* totalS);

int main()
{
    int N;
    int a[MAX_SIZE][MAX_SIZE];
    int b[MAX_SIZE][MAX_SIZE];
    int c[MAX_SIZE][MAX_SIZE];
    double totalP, totalS; //total time of execution in parallel and sequence
    double start, end;
    printf("Enter the size of nxn matrix\n");
    scanf("%d", &N);
    omp_set_num_threads(8);
    printf("\nThe values of %d x %d matrix are generated randomly less than 50\n", N, N);
    printf("\nFor thread number 1: \n");
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            a[i][j] = rand() % 50;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            b[i][j] = rand() % 50;

    printf("\nTime of execution:\n");
    printf("parallel\t\t\t\t\tserial\n");
#pragma omp parallel sections
    {
#pragma omp section
        {
            MatAddition(a, b, c, N, &totalP, &totalS);

            printf("%f\t<<for Matrix addition>> \t%f\n", totalP, totalS);

        }
#pragma omp section
        {
            MatSubtraction(a, b, c, N, &totalP, &totalS);

            printf("%f\t<<for matrix subtraction>> \t%f\n", totalP, totalS);
        }
#pragma omp section
        {
            MatMultiply(a, b, c, N, &totalP, &totalS);

            printf("%f\t<<for matrix product>> \t\t%f\n", totalP, totalS);
        }
#pragma omp section
        {
            SumRow(a, b, c, N, &totalP, &totalS);

            printf("%f\t<<for sum row>> \t\t%f\n", totalP, totalS);
        }
#pragma omp section
        {
            SumColumn(a, b, c, N, &totalP, &totalS);

            printf("%f\t<<for sum column>> \t\t%f\n", totalP, totalS);
        }
#pragma omp section
        {
            LUFactorization(a, b, c, N, &totalP, &totalS);

            printf("%f\t<<for LU factorization>> \t%f\n", totalP, totalS);
        }
    }
}

void MatAddition(int a[][MAX_SIZE],int b[][MAX_SIZE],int c[][MAX_SIZE],int N, double *totalP, double *totalS)
{
    double start, end;
    start = omp_get_wtime();
#pragma omp parallel for schedule(static, 8)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    end = omp_get_wtime();
    *totalP = end - start;
    //now without parallel
    start = 0.0;
    end = 0.0;
    start = omp_get_wtime();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    end = omp_get_wtime();
    *totalS = end - start;
}

void MatSubtraction(int a[][MAX_SIZE], int b[][MAX_SIZE], int c[][MAX_SIZE], int N, double* totalP, double* totalS)
{
    double start, end;
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    end = omp_get_wtime();
    *totalP = end - start;
    start = 0.0;
    end = 0.0;
    start = omp_get_wtime();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    end = omp_get_wtime();
    *totalS = end - start;
}

void MatMultiply(int a[][MAX_SIZE], int b[][MAX_SIZE], int c[][MAX_SIZE], int N, double* totalP, double* totalS)
{
    int i, j, k;
    double start, end;
    start = omp_get_wtime();
#pragma omp parallel shared(a,b,c) private(i,j,k)
    {
#pragma omp for  schedule(static)
        for (i = 0; i < N; i = i + 1) {
            for (j = 0; j < N; j = j + 1) {
                a[i][j] = 0.;
                for (k = 0; k < N; k = k + 1) {
                    a[i][j] = (a[i][j]) + ((b[i][k]) * (c[k][j]));
                }
            }
        }
    }
    end = omp_get_wtime();
    *totalP = end - start;
    start = 0.0;
    end = 0.0;
    start = omp_get_wtime();
    for (i = 0; i < N; i = i + 1) {
        for (j = 0; j < N; j = j + 1) {
            a[i][j] = 0.;
            for (k = 0; k < N; k = k + 1) {
                a[i][j] = (a[i][j]) + ((b[i][k]) * (c[k][j]));
            }
        }
    }
    end = omp_get_wtime();
    *totalS = end - start;
}

void SumRow(int a[][MAX_SIZE], int b[][MAX_SIZE], int c[][MAX_SIZE], int N, double* totalP, double* totalS)
{
    int sumA, sumB;
    double start, end;
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            sumA = sumA + a[i][j];
            sumB = sumB + b[i][j];
        }
    }

    end = omp_get_wtime();
    *totalP = end - start;
    start = 0.0;
    end = 0.0;
    start = omp_get_wtime();
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            sumA = sumA + a[i][j];
            sumB = sumB + b[i][j];
        }
    }
    end = omp_get_wtime();
    *totalS = end - start;
}

void SumColumn(int a[][MAX_SIZE], int b[][MAX_SIZE], int c[][MAX_SIZE], int N, double* totalP, double* totalS)
{
    int sumA, sumB;
    double start, end;
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            sumA = sumA + a[i][j];
            sumB = sumB + b[i][j];
        }
    }

    end = omp_get_wtime();
    *totalP = end - start;
    start = 0.0;
    end = 0.0;
    start = omp_get_wtime();
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            sumA = sumA + a[i][j];
            sumB = sumB + b[i][j];
        }
    }
    end = omp_get_wtime();
    *totalS = end - start;
}

void LUFactorization(int a[][MAX_SIZE], int b[][MAX_SIZE], int c[][MAX_SIZE], int N, double* totalP, double* totalS)
{
    int lower[MAX_SIZE][MAX_SIZE], upper[MAX_SIZE][MAX_SIZE];
    memset(lower, 0, sizeof(lower));
    memset(upper, 0, sizeof(upper));
    double start, end;
    start = omp_get_wtime();
#pragma omp parallel for schedule(static)

    for (int i = 0; i < N; i++) {

        for (int k = i; k < N; k++) {


            int sum = 0;
            for (int j = 0; j < i; j++)
                sum += (lower[i][j] * upper[j][k]);

            upper[i][k] = a[i][k] - sum;
        }

        // Lower Triangular
        for (int k = i; k < N; k++) {
            if (i == k)
                lower[i][i] = 1; // Diagonal as 1
            else {

                // Summation of L(k, j) * U(j, i)
                int sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (lower[k][j] * upper[j][i]);

                // Evaluating L(k, i)
                lower[k][i] = (a[k][i] - sum) / upper[i][i];
            }
        }
    }

    end = omp_get_wtime();
    *totalP = end - start;
    start = 0.0;
    end = 0.0;
    start = omp_get_wtime();
    for (int i = 0; i < N; i++) {

        // Upper Triangular
        for (int k = i; k < N; k++) {

            // Summation of L(i, j) * U(j, k)
            int sum = 0;
            for (int j = 0; j < i; j++)
                sum += (lower[i][j] * upper[j][k]);

            // Evaluating U(i, k)
            upper[i][k] = a[i][k] - sum;
        }

        // Lower Triangular
        for (int k = i; k < N; k++) {
            if (i == k)
                lower[i][i] = 1; // Diagonal as 1
            else {

                // Summation of L(k, j) * U(j, i)
                int sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (lower[k][j] * upper[j][i]);

                // Evaluating L(k, i)
                lower[k][i] = (a[k][i] - sum) / upper[i][i];
            }
        }
    }
    end = omp_get_wtime();
    *totalS = end - start;
}
