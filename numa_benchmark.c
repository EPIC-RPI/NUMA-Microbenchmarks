#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <numa.h>
#include <numaif.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

#define NUM_ITERATIONS 10

void bandwidth_throughput_benchmark(int node, long long array_size)
{
    long long *array, *indices;
    double start, end;
    double elapsed_seq_write, elapsed_seq_read, elapsed_rand_write, elapsed_rand_read;

    // Allocate memory on the specified NUMA node
    array = numa_alloc_onnode(array_size * sizeof(long long), node);
    indices = numa_alloc_onnode(array_size * sizeof(long long), node);
    if (array == NULL || indices == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    printf("NUMA Node %d (Array size: %lld):\n", node, array_size);

// Generate random indices for random access
#pragma omp parallel for
    for (long long i = 0; i < array_size; i++)
    {
        indices[i] = rand() % array_size;
    }

    // Sequential Write benchmark
    start = omp_get_wtime();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
#pragma omp parallel for
        for (long long j = 0; j < array_size; j++)
        {
            array[j] = j;
        }
    }
    end = omp_get_wtime();
    elapsed_seq_write = end - start;

    double seq_write_bw = (array_size * sizeof(long long) * NUM_ITERATIONS) / (elapsed_seq_write * 1e9);
    double seq_write_tp = (array_size * NUM_ITERATIONS) / elapsed_seq_write;
    printf("  Sequential Write: %.3f s, Bandwidth: %.2f GB/s, Throughput: %.2e ops/s\n",
           elapsed_seq_write, seq_write_bw, seq_write_tp);

    // Sequential Read benchmark
    volatile long long sum = 0;
    start = omp_get_wtime();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
#pragma omp parallel for reduction(+ : sum)
        for (long long j = 0; j < array_size; j++)
        {
            sum += array[j];
        }
    }
    end = omp_get_wtime();
    elapsed_seq_read = end - start;

    double seq_read_bw = (array_size * sizeof(long long) * NUM_ITERATIONS) / (elapsed_seq_read * 1e9);
    double seq_read_tp = (array_size * NUM_ITERATIONS) / elapsed_seq_read;
    printf("  Sequential Read:  %.3f s, Bandwidth: %.2f GB/s, Throughput: %.2e ops/s\n",
           elapsed_seq_read, seq_read_bw, seq_read_tp);

    // Random Write benchmark
    start = omp_get_wtime();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
#pragma omp parallel for
        for (long long j = 0; j < array_size; j++)
        {
            array[indices[j]] = j;
        }
    }
    end = omp_get_wtime();
    elapsed_rand_write = end - start;

    double rand_write_bw = (array_size * sizeof(long long) * NUM_ITERATIONS) / (elapsed_rand_write * 1e9);
    double rand_write_tp = (array_size * NUM_ITERATIONS) / elapsed_rand_write;
    printf("  Random Write:     %.3f s, Bandwidth: %.2f GB/s, Throughput: %.2e ops/s\n",
           elapsed_rand_write, rand_write_bw, rand_write_tp);

    // Random Read benchmark
    sum = 0;
    start = omp_get_wtime();
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
#pragma omp parallel for reduction(+ : sum)
        for (long long j = 0; j < array_size; j++)
        {
            sum += array[indices[j]];
        }
    }
    end = omp_get_wtime();
    elapsed_rand_read = end - start;

    double rand_read_bw = (array_size * sizeof(long long) * NUM_ITERATIONS) / (elapsed_rand_read * 1e9);
    double rand_read_tp = (array_size * NUM_ITERATIONS) / elapsed_rand_read;
    printf("  Random Read:      %.3f s, Bandwidth: %.2f GB/s, Throughput: %.2e ops/s\n",
           elapsed_rand_read, rand_read_bw, rand_read_tp);

    // Free the allocated memory
    numa_free(array, array_size * sizeof(long long));
    numa_free(indices, array_size * sizeof(long long));
}

void matmul_benchmark(int node, int matrix_size)
{
    double *a, *b, *c;
    double start, end;
    double elapsed;

    // Allocate matrices on the specified NUMA node
    a = numa_alloc_onnode(matrix_size * matrix_size * sizeof(double), node);
    b = numa_alloc_onnode(matrix_size * matrix_size * sizeof(double), node);
    c = numa_alloc_onnode(matrix_size * matrix_size * sizeof(double), node);

    if (a == NULL || b == NULL || c == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

// Initialize matrices
#pragma omp parallel for
    for (int i = 0; i < matrix_size * matrix_size; i++)
    {
        a[i] = (double)rand() / RAND_MAX;
        b[i] = (double)rand() / RAND_MAX;
        c[i] = 0.0;
    }

    // Matrix multiplication
    start = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            for (int k = 0; k < matrix_size; k++)
            {
                c[i * matrix_size + j] += a[i * matrix_size + k] * b[k * matrix_size + j];
            }
        }
    }
    end = omp_get_wtime();

    elapsed = end - start;

    printf("NUMA Node %d: Matrix Multiplication (%dx%d, %.2f GB)\n", node, matrix_size, matrix_size, 3.0 * matrix_size * matrix_size * sizeof(double) / 1e9);
    printf("  Time: %.3f seconds\n", elapsed);
    printf("  Performance: %.2f GFLOPS\n", 2.0 * matrix_size * matrix_size * matrix_size / (elapsed * 1e9));

    // Free the allocated memory
    numa_free(a, matrix_size * matrix_size * sizeof(double));
    numa_free(b, matrix_size * matrix_size * sizeof(double));
    numa_free(c, matrix_size * matrix_size * sizeof(double));
}

void print_usage(char *program_name)
{
    fprintf(stderr, "Usage: %s <node_id> <array_size> <matrix_size>\n", program_name);
    fprintf(stderr, "  node_id: NUMA node ID to run the benchmark on (-1 for all nodes)\n");
    fprintf(stderr, "  array_size: Number of elements for the bandwidth benchmark\n");
    fprintf(stderr, "  matrix_size: Size of the square matrix for the matrix multiplication benchmark\n");
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        print_usage(argv[0]);
        return 1;
    }

    srand(42);

    int node_id = atoi(argv[1]);
    long long array_size = atoll(argv[2]);
    int matrix_size = atoi(argv[3]);

    if (numa_available() == -1)
    {
        fprintf(stderr, "NUMA is not available on this system\n");
        return 1;
    }

    int num_nodes = numa_num_configured_nodes();
    printf("Number of NUMA nodes: %d\n\n", num_nodes);

    if (node_id >= num_nodes)
    {
        fprintf(stderr, "Invalid node ID. Maximum node ID is %d\n", num_nodes - 1);
        return 1;
    }

    if (node_id == -1)
    {
        for (int i = 0; i < num_nodes; i++)
        {
            bandwidth_throughput_benchmark(i, array_size);
            printf("\n");
            matmul_benchmark(i, matrix_size);
            printf("\n");
        }
    }
    else
    {
        bandwidth_throughput_benchmark(node_id, array_size);
        printf("\n");
        matmul_benchmark(node_id, matrix_size);
        printf("\n");
    }

    return 0;
}