# NUMA Microbenchmarks

## Dependency
- libnuma-dev
- libomp-dev

## Compile
```bash
gcc -O2 -fopenmp -o build/numa_benchmark numa_benchmark.c -lnuma
```

## Run
```bash
sudo build/numa_benchmark <node_id> <array_size> <matrix_size>
```

## Test

- Small:  10000000    1000
- Middle: 100000000   5000
- Large:  1000000000  9000

```bash
sudo build/numa_benchmark 2 10000000 1000
sudo build/numa_benchmark 2 100000000 5000
sudo build/numa_benchmark 2 1000000000 9000
```