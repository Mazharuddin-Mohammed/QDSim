# QDSim Performance Benchmark Report

## 1. Mesh Size Benchmarks

| Mesh Size | Total Time (s) |
|-----------|---------------|
| 21x21 | 0.2080 |
| 51x51 | 1.6223 |
| 101x101 | 7.4760 |
| 151x151 | 21.3075 |
| 201x201 | 47.8463 |

## 2. Element Order Benchmarks

| Element Order | Total Time (s) |
|--------------|---------------|
| P1 | 1.7345 |
| P2 | 2.3834 |
| P3 | 4.7608 |

## 3. Parallel Configuration Benchmarks

Parallel configuration benchmarks were not run or failed.

## 4. GPU Acceleration Benchmarks

GPU acceleration benchmarks were not run or failed.

## Summary

Based on the benchmarks, the following recommendations can be made:

1. **Mesh Size**: A mesh size of 21x21 provides a good balance between accuracy and performance.

2. **Element Order**: P1 elements provide the best performance for this problem.

3. **Parallel Processing**: Parallel processing was not tested or more configurations should be tested.

4. **GPU Acceleration**: GPU acceleration was not tested or is not available.

