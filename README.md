# Random Cuda Mediated maze generation for Q learning agent
Project that utilizes CUDA programming and pybind to create test maze courses for Q-learning agent

Dependencies:
Numpy
Matplotlib
CUDA
Pybind11
Python3.9

1. Build with Cmake
2. Use Visual studio 2019
3. generate and open project
4. build project using release settings
5. Run python scripts

# CUDA-Mediated Maze Generation

This repository contains a CUDA-based pipeline for **fast, randomized maze generation** designed for use in **reinforcement learning (RL)** and **Q-learning** experiments. The goal is to efficiently generate large numbers of **solvable, feature-rich mazes** without manual design bias.

---

## Motivation

Training RL agents requires exposure to many diverse environments. Hand-designed mazes are:
- Time-consuming to create
- Biased by human design choices
- Difficult to scale for large experiments

This project solves that problem by using **GPU-parallelized CUDA kernels** to generate mazes with random structure, guaranteed paths, and configurable terrain features.

---

## High-Level Pipeline

The maze generation workflow is split into modular CUDA kernels. Each step can be tuned or replaced independently.

### 1. Random Maze Initialization
- Generates a random binary grid (`0 = open`, `1 = wall`)
- Uses GPU-based random number generation
- Start and end positions are tracked during generation

**Control version:** Uses global memory only (no shared memory)

---

### 2. Maze Randomization Pass
- Copies the maze into shared memory
- Applies additional randomization to improve structure
- Writes results back to global memory

**Control version:** Operates directly on global memory

---

### 3. Feature Injection
- Inserts terrain or feature blocks into the maze
- Feature templates are stored in shared memory
- Features are placed at random valid locations

---

### 4. Guaranteed Path Creation
- Ensures a valid path exists from start to end
- Opens corridors along selected rows and columns
- Applies changes selectively to avoid overly open mazes

---

### 5. DFS Terrain Modifier
- Runs multiple parallel Depth-First Searches
- Identifies reachable areas from the goal
- Removes unreachable regions to enforce solvability
- Produces more realistic maze topology

---

### 6. Epsilon Schedule Generation (RL Support)
- Computes epsilon decay values for Q-learning
- Uses asynchronous memory transfers and CUDA streams
- Initialized once before training begins

**Control version:** No async memory or shared memory usage

---

## Performance Testing

Performance was evaluated by:
- Running each kernel and the full pipeline multiple times
- Averaging execution times across runs
- Comparing shared-memory + async versions against control kernels

**Testing conditions:**
- Maze size: `40 x 40`
- Identical start and end points
- Identical random seeds for fair comparison

The epsilon schedule kernel was excluded from full-pipeline timing since it is only initialized once per RL run.

---

## Results Summary

- CUDA acceleration significantly reduced maze generation time
- Shared memory and asynchronous execution improved performance in most cases
- Some kernels benefit more than others depending on memory access patterns

---

## Limitations & Future Improvements

Planned enhancements include:
- Parallel generation of maze blocks and stitching them together
- Additional user-configurable parameters
- Improved path-guarantee logic without brute-force bias
- Better balance between randomness and structural complexity

---

## Conclusion

This project demonstrates that **CUDA-based parallelism** is an effective way to generate large numbers of **diverse, solvable mazes** for reinforcement learning. The modular design makes it easy to extend, benchmark, and integrate into RL pipelines.

---

*Source: Project documentation provided by the author*

