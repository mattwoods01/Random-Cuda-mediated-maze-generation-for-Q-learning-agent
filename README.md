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

<img width="810" height="825" alt="image" src="https://github.com/user-attachments/assets/4e016cbb-26c3-44fd-8480-c7163ff1118a" />

### 1. Random Maze Initialization
- Generates a random binary grid (`0 = open`, `1 = wall`)
- Uses GPU-based random number generation
- Start and end positions are tracked during generation

**Control version:** Uses global memory only (no shared memory)

<img width="397" height="401" alt="image" src="https://github.com/user-attachments/assets/b66e8b21-63c5-4d6b-816e-f5aad4a942c0" />

---

### 2. Maze Randomization Pass
- Copies the maze into shared memory
- Applies additional randomization to improve structure
- Writes results back to global memory

**Control version:** Operates directly on global memory

<img width="473" height="474" alt="image" src="https://github.com/user-attachments/assets/7fe6f14d-5910-4728-8f55-29f6951c3551" />

---

### 3. Feature Injection
- Inserts terrain or feature blocks into the maze
- Feature templates are stored in shared memory
- Features are placed at random valid locations

<img width="434" height="435" alt="image" src="https://github.com/user-attachments/assets/933f839f-ec59-4137-8645-c63597fb13ee" />

---

### 4. Guaranteed Path Creation
- Ensures a valid path exists from start to end
- Opens corridors along selected rows and columns
- Applies changes selectively to avoid overly open mazes

<img width="527" height="531" alt="image" src="https://github.com/user-attachments/assets/911dfcbc-6123-4d54-95ac-b403e46d4e1d" />

---

### 5. DFS Terrain Modifier
- Runs multiple parallel Depth-First Searches
- Identifies reachable areas from the goal
- Removes unreachable regions to enforce solvability
- Produces more realistic maze topology

<img width="434" height="430" alt="image" src="https://github.com/user-attachments/assets/bf83323b-1f98-4c3c-9628-ac54eee12eba" />

---

### 6. Epsilon Schedule Generation (RL Support)
- Computes epsilon decay values for Q-learning
- Uses asynchronous memory transfers and CUDA streams
- Initialized once before training begins

**Control version:** No async memory or shared memory usage

<img width="538" height="410" alt="image" src="https://github.com/user-attachments/assets/bc362cb6-6243-4c00-a818-957d7e6ddaec" />

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

