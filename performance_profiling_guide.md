# Performance Profiling

This guide explains how to profile the application using `cProfile` and `SnakeViz` to find performance bottlenecks.

## Requirements

You will need `snakeviz` to visualize the results:

```bash
  pip install snakeviz
```

## How to profile

### 1. Generate Stats File:

Run the application via `cProfile` to create a `.prof` statistics file.

```bash
  python -m cProfile -o profile_results.prof main.py
```

The application will launch. Use it for 15-30 seconds to gather data, then press `q` to quit.

### 2. Visualize the Results:

Run snakeviz on the file you just created.

```bash
  snakeviz profile_results.prof
```

This will automatically open an interactive report in your web browser.

## Reading the report

Look at the Icicle graph or sort the table by `tottime` (total time).

The widest blocks in the graph are the bottlenecks.

`tottime`: Time spent only in that function.

`cumtime`: Time spent in that function plus all functions it called.

Functions with a high `tottime` are your main targets for optimization.