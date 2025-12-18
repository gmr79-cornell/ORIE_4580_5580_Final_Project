# ORIE_4580_Final_Project

# GPU Scheduling Simulation for LLM Inference

## Overview
This project studies how different GPU scheduling policies affect **latency** and
**throughput** for large language model (LLM) inference. Using a discrete-event
simulation, we model how requests arrive over time, how they are processed by a
GPU, and how batching decisions influence user-perceived performance.

The goal is to understand **when batching helps** and **when it introduces
unnecessary delay**, particularly for metrics such as Time To First Token (TTFT)
and total response time.

---

## Key Questions
- How do different GPU scheduling strategies affect user latency?
- Does batching improve throughput under realistic workloads?
- What tradeoffs arise between responsiveness and capacity?
- When is a simple scheduling policy preferable to a more complex one?

---

## Scheduling Policies Studied

### 1. Process-to-Completion (P2C)
- Requests are processed one at a time.
- Each request uses the GPU exclusively until completion.
- No batching is performed.
- Serves as a low-latency baseline.

### 2. Prefill-Prioritizing Batched Decode (PPB)
- Requests are batched during the prefill phase.
- Decode proceeds in synchronized rounds, generating one token per request per
  batch step.
- Prefill is always prioritized over decode.
- Models a more realistic LLM inference pipeline.

---

## Simulation Model

### Arrival Process
- Requests arrive according to a Poisson process with rate $$\lambda$$.

### Request Structure
- Each request has a fixed prompt length $$L$$ and output length $$B$$.

### GPU Service-Time Model
GPU processing time depends on the number of tokens processed together:

$$
S(b) = c + a \cdot \max(0, b - b_0),
$$

where:
- $$c$$ is a fixed batch setup cost,
- $$a$$ is the marginal per-token processing cost,
- $$b_0$$ is a threshold below which the GPU is underutilized.

---

## Metrics Collected
- **Time To First Token (TTFT)**
- **Total latency**
- **Average Time Between Tokens (TBT)**
- **Throughput (requests per second)**

These metrics directly reflect user experience and system capacity.

---

## Validation
The simulation engine is validated using a classical M/M/1 queue. For arrival
rate $$\lambda$$ and service rate $$\mu$$, theory predicts:

$$
E[T] = \frac{1}{\mu - \lambda}.
$$

Simulated results closely match this expression, confirming the correctness of
the event logic and queueing behavior.

---

## Results Summary
- P2C consistently achieves lower TTFT and lower total latency.
- PPB introduces batching delays that increase latency.
- Throughput under both policies closely matches the offered load.
- Under homogeneous workloads and moderate load, batching does not improve
  throughput.

---

## Repository Contents
- **Simulation code**: discrete-event simulator implementing P2C and PPB
- **Validation code**: M/M/1 simulation and theoretical comparison
- **Plots**:
  - `policy_comparison_tradeoffs.png`
- **Tables**:
  - `policy_comparison_summary.csv`

---

## How to Run
1. Open the provided Jupyter notebook.
2. Run all cells from top to bottom.
3. Results will be printed to the console and plots will be saved to disk.

---

## Intended Audience
- Engineering managers evaluating inference system design
- Researchers studying scheduling and queueing tradeoffs
- Students learning discrete-event simulation and performance modeling

---

## Future Extensions
- Heterogeneous prompt and output lengths
- Multi-GPU systems
- Alternative scheduling strategies
- Sensitivity analysis under heavy load

---

## Course Context
This project was developed as part of an ORIE 4580 course, with an emphasis on clarity,
validation, and decision-relevant insights.
