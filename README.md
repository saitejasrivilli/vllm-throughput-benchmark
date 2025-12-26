# vllm-throughput-benchmark
Benchmarked LLM inference throughput using vLLM vs Hugging Face generation, focusing on batching, KV cache behavior, and scalability under concurrent requests.
# High-Throughput LLM Inference Benchmarking with vLLM

This project benchmarks LLM inference throughput using vLLM and compares it against Hugging Face generation under concurrent load.

The goal is to understand how dynamic batching and KV cache management impact throughput, latency, and GPU utilization in real-world LLM serving scenarios.

## Environment

- Platform: Google Colab (free tier)
- GPU: NVIDIA T4 or L4
- Frameworks: vLLM, Hugging Face Transformers
- Model (initial): TinyLlama 1.1B Chat

## Current Status

- vLLM environment setup and sanity check completed
- Deterministic generation configured for benchmarking

## Next Steps

- Implement Hugging Face baseline generation
- Measure token throughput under varying concurrency
- Compare throughput, latency, and memory usage

## How to Run

Open `vllm_throughput_benchmark.ipynb` in Google Colab, enable GPU runtime, and run all cells from top to bottom.

## Motivation

Most LLM performance issues in production come from inference pipelines rather than model quality. This project focuses on understanding and measuring inference behavior under realistic serving conditions.
