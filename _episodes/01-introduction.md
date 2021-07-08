---
title: "Introduction"
teaching: 0
exercises: 0
questions:
- "Key question (FIXME)"
objectives:
- "First learning objective. (FIXME)"
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

- [1. Overview](#1-overview)
- [2. CUDA Memory Model](#2-cuda-memory-model)

## 1. Overview

In [MolSSI's Fundamentals of Heterogeneous Parallel Programming with CUDA
 C/C++ at the beginner level](http://education.molssi.org/gpu_programming_beginner),
we have provided a comprehensive presentation of the CUDA programming, compilation 
and execution models. These models layout the fundamental aspects of CUDA programming
platform and expose various conceptual parallelism abstractions at the logical 
architectural and application levels.

The present tutorial extends the scope of Nvidia's heterogeneous parallelization platform to
 **CUDA memory model**, which exposes a unified hierarchical memory abstraction for both 
 host and device memory systems. It is also founded on Nvidia's [Best Practices 
 Guide for CUDA C/C++](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
 and encourages users to follow the **Asses**, **Parallelize**, **Optimize** and **Deploy**
 ([**APOD**](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#assess-parallelize-optimize-deploy))
 application design cycle for an efficient and rapid recognition of the parallelization 
 opportunities in programs and improving the code quality and performance. In our analysis for
 performance improvement and code optimization, we will adopt a quantitative profile-driven
 approach and make an extensive use of profiling tools provided by Nvidia's **Nsight System**.

Before we begin writing code, let us delve into the important aspects of the of CUDA memory model
in more details.

## 2. CUDA Memory Model
 


 The CUDA memory model is based on the *locality principle* which allows users to fully take
 control of the data flow within programmable memory levels such as registers, 
 shared memory *etc.* .



{% include links.md %}

