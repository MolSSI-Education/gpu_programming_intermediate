---
title: "CUDA Memory Model: Basics"
teaching: 0
exercises: 0
questions:
- "What is CUDA Memory Model?"
- "What is the principle of locality and how does it reduce the memory access latency?"
- "Why is there a memory hierarchy and how is it defined?"
objectives:
- "Understanding the CUDA Memory Model and its role in CUDA C/C++ programming."
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

- [1. Overview](#1-overview)
- [2. CUDA Memory Model](#2-cuda-memory-model)
  - [2.1. Principle of Locality](#21-principle-of-locality)
  - [2.2. Memory Hierarchy](#22-memory-hierarchy)
    - [2.2.1. Registers](#221-registers)

## 1. Overview

In [MolSSI's Fundamentals of Heterogeneous Parallel Programming with CUDA
 C/C++ at the beginner level](http://education.molssi.org/gpu_programming_beginner),
we have provided a comprehensive presentation of the CUDA programming, compilation 
and execution models. These models layout the fundamental aspects of CUDA programming
platform and expose various conceptual parallelism abstractions at the logical 
architectural and application levels.

The present tutorial extends the scope of NVIDIA's heterogeneous parallelization platform to
 **CUDA memory model**, which exposes a unified hierarchical memory abstraction for both 
 host and device memory systems. It is also founded on NVIDIA's [Best Practices 
 Guide for CUDA C/C++](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
 and encourages users to follow the **Asses**, **Parallelize**, **Optimize** and **Deploy**
 ([**APOD**](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#assess-parallelize-optimize-deploy))
 application design cycle for an efficient and rapid recognition of the parallelization 
 opportunities in programs and improving the code quality and performance. In our analysis for
 performance improvement and code optimization, we will adopt a quantitative profile-driven
 approach and make an extensive use of profiling tools provided by NVIDIA's **Nsight System**.


Before we begin writing code, let us delve into the important aspects of the of CUDA memory model
in more details.

## 2. CUDA Memory Model


### 2.1. Principle of Locality

 The CUDA memory model is based on the **locality principle** which reduces the memory access 
 latency through an efficient way of reusing data. There are two types of locality:

- **Spatial locality** (locality in space)
- **Temporal locality** (locality in time)

Spatial locality assumes that once a memory address is referenced, its neighboring memory locations
become more likely to be referenced as well. As example is when the processor attempts to access a
contiguous array of data stored on the global memory. Temporal locality assumes that once a memory
location is accessed, there is a higher probability for it to be referenced again in a short period
of time and lower probabilities at later times.

In [Fundamentals of Heterogeneous Parallel Programming with CUDA C/C++](http://education.molssi.org/gpu_programming_beginner/
{% link _episodes/01-introduction.md %}#2-parallel-programming-paradigms), we described the main 
features of a typical modern GPU architecture which comparing them with those of GPU. There, 
we explained that one of the most important hardware features of the CPU is its relatively 
large cache memory size which allows it to improve the application optimization process by 
benefiting from temporal and spatial locality.

Allows users to fully take control of the data flow within programmable memory levels such as registers,
 shared memory *etc.* . Here, 

### 2.2. Memory Hierarchy

In order to improve the performance of the memory operations, CUDA memory model adopts 
the **memory hierarchy** consisting of various memory levels with different bandwidths, latencies, 
and capacities. Within this hierarchy, as the capacity of the memory type increases, the latency also
increases. 

As we discussed in [Fundamentals of Heterogeneous Parallel Programming with CUDA C/C++](http://education.molssi.org/
gpu_programming_beginner/{% link _episodes/01-introduction.md %}#2-parallel-programming-paradigms), 
both CPU and GPU main memory spaces are constructed by dynamic random access memory (DRAM). The lower-latency 
memory units such as cache, however, are built using static random access memory (SRAM). As such, 
based on the memory hierarchy, it would be logical to keep the data that are actively used by the processor
in the low-latency and low-capacity memory spaces and store the less frequently used ones in high-latency high-capacity
memory spaces for possible future usage. 

Although both CPU and GPU adopt similar hierarchical memory design models, CUDA programming model exposes much more 
control over and access to memory levels in the hierarchy than what is possible with CPUs. There are two main memory 
categories:

- **Non-programmable** where programmer has no control over data flow in the memory unit, and
- **Programmable** where the user is in charge of data load/storage within the memory unit.

The CPU L1 and L2 cache are examples of non-programmable memories. Nevertheless, CUDA memory 
model exposes several types of programmable memory spaces on the device, each with its own 
*lifetime*, *scope* and *caching rules*:

- **Registers**
- **Shared memory**
- **Local memory**
- **Constant memory**
- **Texture memory**
- **Global memory**

The following figure provides a simplified representation of the memory hierarchy.

![Figure 1]()

As the figure illustrates, each thread within a kernel has its own private local memory.  
Shared memory belongs to all threads in a block. The contents in the shared memory are accessible
to all threads within a block and have the same lifetime as that of the thread block. The contents of
constant, texture and global memories have the same lifetime as that of the application and are 
accessible to all threads on the device; however, their applications are quite different, which we 
will explain, shortly.

#### 2.2.1. Registers

Registers are precious resources partitioned among active warps on the GPU with the lowest capacity 
and highest data transfer speed. Variables stored on registers such as automatic kernel variables 
declared without qualifiers or arrays declared in kernels with constant referencing indices determined 
at the compilation time, are private to each thread. According to the principle of locality, the data 
being held in registers are often frequently accessed by kernels while their lifetime ends with the 
completion of kernel execution.

Using fewer registers within kernels can lead to higher performance resulting from the increased 
**occupancy** of the thread locks per streaming multiprocessor (SM).

> ## Note:
> Occupancy is a helpful performance metric which is based on the ideal intention of 
> keeping as many device cores occupied as possible. Occupancy is defined as the ration
> of active warps to the maximum number of warps per SM. Although it is a useful metric for
> analysis and description of the observed benchmark profiling logs, it should not be a hard
> and only reference for code optimization. There can be many cases that increased occupancy
> does not always mean improved performance. We will discuss profiling performance metrics in
> [here]().
{: .discussion}



{% include links.md %}

