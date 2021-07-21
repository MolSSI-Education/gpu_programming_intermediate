---
title: "CUDA Memory Model: Basics"
teaching: 0
exercises: 0
questions:
- "What is CUDA Memory Model?"
- "What is the principle of locality and how does it reduce the memory access latency?"
- "Why is there a memory hierarchy and how is it defined?"
objectives:
- "Understanding the CUDA Memory Model and its role in CUDA C/C++ programming"
keypoints:
- "First key point. Brief Answer to questions. (FIXME)"
---

- [1. Overview](#1-overview)
- [2. CUDA Memory Model](#2-cuda-memory-model)
  - [2.1. Principle of Locality](#21-principle-of-locality)
  - [2.2. Memory Hierarchy](#22-memory-hierarchy)
    - [2.2.1. Registers](#221-registers)
    - [2.2.2. Local Memory](#222-local-memory)
    - [2.2.3. Shared Memory](#223-shared-memory)
    - [2.2.4. Constant Memory](#224-constant-memory)
    - [2.2.5. Texture Memory](#225-texture-memory)
    - [2.2.6. Global Memory](#226-global-memory)
  - [2.3. Host-Device Memory Management](#23-host-device-memory-management)
    - [2.3.1. Pinned Memory](#231-pinned-memory)
    - [2.3.2. Zero-copy Memory](#232-zero-copy-memory)
    - [2.3.3. Unified Memory](#233-unified-memory)
- [3. Example: Vector Addition (AXPY)](#3-example-vector-addition-axpy)
- [4. Example: Matrix Addition](#4-example-matrix-addition)

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
[**occupancy**](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html#group__CUDART__OCCUPANCY)
of the thread locks per streaming multiprocessor (SM).

> ## Note:
> Occupancy is a helpful performance metric which is based on the ideal intention of 
> keeping as many device cores occupied as possible. Occupancy is defined as the ration
> of active warps to the maximum number of warps per SM. Although it is a useful metric for
> analysis and description of the observed benchmark profiling logs, it should not be a hard
> and only reference for code optimization. There can be many cases that increased occupancy
> does not always mean improved performance. We will discuss profiling performance metrics in
> [here]().
{: .discussion}

On the other hand, if a kernel attempts to utilize more registers than the limit imposed by the 
hardware resources, the excess memory would spill over local memory. The **register spill** and 
[**register pressure**](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#register-pressure)
should be avoided if possible due to its serious performance consequences. Here, we provide two ways
to control the number of registers: 1) `-maxrregcount` compiler option, and 2) 
[`__launch_bounds__()` qualifier method](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds).
The former approach can be used by simply passing the [`-maxrregcount=N` option](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation-maxrregcount) to the nvcc 
compiler where `N` denotes the maximum number of registers *used by all kernels*. The latter method uses the 
`__launch_bounds__()` qualifier method after the kernel declaration specification qualifier in order to provide
the necessary information to the compiler through its arguments, `maxThreadsPerBlock` and `minBlocksPerMultiProcessor` as

~~~
__global__ void __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
kernel(...) {
  // kernel body implementation
}
~~~
{: .language-cuda}

Here, `maxThreadsPerBlock` specifies the maximum number of threads per block with which the `kernel()` is launched.
The `minBlocksPerMultiprocessor` is an optional argument which denotes the desired minimum number of resident thread 
blocks per SM.

The provided information through `__launch_bounds__()` method take precedence over that provided 
by the compiler option `-maxrregcount` and in cases where both methods are adopted, the latter is ignored.

#### 2.2.2. Local Memory

Kernel variables that cannot fit into registers create a *register pressure* and *spill* into local memory.
Variables types that are eligible to be stored in local memory are: 1) local arrays with reference indices that
cannot be inferred at compilation time, and 2) any variable (such as local arrays or structures) that are too large
to fit in register.

Note that those data that are spilled into the *local* memory reside in the same physical location as *global* memory. 
Therefore, significant performance degradation is expected as the data access/transfer will now be subjected to the 
low bandwidth and high latency limitations of the global memory.

> ## Note:
> The resident data in local memory are cached in each SM's L1 and each device's L2 cache memory spaces for GPUs 
> with compute capability 2.0 and higher.
{: .discussion}

#### 2.2.3. Shared Memory

Similar to registers, shared memory is a valuable on-chip programmable memory resource with significantly lower latency 
and higher bandwidth than those of local/global memory. The `__shared__` qualifier can be used for explicit shared memory 
variable declaration. Shared memory can be allocated statically or dynamically and variables can be declared within the 
global or kernel's local scope. Let us statically allocate the shared memory for a 2-dimensional array of integers

~~~
__shared__ int array[dimX][dimY];
~~~
{: .language-cuda}

where `dimX` and `dimY` are predefined integer variables. If the size of the required shared memory (in this case, 
`dimX * dimY * sizeof(int)`) is not known at the compilation time, the memory block for the array of variable size 
can be allocated dynamically using the `extern` keyword

~~~
extern __shared__ int array[];
~~~
{: .language-cuda}

The postponed specification of the desired allocated memory size for the array should now be defined at the run-time
for each thread as the third argument in the execution configuration (triple angular brackets)

~~~
kernel<<< numberOfBlocksInGrid, numberOfThreadsinBlock, dimX * dimY * sizeof(int) >>>(array, ...)
~~~
{: .language-cuda}

where the desired size should be expressed in bytes, hence the use of [`sizeof()`](https://en.cppreference.com/w/cpp/language/sizeof).
> ## Note:
> Only 1-dimensional arrays can be declared dynamically in shared memory.
{: .discussion}

Since shared memory is distributed among thread blocks and is key for intra-block/inter-thread cooperation, a naive usage 
of shared memory can limit the number of active warps and affect the performance. Furthermore, the lifetime and scope of 
shared memory is limited by those of kernels and when the thread block finishes its execution, the allocated shared memory
for that block is released and becomes available to other thread blocks.

In order to create an explicit barrier for synchronization of all threads in the same thread block, CUDA runtime introduces the
following functionality

~~~
void __syncthreads();
~~~
{: .language-cuda}

which is especially useful for preventing data race [hazards](https://docs.nvidia.com/cuda/cuda-memcheck/index.html#what-are-hazards)
in parallel applications. Data hazards often happen when multiple threads attempt to access a memory address in an arbitrary order 
where at least one thread performs a store (or write) operation. Care must be taken with the usage 
of [`__syncthreads()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) since it can negatively
affect the performance by stalling the SM, frequently.

The on-chip memory space and hardware resources used for both L1 cache and shared memory is statically partitioned by default. 
However, this configuration can be dynamically modified at using the CUDA runtime function [`cudaFuncSetCacheConfig()`](https://docs.nvidia.com/
cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g6699ca1943ac2655effa0d571b2f4f15)

~~~
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig);
~~~
{: .language-cuda}

The first argument, `func`, denotes the device function symbol and [`cudaFuncCache`](https://docs.nvidia.com/cuda/
cuda-runtime-api/group__CUDART__TYPES.html) is an `enum` type variable that stands for the CUDA cache configurations 
and can take the following values

| Cache Configuration       | Value |                     Meaning                      |
| :------------------------ | :---: | :----------------------------------------------: |
| cudaFuncCachePreferNone   |   0   |      No preference (default configuration)       |
| cudaFuncCachePreferShared |   1   | Prefer larger shared memory and smaller L1 cache |
| cudaFuncCachePreferL1     |   2   | Prefer larger L1 cache and smaller shared memory |
| cudaFuncCachePreferEqual  |   3   |   Prefer equal size L1 cache and shared memory   |

> ## Note:
> The `cudaFuncSetCacheConfig()` function does nothing on devices with fixed L1 cache and shared memory sizes.
{: .discussion}

#### 2.2.4. Constant Memory

Variables can be declared in constant memory space through using the `__constant__` qualifier.
The constant memory variables must be declared in global scope. Furthermore, the amount of constant memory that
can be declared is limited: 64 kB for all compute capabilities. Moreover, the constant memory is statically allocated
and its content is visible to all threads and kernels in the read-only mode. The best performance from using constant 
memory is expected when all threads within a warp read from the same memory address: here the contents of the constant 
memory location is broadcasted to all threads in a warp through a single load operation. For example, a numerical constant
can be stored in constant memory and read by threads in warp(s) to scale the components of an array.

The [`cudaMemcpyToSymbol()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
#group__CUDART__MEMORY_1g9bcf02b53644eee2bef9983d807084c7) function can be used to initialize the constant memory from
the host

~~~
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count);
~~~
{: .language-cuda}

Here, `count` bytes from the memory address pointed to by the pointer variable `src` is copied to the memory location
pointed to by `symbol` residing in the constant or global memory space. The `cudaMemcpyToSymbol()` is 
[synchronous](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-sync) with
respect to the host in most cases. Constant memory is cached using a dedicated per-SM constant cache space and is best
used in uniform read operations where each thread in a warp accesses the same memory address.

#### 2.2.5. Texture Memory

Similar to the constant memory, the texture memory is also cached per-SM through read-only cache which supports hardware
filtering such as performing floating-point interpolation as part of the data load process. Contrary to the constant cache
where the accessed data is usually small and read uniformly by the threads in a warp, the read-only cache is more suitable
for the scattered data access on larger data sets. The texture memory is designed to benefit for the 2-dimensional spatial
locality. Therefore, the best performance can be expected from texture memory when the accessed data is 2-dimensional.
Note that depending on the application, the expected performance from texture memory might be lower than that of the global
memory.

#### 2.2.6. Global Memory




### 2.3. Host-Device Memory Management

#### 2.3.1. Pinned Memory

#### 2.3.2. Zero-copy Memory

#### 2.3.3. Unified Memory

## 3. Example: Vector Addition (AXPY)

## 4. Example: Matrix Addition

{% include links.md %}

