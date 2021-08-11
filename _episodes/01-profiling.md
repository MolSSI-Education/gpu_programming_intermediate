---
title: "Profiling with NVIDIA Nsight"
teaching: 60
exercises: 0
questions:
- "What is profiling? Why and how is it useful for parallelization?"
- "What are NVIDIA Nsight Systems and Nsight Compute? What do they do and how can I use them?"
objectives:
- "Mastering best practices in profiling-driven approach in CUDA C/C++ programming"
keypoints:
- "NVIDIA Nsight Systems"
- "Profiling-driven CUDA C/C++ programming"
- "APOC application design cycle"
---

- [1. Overview](#1-overview)
- [2. NVIDIA Nsight Systems](#2-nvidia-nsight-systems)
  - [2.1. Command Line Interface Profiler](#21-command-line-interface-profiler)
  - [2.2. Graphical User Interface Profiler](#22-graphical-user-interface-profiler)
- [3. NVIDIA Nsight Compute](#3-nvidia-nsight-compute)
  - [3.1. Command Line Interface Profiler](#31-command-line-interface-profiler)
  - [3.2. Graphical User Interface Profiler](#32-graphical-user-interface-profiler)
- [4. Example: Vector Addition (AXPY)](#4-example-vector-addition-axpy)
- [5. Example: Matrix Addition](#5-example-matrix-addition)

## 1. Overview

The present tutorial is a continuation of [MolSSI's Fundamentals of Heterogeneous Parallel Programming 
with CUDA C/C++ at the beginner level](http://education.molssi.org/gpu_programming_beginner) where we
provide a deeper look into the close relationship between the GPU architecture and the application performance.
Adopting a systematic approach to leverage this relationship for writing more efficient programs,
we have based our approach on NVIDIA's [Best Practices  Guide for CUDA C/C++](https://docs.nvidia.com/
cuda/cuda-c-best-practices-guide/index.html). These best practices guidelines encourage users to follow the **Asses**, 
**Parallelize**, **Optimize** and **Deploy** ([**APOD**](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
index.html#assess-parallelize-optimize-deploy)) application design cycle for an efficient and rapid recognition
of the parallelization opportunities in programs and improving the code quality and performance. 

The NVIDIA Nsight family consists of three members:

- **Nsight Systems**: A comprehensive system-wide tool for the application performance analysis and optimization
- **Nsight Compute**: A professional tool for kernel-level performance analysis and debugging
- **Nsight Graphics**: An optimization and debugging software for graphical workflows such as rendering performance *etc.*

NVIDIA recommends developers to start the profiling process by using **Nsight Systems** in order to identify the most important and impactful
system-wide opportunities for optimization and performance improvement. Further optimizations and fine-tunings at the CUDA kernel and API
level can be performed through **Nsight Compute**. In our analysis for performance improvement and code optimization, we will adopt a 
quantitative profile-driven approach and make an extensive use of profiling tools provided by NVIDIA's Nsight Systems  and Nsight Compute.

## 2. NVIDIA Nsight Systems

NVIDIA Nsight Systems is a system-wide performance analysis tool and sampling profiler with tracing feature which allows users to collect and 
process CPU-GPU performance statistics. NVIDIA Nsight Systems recognizes three main activities: 1) profiling, 2) Sampling, and 3) tracing.
The performance data collection is called profiling. In order to collect information on the timings spent on function calls during 
the program, the profiler periodically stops the application under investigation (profilee) to collect information on call stacks 
of active threads (backtraces). The sampling results are generally less precise when the number of samples are small. Tracing refers
to the collection of precise quantitative information about a variety of activities that might be happening in profilee or the OS.
The Nsight Systems collects the information in a profiling session which usually involves both sampling and tracing activities.

NVIDIA Nsight Systems offers two major interfaces, through which users can profile an application:

- Command-Line Interface (CLI)
- Graphical User Interface (GUI)

In the following sections, we overview the mechanics of using each method in details.

### 2.1. Command Line Interface Profiler

The general form of the Nsight Systems command line interface (CLI) profiler, **nsys**, is similar to that of 
nvprof we saw in [MolSSI's Fundamentals of Heterogeneous Parallel Programming with CUDA C/C++ at the beginner 
level](http://education.molssi.org/gpu_programming_beginner) 

~~~
$ nsys [command_switch] [optional command_switch_options] <application> [optional application_options]
~~~
{: .language-bash}

A list of possible values for `command_switch` and optional `command_switch_options` are provided in [Nsight Systems s 
User Manual](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-options). The `<application>` refers to the name of
the profilee executable. 

Despite a rather complicated form mentioned above, the following command will be sufficient for the majority of our applications
in this tutorial

~~~
$ nsys profile --stats=true <application>
~~~
{: .language-bash}

Here, the `--stats=true` option triggers the post processing and generation of the statistical data summary collected by
the Nsight Systems profiler. 

### 2.2. Graphical User Interface Profiler

## 3. NVIDIA Nsight Compute

### 3.1. Command Line Interface Profiler

### 3.2. Graphical User Interface Profiler

## 4. Example: Vector Addition (AXPY)

## 5. Example: Matrix Addition

{% include links.md %}