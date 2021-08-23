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
- "APOD application design cycle"
---

- [1. Overview](#1-overview)
- [2. NVIDIA Nsight Systems](#2-nvidia-nsight-systems)
  - [2.1. Command Line Interface Profiler](#21-command-line-interface-profiler)
    - [2.1.1. CUDA API Statistics](#211-cuda-api-statistics)
    - [2.1.2. CUDA Kernel Statistics](#212-cuda-kernel-statistics)
    - [2.1.3. CUDA Memory Operations Statistics](#213-cuda-memory-operations-statistics)
    - [2.1.4. Operating System Runtime API Statistics](#214-operating-system-runtime-api-statistics)
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
the Nsight Systems profiler. For sample outputs from using the `--stats` option with CLI profiler in various OSs, see
the [Nsight Systems' documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-output-stats-option).

The CLI gathers the results in an intermediate *.qdstrm* file which needs to be further processed either by importing it in a GUI 
or using the standalone Qdstrm Importer in order to generate an optimized *.qdrep* report file. For portability reasons and future
analysis of the reports on the same or different host machine and sharing the results, the .qdrep formant should be used.

> ## Note:
> In order to import a .qdstrm file in a GUI, the host GUI and CLI version must match. The host GUI is only backwards compatible with
> .qdrep files.
{: .discussion}

At the time of writing this tutorial, Nsight Systems attempts to convert the intermediate report files to their .qdrep report 
counterparts with the same names after finishing the profiling run if the necessary set of required libraries are available.
See [Nsight Systems' documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#importing-qdstrm-files) for further
details. It is important to note that setting the `--stats` option to `True` results in the creation of a SQLite database after the 
collection of results by Nsight Systems. If a large amount of data is captured, creating the corresponding database(s) may require longer 
than normal time periods to complete. For safety reasons, Nsight Systems does not rewrite the results on the same output files by default.
If intended otherwise, users can adopt the `-f` or `--force-overwrite=true` options to overwrite the (.qdstrm, .qdrep, and .sqlite) result files.

Let us run this command on the [vector sum example](https://github.com/MolSSI-Education/gpu_programming_beginner/tree/gh-pages/src/
gpu_vector_sum/v2_cudaCode) from our beginner level GPU workshop. In order to be able to profile a program, an executable application file
is required. Let's compile our example code and run it

~~~
$ nvcc gpuVectorSum.cu cCode.c cudaCode.cu -o vecSum --run
~~~
{: .language-bash}

The aforementioned command gives the following results

~~~
Kicking off ./vecSum

GPU device GeForce GTX 1650 with index (0) is set!

Vector size: 16777216 floats (64 MB)

Elapsed time for dataInitializer: 0.757924 second(s)
Elapsed time for arraySumOnHost: 0.062957 second(s)
Elapsed time for arraySumOnDevice <<< 16384, 1024 >>>: 0.001890 second(s)

Arrays are equal.
~~~
{: .output}

The `--run` flag runs the resulting executable (here, `vecSum`) after compilation. Now, we have a program executable
ready to be profiled

~~~
$ nsys profile --stats=true vecSum
~~~
{: .language-bash}

The output of a `nsys profile --stats=true <application>` commands has three main sections: 1) the application output(s), if any; 2)
summary of processing reports along with their temporary storage destination folders; and 3) profiling statistics.
The first section is exactly the same as the program output given above. The second part of the profiler output yields processing details
about the .qdstrm, .qdrep, and .sqlite report files and their temporary storage folders.

~~~
Processing events...
Capturing symbol files...
Saving temporary "/tmp/nsys-report-2d75-dd10-9860-d729.qdstrm" file to disk...
Creating final output files...

Processing [==============================================================100%]
Saved report file to "/tmp/nsys-report-2d75-dd10-9860-d729.qdrep"
Exporting 1424 events: [==================================================100%]

Exported successfully to
/tmp/nsys-report-2d75-dd10-9860-d729.sqlite
~~~
{: .output}

The profiling statistics is the last part of our `nsys profile --stats=true` output, which consists of four separate
sections:

- **CUDA API Statistics**
- **CUDA Kernel Statistics**
- **CUDA Memory Operations Statistics** (in terms of time or size)
- **Operating System Runtime API Statistics**

Let us overview each section one by one and see what types of information is available to us for performance analysis without
going though the details of each entry and analyzing the numbers.

#### 2.1.1. CUDA API Statistics

[CUDA API statistics report tables](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cudaapisum) have seven columns,
in which all timings are in nanoseconds (ns):

1. **Time (%)**: The percentage of the **Total Time** for all calls to the function listed in the **Name** column
2. **Total Time (ns)**: The total execution time of all calls to the function listed in the **Name** column
3. **Num Calls**: The number of calls to the function listed in the **Name** column
4. **Average**: The average execution time of the function listed in the **Name** column
5. **Minimum**: The smallest execution time among the current set of function calls to the function listed in the **Name** column
6. **Maximum**: The largest execution time among the current set of function calls to the function listed in the **Name** column
7. **Name**: The name of the function being profiled

~~~
CUDA API Statistics:

 Time(%)  Total Time (ns)  Num Calls     Average      Minimum      Maximum            Name         
 -------  ---------------  ---------  -------------  ----------  -----------  ---------------------
    87.2      415,182,743          3  138,394,247.7     519,539  414,129,668  cudaMalloc           
    12.3       58,403,179          4   14,600,794.8  12,721,839   17,162,836  cudaMemcpy           
     0.4        1,909,047          1    1,909,047.0   1,909,047    1,909,047  cudaDeviceSynchronize
     0.1          613,510          3      204,503.3     160,057      280,734  cudaFree             
     0.0           41,222          1       41,222.0      41,222       41,222  cudaLaunchKernel 
~~~
{: .output}

#### 2.1.2. CUDA Kernel Statistics

[CUDA Kernel statistics report summary](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpukernsum) has seven columns,
in which all timings are in nanoseconds (ns):

1. **Time (%)**: The percentage of the **Total Time** for all kernel executions listed in the **Name** column
2. **Total Time (ns)**: The total execution time of all kernel launches listed in the **Name** column
3. **Instances**: The number of kernel launches listed in the **Name** column
4. **Average**: The average execution time of the kernel listed in the **Name** column
5. **Minimum**: The smallest execution time among the current set of kernel launches listed in the **Name** column
6. **Maximum**: The largest execution time among the current set of kernel launches listed in the **Name** column
7. **Name**: The name of the GPU kernels being profiled

~~~
CUDA Kernel Statistics:

 Time(%)  Total Time (ns)  Instances    Average     Minimum    Maximum                       Name                     
 -------  ---------------  ---------  -----------  ---------  ---------  ---------------------------------------------
   100.0        1,781,674          1  1,781,674.0  1,781,674  1,781,674  arraySumOnDevice(float*, float*, float*, int)
~~~
{: .output}

#### 2.1.3. CUDA Memory Operations Statistics

CUDA Memory Operations reports are tabulated by [time](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpumemtimesum)
(in ns) involving seven columns:

1. **Time (%)**: The percentage of the **Total Time** for all memory operations listed in the **Operation** column
2. **Total Time (ns)**: The total execution time of all memory operations listed in the **Operation** column
3. **Operations**:The number of times the memory operations listed in the **Operation** column have been executed
4. **Average**: The average memory size used for executing the **Operation**(s)
5. **Minimum**: The smallest execution time among the current set of **Operation**(s)
6. **Maximum**: The largest execution time among the current set of **Operation**(s)
7. **Operation**: The name of the memory operation being profiled

~~~
CUDA Memory Operation Statistics (by time):

 Time(%)  Total Time (ns)  Operations    Average      Minimum     Maximum        Operation     
 -------  ---------------  ----------  ------------  ----------  ----------  ------------------
    77.2       44,485,334           3  14,828,444.7  12,579,141  16,962,494  [CUDA memcpy HtoD]
    22.8       13,110,889           1  13,110,889.0  13,110,889  13,110,889  [CUDA memcpy DtoH]
~~~
{: .output}

or by [size](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpumemsizesum) (in kB) consisting of
six columns:

1. **Total**: The total amount of GPU memory used for the memory operations listed in the **Operation** column
2. **Operations**: The number of times the memory operations listed in the **Operation** column have been executed
3. **Average**: The average execution time of the **Operation**(s)
4. **Minimum**: The minimum amount of memory used among the current set of memory operations **Operation**(s) executed
5. **Maximum**: The maximum amount of memory used among the current set of memory operations **Operation**(s) executed
6. **Operation**: The name of the memory operation being profiled

~~~
CUDA Memory Operation Statistics (by size in KiB):

    Total     Operations   Average     Minimum     Maximum        Operation     
 -----------  ----------  ----------  ----------  ----------  ------------------
  65,536.000           1  65,536.000  65,536.000  65,536.000  [CUDA memcpy DtoH]
 196,608.000           3  65,536.000  65,536.000  65,536.000  [CUDA memcpy HtoD]
~~~
{: .output}

#### 2.1.4. Operating System Runtime API Statistics

[The OS Runtime API report table](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#osrtsum) has seven columns,
in which all timings are in nanoseconds (ns):

1. **Time (%)**: The percentage of the **Total Time** for all calls to the function listed in the **Name** column
2. **Total Time (ns)**: The total execution time of all calls to the function listed in the **Name** column
3. **Num Calls**: The number of calls to the function listed in the **Name** column
4. **Average**: The average execution time of the function listed in the **Name** column
5. **Minimum**: The smallest execution time among the current set of function calls to the function listed in the **Name** column
6. **Maximum**: The largest execution time among the current set of function calls to the function listed in the **Name** column
7. **Name**: The name of the function being profiled

~~~
Operating System Runtime API Statistics:

 Time(%)  Total Time (ns)  Num Calls    Average     Minimum    Maximum         Name     
 -------  ---------------  ---------  ------------  -------  -----------  --------------
    85.5      501,135,238         16  31,320,952.4   19,438  100,185,161  poll          
    13.5       78,934,009        674     117,112.8    1,191   11,830,047  ioctl         
     0.5        2,810,648         87      32,306.3    1,724      838,818  mmap          
     0.2          899,070         82      10,964.3    4,605       27,860  open64        
     0.1          731,813         10      73,181.3   15,345      325,547  sem_timedwait 
     0.1          412,876         28      14,745.6    1,414      272,205  fopen         
     0.0          291,176          5      58,235.2   27,768      110,522  pthread_create
     0.0          196,169          3      65,389.7   62,525       67,914  fgets         
     0.0           82,791          4      20,697.8    3,946       56,969  fgetc         
     0.0           59,983         10       5,998.3    3,172       14,405  munmap        
     0.0           52,364         22       2,380.2    1,201        8,422  fclose        
     0.0           47,041         11       4,276.5    2,333        6,156  write         
     0.0           36,940          6       6,156.7    4,222        8,393  fread         
     0.0           30,565          5       6,113.0    4,716        7,174  open          
     0.0           26,866         12       2,238.8    1,025        7,491  fcntl         
     0.0           24,537         13       1,887.5    1,315        2,772  read          
     0.0            9,755          2       4,877.5    3,702        6,053  socket        
     0.0            7,798          1       7,798.0    7,798        7,798  connect       
     0.0            7,306          1       7,306.0    7,306        7,306  pipe2         
     0.0            2,033          1       2,033.0    2,033        2,033  bind          
     0.0            1,315          1       1,315.0    1,315        1,315  listen        
~~~
{: .output}

So far, we have demonstrated that the nsys CLI profiler provides a comprehensive report on statistics of CUDA Runtime 
APIs, GPU kernel executions, CUDA Memory Operations, and OS Runtime API calls. These reports provide useful information
about the performance of the application and offer a great tool for adopting the APOD cycle for both analysis and
performance optimization. In addition to the CLI profiler, NVIDIA offers profiling tools using GUIs. These are convenient 
ways to analyze profiling reports or compare performance results from different profiling runs on the same application.
In the following sections, we overview the main aspects of the NVIDIA Nsight Systems' GUI profiler.

### 2.2. Graphical User Interface Profiler



## 3. NVIDIA Nsight Compute

### 3.1. Command Line Interface Profiler

### 3.2. Graphical User Interface Profiler

## 4. Example: Vector Addition (AXPY)

## 5. Example: Matrix Addition

{% include links.md %}