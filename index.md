---
title: Parallel processing in Python, R, MATLAB, and C/C++
layout: default
author: Christopher Paciorek
---

## 1 This tutorial

THIS TUTORIAL IS UNDER CONSTRUCTION. The Python page is still undergoing major changes.

This tutorial covers the use of parallelization (on either one machine or multiple machines/nodes) in Python, R, MATLAB and C/C++. Please click on the links above for material specific to each language.

You should be able to replicate much of what is covered here provided you have the relevant software on your computer, but some of the parallelization approaches may not work on Windows.

This tutorial assumes you have a working knowledge of the relevant language, but not necessarily knowledge of parallelization.

Materials for this tutorial, including the Markdown files and associated code files that were used to create these documents are available on [GitHub](https://github.com/berkeley-scf/tutorial-parallelization) in the `gh-pages` branch.  You can download the files by doing a git clone from a terminal window on a UNIX-like machine, as follows:

```bash
git clone https://github.com/berkeley-scf/tutorial-parallelization
```

This tutorial by Christopher Paciorek of the UC Berkeley Statistical Computing Facility is licensed under a Creative Commons Attribution 3.0 Unported License.

## 2 Some useful terminology

### 2.1 Computer architecture

Everyday personal computers usually have more than one processor (more than one chip) and on a given processor, often have more than one core (multi-core). A multi-core processor has multiple processors on a single computer chip. On personal computers, all the processors and cores share the same memory. For the purpose of this tutorial, there is little practical distinction between multi-processor and multi-core situations. The main issue is whether processes share memory or not. In general, I won't distinguish between cores and processors. We'll just focus on the number of cores on given personal computer or a given node in a cluster. 

Supercomputers and computer clusters generally have tens, hundreds, or thousands of 'nodes', linked by a fast local network. Each node is essentially a computer with its own cores and memory. Memory is local to each node (distributed memory). One basic principle is that communication between a processor and its memory is much faster than communication across nodes between processors accessing different memory. 

### 2.2 Glossary of terms

  - *cores*: We'll use this term to mean the different processing
units available on a single machine or node.
  - *nodes*: We'll use this term to mean the different computers,
each with their own distinct memory, that make up a cluster or supercomputer.
  - *processes*: instances of a program executing on a machine; multiple
processes may be executing at once. A given executable (e.g., Python or R) may start up multiple
processes at once. Ideally we have no more user processes than cores on
a node.
  - *workers*: the individual processes that are carrying out the (parallelized) computation. We'll use worker and process interchangeably. 
  - *tasks*: This term gets used in various ways (including in place of 'processes' in the context of Slurm and MPI), but we'll use it to refer to the individual computational items you want to complete - e.g., one task per cross-validation fold or one task per simulation replicate/iteration.
  - *threads*: multiple paths of execution within a single process;
the OS sees the threads as a single process, but one can think of
them as 'lightweight' processes. Ideally when considering the processes
and their threads, we would have the number of total threads across
all processes not exceed the number of cores on a node.
  - *forking*: child processes are spawned that are identical to
the parent, but with different process IDs and their own memory. In some cases if objects are not changed, the objects in the child process may refer back to the original objects in the original process, avoiding making copies.
  - *sockets*: some of R's parallel functionality involves creating
new R processes (e.g., starting processes via *Rscript*) and
communicating with them via a communication technology called sockets.
  - scheduler: a program that manages users' jobs on a cluster. 
  - load-balanced: when all the cores that are part of a computation are busy for the entire period of time the computation is running.
  
 
## 3 Types of parallel processing

There are two basic flavors of parallel processing (leaving aside
GPUs): shared memory (single machine) and distributed memory (multiple machines). With shared memory, multiple
processors (which I'll call cores) share the same memory. With distributed
memory, you have multiple nodes, each with their own memory. You can
think of each node as a separate computer connected by a fast network. 


### 3.1 Shared memory

For shared memory parallelism, each core is accessing the same memory
so there is no need to pass information (in the form of messages)
between different machines. 

However, except for certain special situations (involving software threads or forked processes), the different worker processes on a given machine do not share objects in memory. So most often, one has multiple copies of the same objects, one per worker process.

#### Threading

Threads are multiple paths of execution within a single process. If you are monitoring CPU
usage (such as with *top* in Linux or Mac) and watching a job that is executing threaded code, you'll
see the process using more than 100% of CPU. When this occurs, the
process is using multiple cores, although it appears as a single process
rather than as multiple processes.

Note that this is a different notion than a processor that is hyperthreaded. With hyperthreading a single core appears as two cores to the operating system.

Threads generally do share objects in memory, thereby allowing us to have a single copy of objects instead of one per thread. 

One very common use of threading is for linear algebra, using threaded linear alebra packages accessed from Python, R, MATLAB, or C/C++.

### 3.2 Distributed memory

Parallel programming for distributed memory parallelism requires passing
messages containing information (code, data, etc.) between the different nodes. 

A standard protocol for passing messages is MPI, of which there are various versions, including *openMPI*.

Tools such as ipyparallel, Dask and Ray in Python and R's future package all manage the work of moving information between nodes for you (and don't generally use MPI). 

### 3.3 Other type of parallel processing

We won't cover either of these in this tutorial.

#### GPUs

GPUs (Graphics Processing Units) are processing units originally designed
for rendering graphics on a computer quickly. This is done by having
a large number of simple processing units for massively parallel calculation.
The idea of general purpose GPU (GPGPU) computing is to exploit this
capability for general computation. 

Most researchers don't program for a GPU directly but rather use software (often machine learning software such as Tensorflow or PyTorch) that has been programmed to take advantage of a GPU if one is available.

#### Spark and Hadoop

Spark and Hadoop are systems for implementing computations in a distributed
memory environment, using the MapReduce approach.

Note that Dask provides a lot of the same functionality as Spark, allowing one to create distributed datasets where pieces of the dataset live on different machines but can be treated as a single dataset from the perspective of the user.

# 4 Parallelization strategies

Some of the considerations that apply when thinking about how effective a given parallelization approach will be include:

  - the amount of memory that will be used by the various processes,
  - the amount of communication that needs to happen – how much data will need to be passed between processes,
  - the latency of any communication - how much delay/lag is there in sending data between processes or starting up a worker process, and
  - to what extent do processes have to wait for other processes to finish before they can do their next step.

The following are some basic principles/suggestions for how to parallelize your computation.

  -  Should I use one machine/node or many machines/nodes?
    -  If you can do your computation on the cores of a single node using shared memory, that will be faster than using the same number of cores (or even somewhat more cores) across multiple nodes. Similarly, jobs with a lot of data/high memory requirements that one might think of as requiring Spark or Hadoop may in some cases be much faster if you can find a single machine with a lot of memory. 
    - That said, if you would run out of memory on a single node, then you'll need to use distributed memory.

  - What level or dimension should I parallelize over?
    - If you have nested loops, you generally only want to parallelize at one level of the code. That said, in this unit we'll see some tools for parallelizing at multiple levels. Keep in mind whether your linear algebra is being threaded. Often you will want to parallelize over a loop and not use threaded linear algebra within the iterations of the loop. 
    - Often it makes sense to parallelize the outer loop when you have nested loops. 
    - You generally want to parallelize in such a way that your code is load-balanced and does not involve too much communication. 

  - How do I balance communication overhead with keeping my cores busy?
    - If you have very few tasks, particularly if the tasks take different amounts of time, often some processors will be idle and your code poorly load-balanced. 
    - If you have very many tasks and each one takes little time, the overhead of starting and stopping the tasks will reduce efficiency.

  - Should multiple tasks be pre-assigned (statically assigned) to a process (i.e., a worker) (sometimes called prescheduling) or should tasks be assigned dynamically as previous tasks finish? 
    - To illustrate the difference, suppose you have 6 tasks and 3 workers. If the tasks are pre-assigned, worker 1 might be assigned tasks 1 and 4 at the start, worker 2 assigned tasks 2 and 5, and worker 3 assigned tasks 3 and 6. If the tasks are dynamically assigned, worker 1 would be assigned task 1, worker 2 task 2, and worker 3 task 3. Then whichever worker finishes their task first (it woudn't necessarily be worker 1) would be assigned task 4 and so on. 
    - Basically if you have many tasks that each take similar time, you want to preschedule the tasks to reduce communication. If you have few tasks or tasks with highly variable completion times, you don't want to preschedule, to improve load-balancing. 
    - For R in particular, some of R's parallel functions allow you to say whether the tasks should be prescheduled. In the future package, `future_lapply` has arguments `future.scheduling` and `future.chunk.size`. Similarly, there is the `mc.preschedule` argument in `mclapply()`. 
