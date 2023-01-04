---
jupyter: julia-1.6-t4
title: Parallel processing in Julia
toc-title: Table of contents
---

# Parallel processing in Julia

## 1 Overview

Julia provides built-in support for various kinds of parallel processing
on one or more machines. This material focuses on some standard
approaches that are (mostly) analogous to functionality in Python and R.
However there is other functionality available, including the ability to
control tasks and sending data between processes in a fine-grained way.

In addition to parallelization, the last section discusses some issues
related to efficiency with for loops, in particular *fused* operations.
This is not directly related to parallelization but given the focus on
loops in this document, it's useful and interesting to know about.

## 2 Threading

Threaded calculations are done in parallel on [software
threads](../#31-shared-memory).

Threads share objects in memory with the parent process, which is useful
for avoiding copies but raises the danger of a "race condition", where
different threads modify data that other threads are using and cause
errors..

### 2.1 Threaded linear algebra

As with Python and R, Julia uses BLAS, a standard library of basic
linear algebra operations (written in Fortran or C), for linear algebra
operations. A fast BLAS can greatly speed up linear algebra relative to
the default BLAS on a machine. Julia uses a fast, open source, free BLAS
library called *OpenBLAS*. In addition to being fast when used on a
single core, the openBLAS library is threaded - if your computer has
multiple cores and there are free resources, your linear algebra will
use multiple cores

Here's an example.

::: {.cell execution_count="1"}
``` {.julia .cell-code}
using LinearAlgebra
using Distributions
n = 7000
x = rand(Uniform(0,1), n,n);

println(BLAS.get_num_threads())
```

::: {.cell-output .cell-output-stdout}
    8
:::
:::

::: {.cell execution_count="2"}
``` {.julia .cell-code}
function chol_xtx(x)
    z = x'*x   ## z is positive definite
    C = cholesky(z) 
end

BLAS.set_num_threads(4)
@time chol = chol_xtx(x);  
```

::: {.cell-output .cell-output-stdout}
      4.753901 seconds (2.62 M allocations: 888.768 MiB, 4.58% gc time, 20.26% compilation time)
:::
:::

::: {.cell execution_count="3"}
``` {.julia .cell-code}
BLAS.set_num_threads(1)
@time chol = chol_xtx(x);  
```

::: {.cell-output .cell-output-stdout}
     10.599172 seconds (7 allocations: 747.681 MiB, 0.22% gc time)
:::
:::

We see that using four threads is faster than one, but in this case we
don't get a four-fold speedup.

#### Number of threads

By default, Julia will set the number of threads for linear algebra
equal to the number of processors on your machine.

As seen above, you can check the number of threads being used with:

::: {.cell execution_count="4"}
``` {.julia .cell-code}
BLAS.get_num_threads()
```

::: {.cell-output .cell-output-display execution_count="5"}
    1
:::
:::

Other ways to control the number of threads used for linear algebra
include:

-   setting the `OMP_NUM_THREADS` environment variable in the shell
    before starting Julia, and
-   using `BLAS.set_num_threads(n)`.

### 2.2 Threaded for loops

In Julia, you can directly set up [software threads to use for parallel
processing](https://docs.julialang.org/en/v1/manual/multi-threading).

Here we'll see some examples of running a for loop in parallel, both
acting on a single object and used as a parallel map operation.

Here we can operate on a vector in parallel:

::: {.cell execution_count="5"}
``` {.julia .cell-code}
using Base.Threads

n = 50000000;
x = rand(n);

@threads for i in 1:length(x)
    x[i] = exp(x[i]) + sin(x[i]);
end
```
:::

We could also threads to carry out a parallel map operation, implemented
as a for loop.

::: {.cell execution_count="6"}
``` {.julia .cell-code}
n = 1000

function test(n)
    x = rand(Uniform(0,1), n,n)
    z = x'*x 
    C = cholesky(z)
    return(C.U[1,1])
end

a = zeros(12)
@threads for i in 1:12
    a[i] = test(n)
end
```
:::

### 2.3 Spawning tasks on threads

You can also create (aka 'spawn') individual tasks on threads, with the
tasks running in parallel.

Let's see an example (taken from
[here](http://ferestrepoca.github.io/paradigmas-de-programacion/paralela/tutoriales/julia/notebooks/parallelProgrammingApplications.html)
of sorting a vector in parallel, by sorting subsets of the vector in
separate threads.

::: {.cell execution_count="7"}
``` {.julia .cell-code}
import Base.Threads.@spawn

# sort the elements of `v` in place, from indices `lo` to `hi` inclusive
function psort!(v, lo::Int=1, hi::Int=length(v))
    println(current_task(), ' ', lo, ' ', hi)
    if lo >= hi                       # 1 or 0 elements; nothing to do
        return v
    end
    if hi - lo < 100000               # below some cutoff, run in serial
        sort!(view(v, lo:hi), alg = MergeSort)
        return v
    end

    mid = (lo+hi)>>>1                 # find the midpoint

    ### Set up parallelization here ###

    ## Sort two halves in parallel, one in current call and one in a new task
    ## in a separate thread:
    
    half = @spawn psort!(v, lo, mid)  # task to sort the lower half
    psort!(v, mid+1, hi)              # sort the upper half in the current call
    
    wait(half)                        # wait for the lower half to finish

    temp = v[lo:mid]                  # workspace for merging

    i, k, j = 1, lo, mid+1            # merge the two sorted sub-arrays
    @inbounds while k < j <= hi
        if v[j] < temp[i]
            v[k] = v[j]
            j += 1
        else
            v[k] = temp[i]
            i += 1
        end
        k += 1
    end
    @inbounds while k < j
        v[k] = temp[i]
        k += 1
        i += 1
    end

    return v
end
```

::: {.cell-output .cell-output-display execution_count="8"}
    psort! (generic function with 3 methods)
:::
:::

How does this work? Let's consider an example where we sort a vector of
length 250000.

The vector gets split into elements 1:125000 (run in task #1) and
125001:250000 (run in the main call). Then the elements 1:125000 are
split into 1:62500 (run in task #2) and 62501:125000 (run in task #1),
while the elements 125001:250000 are split into 125001:187500 (run in
task #3) and 187501:250000 (run in the main call). No more splitting
occurs because vectors of length less than 100000 are run in serial.

Assuming we have at least four threads (including the main process),
each of the tasks will run in a separate thread, and all four sorts on
the vector subsets will run in parallel.

::: {.cell execution_count="8"}
``` {.julia .cell-code}
x = rand(250000);
psort!(x);
```

::: {.cell-output .cell-output-stdout}
    Task (runnable) @0x00007f4de04c6cb0 1 250000
    Task (runnable) @0x00007f4de04c6cb0 125001 250000
    Task (runnable) @0x00007f4de04c6cb0 187501 250000
    Task (runnable) @0x00007f4de19f6380 1 125000
    Task (runnable) @0x00007f4de19f6380 62501 125000
    Task (runnable) @0x00007f4e4bb11510 1 62500
    Task (runnable) @0x00007f4de19f64d0 125001 187500
:::
:::

We see that the output from `current_task()` shows that the task labels
correspond with what I stated above.

The number of tasks running in parallel will be at most the number of
threads set in the Julia session.

### 2.4 Controlling the number of threads

You can see the number of threads available:

::: {.cell execution_count="9"}
``` {.julia .cell-code}
Threads.nthreads()
```

::: {.cell-output .cell-output-display execution_count="10"}
    4
:::
:::

You can control the number of threads used for threading in Julia (apart
from linear algebra) either by:

-   setting the `JULIA_NUM_THREADS` environment variable in the shell
    before starting Julia, or
-   starting Julia with the `-t` (or `--threads`) flag, e.g.:
    `julia -t 4`.

Note that we can't use `OMP_NUM_THREADS` as the Julia threading is not
based on openMP.

## 3 Multi-process parallelization

### 3.1 Parallel map operations

We can use `pmap` to run a parallel map operation across multiple Julia
processes (on one or more machines). `pmap` is good for cases where each
task takes a non-negligible amount of time, as there is overhead
(latency) in starting the tasks.

Here we'll carry out multiple computationally expensive calculations in
the map.

We need to import packages and create the function on each of the worker
processes using `@everywhere`.

::: {.cell execution_count="10"}
``` {.julia .cell-code}
using Distributed

if nprocs() == 1
    addprocs(4)
end

nprocs()
```

::: {.cell-output .cell-output-stderr}
    WARNING: using Distributed.@spawn in module Main conflicts with an existing identifier.
:::

::: {.cell-output .cell-output-display execution_count="11"}
    5
:::
:::

::: {.cell execution_count="11"}
``` {.julia .cell-code}
@everywhere begin
    using Distributions
    using LinearAlgebra
    function test(n)
        x = rand(Uniform(0,1), n,n)
        z = transpose(x)*x 
        C = cholesky(z)
        return C.U[2,3]
    end
end

result = pmap(test, repeat([5000],12))
```

::: {.cell-output .cell-output-display execution_count="12"}
    12-element Vector{Float64}:
     11.16226720557071
     12.014264289400314
     11.898111818872662
     11.09311951995371
     11.404196895665349
     12.673542505580757
     11.73011695918563
     11.817326190943426
     11.34666732332691
     10.967735044139209
     11.205400337082677
     11.202193460748065
:::
:::

One can use static allocation (prescheduling) with the `batch_size`
argument, thereby assigning that many tasks to each worker to reduce
latentcy.

### 3.2 Parallel for loops

One can execute for loops in parallel across multiple worker processes
as follows. This is particularly handy for cases where one uses a
reduction operator (e.g., the `+` here) so that little data needs to be
copied back to the main process. (And in this case we don't copy any
data to the workers either.)

Here we'll sum over a large number of random numbers with chunks done on
each of the workers, comparing the time to a basic for loop.

::: {.cell execution_count="12"}
``` {.julia .cell-code}
function forfun(n)
    sum = 0.0
    for i in 1:n
        sum += rand(1)[1]
    end
    return(sum)
end

function pforfun(n)
   out = @sync @distributed (+) for i = 1:n
       rand(1)[1]
   end
   return(out)
end

n=50000000
@time forfun(n);
```

::: {.cell-output .cell-output-stdout}
      4.300862 seconds (50.02 M allocations: 4.472 GiB, 16.62% gc time, 0.24% compilation time)
:::
:::

::: {.cell execution_count="13"}
``` {.julia .cell-code}
@time pforfun(n); 
```

::: {.cell-output .cell-output-stdout}
      1.874745 seconds (1.05 M allocations: 61.659 MiB, 16.38% compilation time)
:::
:::

The use of `@sync` causes the operation to block until the result is
available so we can get the correct timing.

Without a reduction operation, one would generally end up passing a lot
of data back to the main process, and this could take a lot of time. For
such calculations, one would generally be better off using threaded for
loops in order to take advantage of shared memory.

We'd have to look into how the random number seed is set on each worker
to better understand any issues that might arise from parallel random
number generation, but I believe that each worker has a different seed
(but note that this does not explicitly ensure that the random number
streams on the workers are distinct, as is the case if one uses the
L'Ecuyer algorithm).

### 3.3 Passing data to the workers

With multiple workers, particularly on more than one machine, one
generally wants to be careful about having to copy large data objects to
each worker, as that could make up a substantial portion of the time
involved in the computation.

One can explicitly copy a variable to the workers in an `@everywhere`
block by using Julia's interpolation syntax:

::: {.cell execution_count="14"}
``` {.julia .cell-code}
@everywhere begin
    x = $x  # copy to workers using interpolation syntax
    println(pointer_from_objref(x), ' ', x[1])  
end
```

::: {.cell-output .cell-output-stdout}
    Ptr{Nothing} @0x00007f4d34abaa90 6.622218080343245e-6
          From worker 5:    Ptr{Nothing} @0x00007f31fa97c330 6.622218080343245e-6
          From worker 2:    Ptr{Nothing} @0x00007fe477b80330 6.622218080343245e-6
          From worker 3:    Ptr{Nothing} @0x00007f7ef787c330 6.622218080343245e-6
          From worker 4:    Ptr{Nothing} @0x00007fc9e8608330 6.622218080343245e-6
:::
:::

We see based on `pointer_from_objref` that each copy of `x` is stored at
a distinct location in memory, even when processes are on the same
machine.

Also note that if one creates a variable within an `@everywhere` block,
that variable is available to all tasks run on the worker, so it is
global' with respect to those tasks. Note the repeated values in the
result here.

::: {.cell execution_count="15"}
``` {.julia .cell-code}
@everywhere begin
    x = rand(5)
    function test(i)
        return sum(x)
    end
end

result = pmap(test, 1:12, batch_size = 3)
```

::: {.cell-output .cell-output-display execution_count="16"}
    12-element Vector{Float64}:
     1.6357038263726364
     2.524528943958307
     2.80134501297326
     2.1779980896005253
     1.6357038263726364
     2.524528943958307
     2.80134501297326
     2.1779980896005253
     1.6357038263726364
     2.524528943958307
     2.80134501297326
     2.1779980896005253
:::
:::

If one wants to have multiple processes all work on the same object,
without copying it, one can consider using Julia's
[SharedArray](https://docs.julialang.org/en/v1/stdlib/SharedArrays/#SharedArrays.SharedArray)
(one machine) or [DArray from the DistributedArrays
package](https://juliaparallel.org/DistributedArrays.jl/stable/)
(multiple machines) types, which break up arrays into pieces, with
different pieces stored locally on different processes.

### 3.4 Spawning tasks

One can use the `Distributed.@spawnat` macro to run tasks on processes,
in a fashion similar to using `Threads.@spawn`. More details can be
found
[here](https://docs.julialang.org/en/v1/stdlib/Distributed/#Distributed.@spawnat).

### 3.5 Using multiple machines

In addition to using processes on one machine, one can use processes
across multiple machines. One can either start the processes when you
start the main Julia session or you can start them from within the Julia
session. In both cases you'll need to have the ability to ssh to the
other machines without entering your password.

To start the processes when starting Julia, create a "machinefile" that
lists the names of the machines and the number of worker processes to
start on each machine.

Here's an example machinefile:

    arwen.berkeley.edu
    arwen.berkeley.edu
    gandalf.berkeley.edu
    gandalf.berkeley.edu

Note that if you're using Slurm on a Linux cluster, you could generate
that file in the shell from within your Slurm allocation like this:

    srun hostname > machines

Then start Julia like this:

    julia --machine-file machines

From within Julia, you can add processes like this (first we'll remove
the existing worker processes started using `addprocs()` previously):

::: {.cell execution_count="16"}
``` {.julia .cell-code}
rmprocs(workers())

addprocs([("arwen", 2), ("gandalf", 2)])
```

::: {.cell-output .cell-output-display execution_count="17"}
    4-element Vector{Int64}:
     6
     7
     8
     9
:::
:::

To check on the number of processes:

::: {.cell execution_count="17"}
``` {.julia .cell-code}
nprocs()
```

::: {.cell-output .cell-output-display execution_count="18"}
    5
:::
:::

## 4 Loops and fused operations

Consider the following vectorized code that you might run in a variety
of languages (e.g., Julia, Python, R).

    x = exp(x) + 3*sin(x)

If run as vectorized code, this has downsides. First, it will use
additional memory (temporary arrays will be created to store `exp(x)`,
`sin(x)`, `3*sin(x)`). Second, multiple for loops will have to get
executed when the vectorized code is run, looping over the elements of
`x` to calculate `exp(x)`, `sin(x)`, etc. (For example in R or
Python/numpy, multiple for loops would get run in the underlying C
code.)

Contrast that to running directly as a for loop (e.g., in Julia or in
C/C++):

::: {.cell execution_count="18"}
``` {.julia .cell-code}
for i in 1:length(x)
    x[i] = exp(x[i]) + 3*sin(x[i])
end
```
:::

Here temporary arrays don't need to be allocated and there is only a
single for loop.

Combining loops is called 'fusing' and is an [important optimization
that Julia can
do](https://docs.julialang.org/en/v1/manual/performance-tips/#More-dots:-Fuse-vectorized-operations).
(It's also a [key optimization done by
XLA](https://www.tensorflow.org/xla), a compiler used with Tensorflow.)

Of course you might ask why use vectorized code at all given that Julia
will [JIT
compile](https://en.wikipedia.org/wiki/Just-in-time_compilation) the for
loop above and run it really quickly. That's true, but reading and
writing vectorized code is easier than writing for loops.

Let's compare the speed of the following approaches. We'll put
everything into functions as generally [recommended when timing Julia
code](https://www.juliabloggers.com/timing-in-julia) to avoid [global
variables that incur a performance penalty because their type can
change](https://julialang.org/blog/2022/08/julia-1.8-highlights/#typed_globals).

First, let's find the time when directly using a for loop, as a
baseline.

::: {.cell execution_count="19"}
``` {.julia .cell-code}
n = 50000000
y = Array{Float64}(undef, n);
x = rand(n);
```
:::

::: {.cell execution_count="20"}
``` {.julia .cell-code}
function direct_for_loop_calc(x, y)
    for i in 1:length(x)
        y[i] = exp(x[i]) + sin(x[i])
    end
end

using BenchmarkTools
@benchmark direct_for_loop_calc(x, y)
```

::: {.cell-output .cell-output-display execution_count="20"}
    BenchmarkTools.Trial: 6 samples with 1 evaluation.
     Range (min … max):  753.119 ms … 936.340 ms  ┊ GC (min … max): 0.00% … 0.00%
     Time  (median):     833.486 ms               ┊ GC (median):    0.00%
     Time  (mean ± σ):   843.634 ms ±  61.026 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

      █                       █ ██              █                 █  
      █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁██▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
      753 ms           Histogram: frequency by time          936 ms <

     Memory estimate: 0 bytes, allocs estimate: 0.
:::
:::

Notice the lack of additional memory use.

Now let's try a basic vectorized calculation (for which we need the
various periods to get vectorization), without fusing. We'll reassign
the result to the allocated `y` vector for comparability to the for loop
implementation above.

::: {.cell execution_count="21"}
``` {.julia .cell-code}
function basic_vectorized_calc(x, y)
     y .= exp.(x) + 3 * sin.(x)
end

using BenchmarkTools
@benchmark basic_vectorized_calc(x, y)
```

::: {.cell-output .cell-output-display execution_count="21"}
    BenchmarkTools.Trial: 4 samples with 1 evaluation.
     Range (min … max):  1.320 s …   1.434 s  ┊ GC (min … max): 0.27% … 7.37%
     Time  (median):     1.378 s              ┊ GC (median):    5.23%
     Time  (mean ± σ):   1.377 s ± 46.701 ms  ┊ GC (mean ± σ):  4.60% ± 3.01%

      █                          █  █                         █  
      █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
      1.32 s         Histogram: frequency by time        1.43 s <

     Memory estimate: 1.49 GiB, allocs estimate: 8.
:::
:::

The original `x` array is 400 MB; notice the additional memory
allocation and that this takes almost twice as long as the original for
loop.

Here's a fused version of the vectorized calculation, where the `@.`
causes the loops to be fused.

::: {.cell execution_count="22"}
``` {.julia .cell-code}
function fused_vectorized_calc(x, y)
    y .= @. exp(x) + 3 * sin(x)
end

@benchmark fused_vectorized_calc(x, y)
```

::: {.cell-output .cell-output-display execution_count="22"}
    BenchmarkTools.Trial: 7 samples with 1 evaluation.
     Range (min … max):  707.601 ms … 735.076 ms  ┊ GC (min … max): 0.00% … 0.00%
     Time  (median):     725.094 ms               ┊ GC (median):    0.00%
     Time  (mean ± σ):   723.132 ms ±  10.516 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

      █        █                 █          █          █     █    █  
      █▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁█▁▁▁▁█ ▁
      708 ms           Histogram: frequency by time          735 ms <

     Memory estimate: 0 bytes, allocs estimate: 0.
:::
:::

We see that the time and (lack of) memory allocation are essentially the
same as the original basic for loop.

Finally one can achieve the same fusion by having the function just
compute scalar quantities and then using the vectorized version of the
function (by using `scalar_calc.()` instead of `scalar_calc()`), which
also does the fusion.

::: {.cell execution_count="23"}
``` {.julia .cell-code}
function scalar_calc(x)
    return(exp(x) + 3 * sin(x))
end

@benchmark y .= scalar_calc.(x)
```

::: {.cell-output .cell-output-display execution_count="23"}
    BenchmarkTools.Trial: 8 samples with 1 evaluation.
     Range (min … max):  653.546 ms … 672.753 ms  ┊ GC (min … max): 0.00% … 0.00%
     Time  (median):     664.597 ms               ┊ GC (median):    0.00%
     Time  (mean ± σ):   663.533 ms ±   6.299 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

      █         █            █          █ █   █       █           █  
      █▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁█▁█▁▁▁█▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁█ ▁
      654 ms           Histogram: frequency by time          673 ms <

     Memory estimate: 32 bytes, allocs estimate: 2.
:::
:::
