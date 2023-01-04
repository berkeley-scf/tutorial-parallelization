Parallel processing in Julia
================

## 1 Overview

Julia provides built-in support for various kinds of parallel processing
on one or more machines. This material focuses on some standard
approaches that are (mostly) analogous to functionality in Python and R.
However there is other functionality available, including the ability to
control tasks and sending data between processes in a fine-grained way.

In addition to parallelization, the last section discusses some issues
related to efficiency with for loops, in particular *fused* operations.
This is not directly related to parallelization but given the focus on
loops in this document, it’s useful and interesting to know about.

## 2 Threading

Threaded calculations are done in parallel on [software
threads](./#31-shared-memory).

Threads share objects in memory with the parent process, which is useful
for avoiding copies but raises the danger of a “race condition”, where
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

Here’s an example.

``` julia
using LinearAlgebra
using Distributions
n = 7000
x = rand(Uniform(0,1), n,n);

println(BLAS.get_num_threads())
```

    8

``` julia
function chol_xtx(x)
    z = x'*x   ## z is positive definite
    C = cholesky(z) 
end

BLAS.set_num_threads(4)
@time chol = chol_xtx(x);  
```

      4.703782 seconds (2.62 M allocations: 888.768 MiB, 4.72% gc time, 20.43% compilation time)

``` julia
BLAS.set_num_threads(1)
@time chol = chol_xtx(x);  
```

     10.630900 seconds (7 allocations: 747.681 MiB, 0.22% gc time)

We see that using four threads is faster than one, but in this case we
don’t get a four-fold speedup.

#### Number of threads

By default, Julia will set the number of threads for linear algebra
equal to the number of processors on your machine.

As seen above, you can check the number of threads being used with:

``` julia
BLAS.get_num_threads()
```

    1

Other ways to control the number of threads used for linear algebra
include:

-   setting the `OMP_NUM_THREADS` environment variable in the shell
    before starting Julia, and
-   using `BLAS.set_num_threads(n)`.

### 2.2 Threaded for loops

In Julia, you can directly set up [software threads to use for parallel
processing](https://docs.julialang.org/en/v1/manual/multi-threading).

Here we’ll see some examples of running a for loop in parallel, both
acting on a single object and used as a parallel map operation.

Here we can operate on a vector in parallel:

``` julia
using Base.Threads

n = 50000000;
x = rand(n);

@threads for i in 1:length(x)
    x[i] = exp(x[i]) + sin(x[i]);
end
```

We could also threads to carry out a parallel map operation, implemented
as a for loop.

``` julia
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

### 2.3 Spawning tasks on threads

You can also create (aka ‘spawn’) individual tasks on threads, with the
tasks running in parallel.

Let’s see an example (taken from
[here](http://ferestrepoca.github.io/paradigmas-de-programacion/paralela/tutoriales/julia/notebooks/parallelProgrammingApplications.html)
of sorting a vector in parallel, by sorting subsets of the vector in
separate threads.

``` julia
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

    psort! (generic function with 3 methods)

How does this work? Let’s consider an example where we sort a vector of
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

``` julia
x = rand(250000);
psort!(x);
```

    Task (runnable) @0x00007f06e44c6cb0 1 250000
    Task (runnable) @0x00007f06e44c6cb0 125001 250000
    Task (runnable) @0x00007f06e44c6cb0 187501 250000
    Task (runnable) @0x00007f06e5b4d660 1 125000
    Task (runnable) @0x00007f06e5b4d7b0 125001 187500
    Task (runnable) @0x00007f06e5b4d660 62501 125000
    Task (runnable) @0x00007f074c975510 1 62500

We see that the output from `current_task()` shows that the task labels
correspond with what I stated above.

The number of tasks running in parallel will be at most the number of
threads set in the Julia session.

### 2.4 Controlling the number of threads

You can see the number of threads available:

``` julia
Threads.nthreads()
```

    4

You can control the number of threads used for threading in Julia (apart
from linear algebra) either by:

-   setting the `JULIA_NUM_THREADS` environment variable in the shell
    before starting Julia, or
-   starting Julia with the `-t` (or `--threads`) flag, e.g.:
    `julia -t 4`.

Note that we can’t use `OMP_NUM_THREADS` as the Julia threading is not
based on openMP.

## 3 Multi-process parallelization

### 3.1 Parallel map operations

We can use `pmap` to run a parallel map operation across multiple Julia
processes (on one or more machines). `pmap` is good for cases where each
task takes a non-negligible amount of time, as there is overhead
(latency) in starting the tasks.

Here we’ll carry out multiple computationally expensive calculations in
the map.

We need to import packages and create the function on each of the worker
processes using `@everywhere`.

``` julia
using Distributed

if nprocs() == 1
    addprocs(4)
end

nprocs()
```

    WARNING: using Distributed.@spawn in module Main conflicts with an existing identifier.

    5

``` julia
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

    12-element Vector{Float64}:
     11.393582444472948
     11.105010289847105
     11.775490377807094
     10.84171509813776
     11.58028782674828
     11.89511643725844
     11.665171194143703
     12.00016555638367
     12.30999760747987
     11.544900828541468
     11.298421536527481
     11.675410441519608

One can use [static allocation
(prescheduling)](./#4-parallelization-strategies) with the `batch_size`
argument, thereby assigning that many tasks to each worker to reduce
latentcy.

### 3.2 Parallel for loops

One can execute for loops in parallel across multiple worker processes
as follows. This is particularly handy for cases where one uses a
reduction operator (e.g., the `+` here) so that little data needs to be
copied back to the main process. (And in this case we don’t copy any
data to the workers either.)

Here we’ll sum over a large number of random numbers with chunks done on
each of the workers, comparing the time to a basic for loop.

``` julia
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

      4.246839 seconds (50.02 M allocations: 4.472 GiB, 16.68% gc time, 0.22% compilation time)

``` julia
@time pforfun(n); 
```

      1.791803 seconds (1.05 M allocations: 61.737 MiB, 16.39% compilation time)

The use of `@sync` causes the operation to block until the result is
available so we can get the correct timing.

Without a reduction operation, one would generally end up passing a lot
of data back to the main process, and this could take a lot of time. For
such calculations, one would generally be better off using threaded for
loops in order to take advantage of shared memory.

We’d have to look into how the random number seed is set on each worker
to better understand any issues that might arise from parallel random
number generation, but I believe that each worker has a different seed
(but note that this does not explicitly ensure that the random number
streams on the workers are distinct, as is the case if one uses the
L’Ecuyer algorithm).

### 3.3 Passing data to the workers

With multiple workers, particularly on more than one machine, one
generally wants to be careful about having to copy large data objects to
each worker, as that could make up a substantial portion of the time
involved in the computation.

One can explicitly copy a variable to the workers in an `@everywhere`
block by using Julia’s interpolation syntax:

``` julia
@everywhere begin
    x = $x  # copy to workers using interpolation syntax
    println(pointer_from_objref(x), ' ', x[1])  
end
```

    Ptr{Nothing} @0x00007f063cc6e950 4.109996736056942e-6
          From worker 2:    Ptr{Nothing} @0x00007fdbf83f4330 4.109996736056942e-6
          From worker 5:    Ptr{Nothing} @0x00007feb3afa4330 4.109996736056942e-6
          From worker 4:    Ptr{Nothing} @0x00007f75878e4330 4.109996736056942e-6
          From worker 3:    Ptr{Nothing} @0x00007fdb884d8330 4.109996736056942e-6

We see based on `pointer_from_objref` that each copy of `x` is stored at
a distinct location in memory, even when processes are on the same
machine.

Also note that if one creates a variable within an `@everywhere` block,
that variable is available to all tasks run on the worker, so it is
global’ with respect to those tasks. Note the repeated values in the
result here.

``` julia
@everywhere begin
    x = rand(5)
    function test(i)
        return sum(x)
    end
end

result = pmap(test, 1:12, batch_size = 3)
```

    12-element Vector{Float64}:
     2.24632549484403
     2.9738803127128417
     1.6320032052901892
     1.809832545041849
     2.24632549484403
     2.9738803127128417
     1.6320032052901892
     1.809832545041849
     2.24632549484403
     2.9738803127128417
     1.6320032052901892
     1.809832545041849

If one wants to have multiple processes all work on the same object,
without copying it, one can consider using Julia’s
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
session. In both cases you’ll need to have the ability to ssh to the
other machines without entering your password.

To start the processes when starting Julia, create a “machinefile” that
lists the names of the machines and the number of worker processes to
start on each machine.

Here’s an example machinefile:

    arwen.berkeley.edu
    arwen.berkeley.edu
    gandalf.berkeley.edu
    gandalf.berkeley.edu

Note that if you’re using Slurm on a Linux cluster, you could generate
that file in the shell from within your Slurm allocation like this:

    srun hostname > machines

Then start Julia like this:

    julia --machine-file machines

From within Julia, you can add processes like this (first we’ll remove
the existing worker processes started using `addprocs()` previously):

``` julia
rmprocs(workers())

addprocs([("arwen", 2), ("gandalf", 2)])
```

    4-element Vector{Int64}:
     6
     7
     8
     9

To check on the number of processes:

``` julia
nprocs()
```

    5

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

``` julia
for i in 1:length(x)
    x[i] = exp(x[i]) + 3*sin(x[i])
end
```

Here temporary arrays don’t need to be allocated and there is only a
single for loop.

Combining loops is called ‘fusing’ and is an [important optimization
that Julia can
do](https://docs.julialang.org/en/v1/manual/performance-tips/#More-dots:-Fuse-vectorized-operations).
(It’s also a [key optimization done by
XLA](https://www.tensorflow.org/xla), a compiler used with Tensorflow.)

Of course you might ask why use vectorized code at all given that Julia
will [JIT
compile](https://en.wikipedia.org/wiki/Just-in-time_compilation) the for
loop above and run it really quickly. That’s true, but reading and
writing vectorized code is easier than writing for loops.

Let’s compare the speed of the following approaches. We’ll put
everything into functions as generally [recommended when timing Julia
code](https://www.juliabloggers.com/timing-in-julia) to avoid [global
variables that incur a performance penalty because their type can
change](https://julialang.org/blog/2022/08/julia-1.8-highlights/#typed_globals).

First, let’s find the time when directly using a for loop, as a
baseline.

``` julia
n = 50000000
y = Array{Float64}(undef, n);
x = rand(n);
```

``` julia
function direct_for_loop_calc(x, y)
    for i in 1:length(x)
        y[i] = exp(x[i]) + sin(x[i])
    end
end

using BenchmarkTools
@benchmark direct_for_loop_calc(x, y)
```

    BenchmarkTools.Trial: 8 samples with 1 evaluation.
     Range (min … max):  660.864 ms … 680.575 ms  ┊ GC (min … max): 0.00% … 0.00%
     Time  (median):     664.081 ms               ┊ GC (median):    0.00%
     Time  (mean ± σ):   665.756 ms ±   6.216 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

      ▁ ▁      █▁ ▁  ▁                                            ▁  
      █▁█▁▁▁▁▁▁██▁█▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
      661 ms           Histogram: frequency by time          681 ms <

     Memory estimate: 0 bytes, allocs estimate: 0.

Notice the lack of additional memory use.

Now let’s try a basic vectorized calculation (for which we need the
various periods to get vectorization), without fusing. We’ll reassign
the result to the allocated `y` vector for comparability to the for loop
implementation above.

``` julia
function basic_vectorized_calc(x, y)
     y .= exp.(x) + 3 * sin.(x)
end

using BenchmarkTools
@benchmark basic_vectorized_calc(x, y)
```

    BenchmarkTools.Trial: 4 samples with 1 evaluation.
     Range (min … max):  1.294 s …   1.395 s  ┊ GC (min … max): 0.23% … 7.37%
     Time  (median):     1.354 s              ┊ GC (median):    5.31%
     Time  (mean ± σ):   1.349 s ± 42.069 ms  ┊ GC (mean ± σ):  4.62% ± 3.04%

      █                             █     █                   █  
      █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
      1.29 s         Histogram: frequency by time         1.4 s <

     Memory estimate: 1.49 GiB, allocs estimate: 8.

The original `x` array is 400 MB; notice the additional memory
allocation and that this takes almost twice as long as the original for
loop.

Here’s a fused version of the vectorized calculation, where the `@.`
causes the loops to be fused.

``` julia
function fused_vectorized_calc(x, y)
    y .= @. exp(x) + 3 * sin(x)
end

@benchmark fused_vectorized_calc(x, y)
```

    BenchmarkTools.Trial: 8 samples with 1 evaluation.
     Range (min … max):  681.662 ms … 697.476 ms  ┊ GC (min … max): 0.00% … 0.00%
     Time  (median):     686.000 ms               ┊ GC (median):    0.00%
     Time  (mean ± σ):   687.778 ms ±   5.390 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

      █       █  █  █   █       █                    █            █  
      █▁▁▁▁▁▁▁█▁▁█▁▁█▁▁▁█▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
      682 ms           Histogram: frequency by time          697 ms <

     Memory estimate: 0 bytes, allocs estimate: 0.

We see that the time and (lack of) memory allocation are essentially the
same as the original basic for loop.

Finally one can achieve the same fusion by having the function just
compute scalar quantities and then using the vectorized version of the
function (by using `scalar_calc.()` instead of `scalar_calc()`), which
also does the fusion.

``` julia
function scalar_calc(x)
    return(exp(x) + 3 * sin(x))
end

@benchmark y .= scalar_calc.(x)
```

    BenchmarkTools.Trial: 8 samples with 1 evaluation.
     Range (min … max):  654.146 ms … 718.404 ms  ┊ GC (min … max): 0.00% … 0.00%
     Time  (median):     659.089 ms               ┊ GC (median):    0.00%
     Time  (mean ± σ):   669.009 ms ±  22.190 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

      █                                                              
      █▇▁▁▁▁▁▁▇▁▁▁▁▁▁▁▁▇▁▁▁▁▁▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇ ▁
      654 ms           Histogram: frequency by time          718 ms <

     Memory estimate: 32 bytes, allocs estimate: 2.
