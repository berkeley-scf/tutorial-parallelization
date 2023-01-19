---
layout: default
title: Parallel processing in R
---

# Parallel processing in R

## 1 Overview

R provides a variety of functionality for parallelization, including
threaded operations (linear algebra), parallel for loops and lapply-type
statements, and parallelization across multiple machines. This material
focuses on R’s future package, a flexible and powerful approach to
parallelization in R.

## 2 Threading

Threading in R is limited to linear algebra, provided R is linked
against a threaded BLAS.

### 2.1 What is the BLAS?

The BLAS is the library of basic linear algebra operations (written in
Fortran or C). A fast BLAS can greatly speed up linear algebra relative
to the default BLAS on a machine. Some fast BLAS libraries are

-   Intel’s *MKL*; may be available for educational use for free
-   *OpenBLAS*; open source and free
-   *vecLib* for Macs; provided with your Mac

In addition to being fast when used on a single core, all of these BLAS
libraries are threaded - if your computer has multiple cores and there
are free resources, your linear algebra will use multiple cores,
provided your installed R is linked against the threaded BLAS installed
on your machine.

**You can [use an optimized BLAS on your own
machine(s)](https://statistics.berkeley.edu/computing/blas).**

### 2.2 Example syntax

Here’s some code that illustrates the speed of using a threaded BLAS:

``` r
library(RhpcBLASctl)  ## package that controls number of threads from within R

x <- matrix(rnorm(5000^2), 5000)

## Control number of threads from within R. See next section for details.
blas_set_num_threads(4)
system.time({
   x <- crossprod(x)
   U <- chol(x)
})

#   user  system elapsed 
# 14.104   5.403   6.752 

blas_set_num_threads(1)
system.time({
   x <- crossprod(x)
   U <- chol(x)
})

#   user  system elapsed 
# 12.393   0.055  12.344 
```

Here the elapsed time indicates that using four threads gave us a two
times (2x) speedup in terms of real time, while the user time indicates
that the threaded calculation took a bit more total processing time
(combining time across all processors) because of the overhead of using
multiple threads. So the threading helps, but it’s not the 4x linear
speedup we would hope for.

### 2.3 Choosing the number of threads

In general, threaded code will detect the number of cores available on a
machine and make use of them. However, you can also explicitly control
the number of threads available to a process.

For most threaded code (that based on the openMP protocol), the number
of threads can be set by setting the OMP_NUM_THREADS environment
variable. Note that under some circumstances you may need to use
VECLIB_MAXIMUM_THREADS if on a Mac or MKL_NUM_THREADS if R is linked
against MKL (which can be seen by running `sessionInfo`).

For example, to set it for four threads in bash:

``` bash
export OMP_NUM_THREADS=4
```

Do this before starting your R or Python session or before running your
compiled executable.

Alternatively, you can set OMP_NUM_THREADS as you invoke your job, e.g.,
here with R:

    OMP_NUM_THREADS=4 R CMD BATCH --no-save job.R job.out

Finally, the R package, `RhpcBLASctl`, allows you to control the number
of threads from within R, as already seen in the example in the previous
subsection.

``` r
library(RhpcBLASctl)
blas_set_num_threads(4)
# now run your linear algebra
```

## 3 Parallel loops (including parallel lapply) via the future package

All of the functionality discussed here applies *only* if the
iterations/loops of your calculations can be done completely separately
and do not depend on one another. This scenario is called an
*embarrassingly parallel* computation. So coding up the evolution of a
time series or a Markov chain is not possible using these tools.
However, bootstrapping, random forests, simulation studies,
cross-validation and many other statistical methods can be handled in
this way.

One can easily parallelize lapply (or sapply) statements or parallelize
for loops using the `future` package. Here’s we’ll just show the basic
mechanics of using the future package. There’s much more detail in [this
SCF tutorial](https://berkeley-scf.github.io/tutorial-dask-future).

In Sections 3.1 and 3.2, we’ll parallelize across multiple cores on one
machine. Section 3.3 shows how to use multiple machines.

### 3.1 Parallel lapply

Here we’ll parallelize an lapply operation. We need to call `plan` to
set up the workers that will carry out the individual tasks (one for
each element of the input list or vector) in parallel.

The `multisession` “plan” simply starts worker processes on the machine
you are working on. You could skip the `workers` argument and the number
of workers will equal the number of cores on your machine. Later we’ll
see the use of the `multicore` and `cluster` “plans”, which set up the
workers in a different way.

Here we parallelize leave-one-out cross-validation for a random forest
model.

``` r
source('rf.R')  # loads in data (X and Y) and looFit()

library(future.apply)
## Set up four workers to run calculations in parallel
plan(multisession, workers = 4)

## Run the cross-validation in parallel, four tasks at a time on the four workers
system.time(
  out <- future_lapply(seq_along(Y), looFit, Y, X, future.seed = TRUE)
)   
#   user  system elapsed 
#  0.684   0.086  19.831 


## Side note: seq_along(Y) is a safe equivalent of 1:length(Y)
```

The use of `future.seed` ensures safe parallel random number generation
as discussed in Section 5.

Here the low user time is because the time spent in the worker processes
is not counted at the level of the overall master process that
dispatches the workers.

Note that one can use `plan` without specifying the number of workers,
in which case it will call `parallelly::availableCores()` and in general
set the number of workers to a sensible value based on your system (and
your scheduler allocation if your code is running on a cluster under a
scheduler such as Slurm).

### 3.2 Parallel for loops

We can use the future package in combination with the `foreach` command
to run a for loop in parallel. Of course this will only be valid if the
iterations can be computed independently.

The syntax for `foreach` is a bit different than a standard for loop.
Also note that the output for each iteration is simply the result of the
last line in the `{ }` body of the foreach statement.

``` r
source('rf.R')  # loads in data (X and Y) and looFit()
```

    ## randomForest 4.7-1.1

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library(future)
plan(multisession, workers = 4)

library(doFuture, quietly = TRUE)
registerDoFuture()

## Toy example of using foreach+future
out <- foreach(i = seq_len(30)) %dopar% {
    mean(1:i)
}
out[1:3]
```

    ## [[1]]
    ## [1] 1
    ## 
    ## [[2]]
    ## [1] 1.5
    ## 
    ## [[3]]
    ## [1] 2

``` r
## Replicate our cross-validation from future_lapply.

## Use %dorng% instead of the standard %dopar% to safely
## generate random numbers in parallel (Section 5)
library(doRNG)
```

    ## Loading required package: rngtools

``` r
out <- foreach(i = seq_along(Y), .combine = c) %dorng% {
    looFit(i, Y, X)
}
out[1:3]
```

    ##         1         2         3 
    ## 0.2772653 0.8708022 1.8286281

Note that *foreach* also provides functionality for collecting and
managing the results to avoid some of the bookkeeping you would need to
do if writing your own standard for loop. The result of *foreach* will
generally be a list, unless we request the results be combined in
different way, as we do here using `.combine = c` to use `c()` to get a
vector rather than a list.

You can debug by running serially using *%do%* rather than *%dopar%*.
Note that you may need to load packages within the *foreach* construct
to ensure a package is available to all of the calculations.

It is possible to use foreach to parallelize over nested loops. Suppose
that the outer loop has too few tasks to effectively parallelize over
and you also want to parallelize over the inner loop as well. Provided
the calculations in each task (defined based on the pair of indexes from
both loops) are independent of the other tasks, you can define two
foreach loops, with the outer foreach using the `%:%` operator and the
inner foreach using the usual `%dopar%` operator. More details can be
found [in this foreach
vignette](https://cran.r-project.org/web/packages/foreach/vignettes/nested.html).

### 3.3 Avoiding copies on each worker

The future package automatically identifies the objects needed by your
future-based code and makes copies of those objects once for each worker
process (thankfully not once for each task).

If you’re working with large objects, making a copy of the objects for
each of the worker processes can be a significant time cost and can
greatly increase your memory use.

The `multicore` plan (not available on Windows or in RStudio) forks the
main R process.

``` r
plan(multicore, workers = 4)
```

This creates R worker processes with the same state as the original R
process.

-   Importantly, this means that global variables in the forked worker
    processes are just references to the objects in memory in the
    original R process.
-   You can use these global variables in functions you call in parallel
    or pass the variables into functions as function arguments.
-   So **the additional processes do not use additional memory for those
    objects** (despite what is shown in *top* as memory used by each
    process).
-   And there is no time involved in making copies.
-   However, if you modify objects in the worker processes then copies
    are made.

So, the take-home message is that using `multicore` on non-Windows
machines can have a big advantage when working with large data objects.

### 3.4 Using multiple machines or cluster nodes

We can use the `cluster` plan to run workers across multiple machines.

If we know the names of the machines and can access them via
password-less SSH (e.g., [using ssh
keys](https://statistics.berkeley.edu/computing/ssh-keys)), then we can
simply provide the names of the machines to create a cluster and use the
‘cluster’ plan.

Here we want to use two cores on one machine and two on another.

``` r
library(future.apply)
workers <- c(rep('arwen.berkeley.edu', 2), rep('gandalf.berkeley.edu', 2))
plan(cluster, workers = workers)
# Now use parallel_lapply, foreach, etc. as before
```

If you are using the Slurm scheduler on a Linux cluster and in your
sbatch or srun command you use `--ntasks`, then the following will allow
you to use as many workers as the value of `ntasks`. One caveat is that
one still needs to be able to access the various machines via
password-less SSH.

``` r
plan(cluster)
# Now use parallel_lapply, parallel_sapply, foreach, etc. as before
```

Alternatively, you could set specify the workers manually. Here we use
`srun` (note this is being done within our original `sbatch` or `srun`)
to run hostname once per Slurm task, returning the name of the node the
task is assigned to.

``` r
workers <- system('srun hostname', intern = TRUE)
plan(cluster, workers = workers)
# Now use parallel_lapply, parallel_sapply, foreach, etc. as before
```

In all cases, we can verify that the workers are running on the various
nodes by checking the nodename of each of the workers:

``` r
tmp <- future_sapply(seq_len(nbrOfWorkers()), 
              function(i)
                cat("Worker running in process", Sys.getpid(),
                    "on", Sys.info()[['nodename']], "\n"))
```

## 4 Older alternatives to the future package for parallel loops/lapply

The future package allows you to do everything that one can do using
older packages/functions such as `mclapply`, `parLapply` and `foreach`
wit backends such as `doParallel`, `doSNOW`, `doMPI`. So my
recommendation is just to use the future package. But here is some
syntax for the older approaches.

As with calculations using the future package, all of the functionality
discussed here applies *only* if the iterations/loops of your
calculations can be done completely separately and do not depend on one
another. This scenario is called an *embarrassingly parallel*
computation. So coding up the evolution of a time series or a Markov
chain is not possible using these tools. However, bootstrapping, random
forests, simulation studies, cross-validation and many other statistical
methods can be handled in this way.

### 4.1 Parallel lapply

Here are a couple of the ways to do a parallel lapply:

``` r
library(parallel)
nCores <- 4  
cl <- makeCluster(nCores) 

# clusterExport(cl, c('x', 'y')) # if the processes need objects
# from master's workspace (not needed here as no global vars used)

# First approach: parLapply
result1 <- parLapply(cl, seq_along(Y), looFit, Y, X)
# Second approach: mclapply
result2 <- mclapply(seq_along(Y), looFit, Y, X)
```

### 4.2 Parallel for loops

And here’s how to use `doParallel` with foreach instead of `doFuture`.

``` r
library(doParallel)  # uses parallel package, a core R package

nCores <- 4  
registerDoParallel(nCores)

out <- foreach(i = seq_along(Y)) %dopar% {
    looFit(i, Y, X)
}
```

### 4.3 Avoiding copies on each worker

Whether you need to explicitly load packages and export global variables
from the main process to the parallelized worker processes depends on
the details of how you are doing the parallelization.

Under several scenarios (but only on Linux and MacOS, not on Windows),
packages and global variables in the main R process are automatically
available to the worker tasks without any work on your part. These
scenarios are

-   `foreach` with the `doParallel` backend,
-   parallel lapply (and related) statements when starting the cluster
    via `makeForkCluster`, instead of the usual `makeCluster`, and
-   use of `mclapply`.

This is because all of these approaches fork the original R process,
thereby creating worker processes with the same state as the original R
process. Interestingly, this means that global variables in the forked
worker processes are just references to the objects in memory in the
original R process. So the additional processes do not use additional
memory for those objects (despite what is shown in `top`) and there is
no time involved in making copies. However, if you modify objects in the
worker processes then copies are made.

Caveat: with `mclapply` you can use a global variable in functions you
call in parallel or pass the global variable in as an argument, in both
cases without copying. However with `parLapply`, passing the global
variable as an argument results in copies being made for some reason.

Importantly, because forking is not available on Windows, the above
statements only apply on Linux and MacOS.

In contrast, with parallel lapply (and related) statements (but not
foreach) when starting the cluster using the standard `makeCluster`
(which sets up a so-called *PSOCK* cluster, starting the R worker
processes via `Rscript`), one needs to load packages within the code
that is executed in parallel. In addition one needs to use
`clusterExport` to tell R which objects in the global environment should
be available to the worker processes. This involves making as many
copies of the objects as there are worker processes, so one can easily
exceed the physical memory (RAM) on the machine if one has large
objects, and the copying of large objects will take time.

### 4.4 Using multiple machines or cluster nodes

One can set up a cluster of workers across multiple nodes using
`parallel::makeCluster`. Then one can use `parLapply` and `foreach` with
that cluster of workers.

``` r
library(parallel)
machines = c(rep("gandalf.berkeley.edu", 2), rep("arwen.berkeley.edu", 2))

cl = makeCluster(machines, type = "SOCK")

# With parLapply or parSapply:

parSapply(cl, 1:5, function(i) return(mean(1:i)))
```

    ## [1] 1.0 1.5 2.0 2.5 3.0

``` r
# With foreach:
library(doSNOW, quietly = TRUE)
```

    ## 
    ## Attaching package: 'snow'

    ## The following objects are masked from 'package:parallel':
    ## 
    ##     clusterApply, clusterApplyLB, clusterCall, clusterEvalQ,
    ##     clusterExport, clusterMap, clusterSplit, makeCluster, parApply,
    ##     parCapply, parLapply, parRapply, parSapply, splitIndices,
    ##     stopCluster

``` r
registerDoSNOW(cl)
# Now use foreach as usual
```

For foreach, we used the `doSNOW` backend. The *doSNOW* backend has the
advantage over `doMPI` that it doesn’t need to have MPI installed on the
system.

## 5 Parallel random number generation

The key thing when thinking about random numbers in a parallel context
is that you want to avoid having the same ‘random’ numbers occur on
multiple processes. On a computer, random numbers are not actually
random but are generated as a sequence of pseudo-random numbers designed
to mimic true random numbers. The sequence is finite (but very long) and
eventually repeats itself. When one sets a seed, one is choosing a
position in that sequence to start from. Subsequent random numbers are
based on that subsequence. All random numbers can be generated from one
or more random uniform numbers, so we can just think about a sequence of
values between 0 and 1.

The worst thing that could happen is that one sets things up in such a
way that every process is using the same sequence of random numbers.
This could happen if you mistakenly set the same seed in each process,
e.g., using *set.seed(mySeed)* in R on every process.

The naive approach is to use a different seed for each process. E.g., if
your processes are numbered `id = 1,2,...,p` with a variable *id* that
is unique to a process, setting the seed to be the value of *id* on each
process. This is likely not to cause problems, but raises the danger
that two (or more sequences) might overlap. For an algorithm with
dependence on the full sequence, such as an MCMC, this probably won’t
cause big problems (though you likely wouldn’t know if it did), but for
something like simple simulation studies, some of your ‘independent’
samples could be exact replicates of a sample on another process. Given
the period length of the default generators in R, this is actually quite
unlikely, but it is a bit sloppy.

One approach to avoid the problem is to do all your RNG on one process
and distribute the random deviates, but this can be infeasible with many
random numbers.

More generally to avoid this problem, the key is to use an algorithm
that ensures sequences that do not overlap.

In R, the *rlecuyer* package deals with this. The L’Ecuyer algorithm has
a period of 2<sup>191</sup>, which it divides into subsequences of
length 2<sup>127</sup>.

### 5.1 Parallel RNG and the future package

The future package [integrates well with the L’Ecuyer parallel RNG
approach](https://www.jottr.org/2020/09/22/push-for-statical-sound-rng/#random-number-generation-in-the-future-framework),
which guarantees non-overlapping random numbers. There is a good
discussion about seeds for `future_lapply` and `future_sapply` in the
help for those functions.

#### 5.1.1 future_lapply

Here we can set a single seed. Behind the scenes the L’Ecuyer-CMRG RNG
is used so that the random numbers generated for each iteration are
independent. Note there is some overhead here when the number of
iterations is large.

``` r
library(future.apply)
n <- 40
set.seed(1)
out1 <- future_sapply(1:n, function(i) rnorm(1), future.seed = TRUE)
set.seed(1)
out2 <- future_sapply(1:n, function(i) rnorm(1), future.seed = TRUE)
identical(out1, out2)
```

    ## [1] TRUE

Basically future_lapply pregenerates a seed for each iteration using
`parallel:::nextRNGStream`, which uses the L’Ecuyer algorithm. See [more
details here](https://github.com/HenrikBengtsson/future/issues/126).

I could also have set `future.seed` to a numeric value, instead of
setting the seed using `set.seed`, to make the generated results
reproducible.

#### 5.1.2 foreach

See the example code in `help(doFuture)` for template code on how to use
the `%doRNG%` operator with foreach to ensure correct RNG with foreach.
(Also shown in Section 3.2.)

### 5.2 Parallel RNG with alternatives to the future package

#### 5.2.1 Parallel lapply style statements

Here’s how you initialize independent sequences on different processes
when using the *parallel* package’s parallel lapply functionality.

``` r
library(parallel)
library(rlecuyer)
nCores <- 4
cl <- makeCluster(nCores)
iseed <- 1
clusterSetRNGStream(cl = cl, iseed = iseed)
## Now proceed with your parLapply, using the `cl` object
```

With `mclapply` you can set the argument `mc.set.seed = TRUE`, which is
the default. This will give different seeds for each process, but for
safety, you should choose the L’Ecuyer algorithm via
`RNGkind("L'Ecuyer-CMRG")` before running `mclapply`.

#### 5.2.2 foreach

For foreach, you can use `registerDoRNG`:

``` r
library(doRNG)
library(doParallel)
registerDoParallel(4)
registerDoRNG(seed = 1)
## Now use foreach with %dopar%
```

#### 5.2.3 mclapply

When using *mclapply*, you can use the *mc.set.seed* argument as follows
(note that *mc.set.seed* is TRUE by default, so you should get different
seeds for the different processes by default), but one needs to invoke
`RNGkind("L'Ecuyer-CMRG")` to get independent streams via the L’Ecuyer
algorithm.

``` r
library(parallel)
library(rlecuyer)
RNGkind("L'Ecuyer-CMRG")
res <- mclapply(seq_len(Y), looFit, Y, X, mc.cores = 4, 
    mc.set.seed = TRUE) 
```

## 6 The *partools* package

*partools* is a package developed by Norm Matloff at UC-Davis. He has
the perspective that Spark/Hadoop are not the right tools in many cases
when doing statistics-related work and has developed some simple tools
for parallelizing computation across multiple nodes, also referred to as
*Snowdoop*. The tools make use of the key idea in Spark/Hadoop of a
distributed file system and distributed data objects but avoid the
complications of trying to ensure fault tolerance, which is critical
only on very large clusters of machines.

I won’t go into details, but *partools* allows you to split up your data
across multiple nodes and then read the data into R in parallel across R
sessions running on those nodes, all controlled from a single master R
session. You can then do operations on the subsets and gather results
back to the master session as needed. One point that confused me in the
*partools* vignette is that it shows how to split up a dataset that you
can read into your R session, but it’s not clear what one does if the
dataset is too big to read into a single R session.
