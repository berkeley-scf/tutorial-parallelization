---
layout: default
title: Parallel processing in Python
---

# Parallel processing in Python

## 1 Overview

Python provides a variety of functionality for parallelization, including threaded operations (in particular for linear algebra), parallel looping and map statements, and parallelization across multiple machines. This material focuses on Python's ipyparallel package, with some discussion of Dask and Ray.

All of the functionality discussed here applies *only* if the iterations/loops of your calculations can be done completely separately and do not depend on one another. This scenario is called an *embarrassingly parallel* computation.  So coding up the evolution of a time series or a Markov chain is not possible using these tools. However, bootstrapping, random forests, simulation studies, cross-validation and many other statistical methods can be handled in this way.


## 2 Threading

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
provided your installed Python  is linked against the threaded BLAS installed
on your machine.

To use a fast, threaded BLAS, one approach is to use the Anaconda/Miniconda Python distribution. When you install numpy and scipy, these should be [automatically linked](https://www.anaconda.com/blog/scikit-learn-speed-up-with-intel-and-anaconda) against a fast, threaded BLAS (MKL). More generally, simply installing numpy from PyPI [should make use of OpenBLAS](https://numpy.org/install/).

### 2.2 Example syntax

Threading in Python is limited to linear algebra (except if using Dask), provided Python is linked against a threaded BLAS.  Python has something
called the [Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock)
that interferes with threading in Python (but not in threaded linear algebra packages called by Python). 

Here's some linear algebra in  Python that will use threading if *numpy* is linked against a threaded BLAS, though I don't compare the timing for different numbers of threads here. 

```python
import numpy as np
n = 5000
x = np.random.normal(0, 1, size=(n, n))
x = x.T.dot(x)
U = np.linalg.cholesky(x)
```

If you watch the Python process via the top command, you should see CPU usage above 100% if Python is linking to a threaded BLAS.

### 2.3 Fixing the number of threads (cores used)

In general, threaded code will
detect the number of cores available on a machine and make use of
them. However, you can also explicitly control the number of threads
available to a process. 

For most threaded code (that based on the openMP protocol), the number
of threads can be set by setting the OMP_NUM_THREADS environment
variable. Note that under some circumstances you may need to use VECLIB_MAXIMUM_THREADS if on a Mac or MKL_NUM_THREADS if numpy/scipy are linked against MKL. 

For example, to set it for four
threads in bash, do this before starting your Python session.

```bash
export OMP_NUM_THREADS=4
```

Alternatively, you can set OMP_NUM_THREADS as you invoke your job, e.g., 

```bash
OMP_NUM_THREADS=4 python job.py > job.out
```

 
## 3 Basic parallelized loops/maps/apply using ipyparallel

### 3.1 Parallel looping on one machine

#### 3.1.1 Starting the workers

First we'll cover IPython Parallel (i.e., the `ipyparallel` package) functionality, which allows one to parallelize on a single machine (discussed here) or across multiple machines (see next section). In later sections, I'll discuss other packages that can be used for parallelization.

First we need to start our workers. As of ipyparallel version 7, we can start the workers from within Python.

```python
## In newer versions of ipyparallel (v. 7 and later)
import ipyparallel as ipp
# Check the version:
ipp.__version__
n = 4
cluster = ipp.Cluster(n = n)
c = cluster.start_and_connect_sync()
```

ipp.Cluster(n=cpu_count).start_and_connect_sync()


#### 3.1.2 Testing our workers

Let's verify that things seem set up ok and we can interact with all our workers:

```python
## Check that we have the number of workers expected:
c.ids
# [0, 1, 2, 3]
## Set up a direct view to interface with all the workers
dview = c[:]
dview
# <DirectView [0, 1, 2, 3]>
## Set blocking so that we wait for the result of the parallel execution
dview.block = True 
dview.apply(lambda : "Hello, World")
# ['Hello, World', 'Hello, World', 'Hello, World', 'Hello, World']
```

`dview` stands for a 'direct view', which is an interface to our cluster that allows us to 'manually' send tasks to the workers.

#### 3.1.3 Parallelized machine learning example: setup

Now let's see an example of how we can use our workers to run code in parallel. 

We'll carry out a statistics/machine learning prediction method (random forest regression) with leave-one-out cross-validation, parallelizing over different held out data.

First let's set up packages, data and our main function on the workers:

```python
dview.execute('from sklearn.ensemble import RandomForestRegressor as rfr')
dview.execute('import numpy as np')

def looFit(index, Ylocal, Xlocal):
    rf = rfr(n_estimators=100)
    fitted = rf.fit(np.delete(Xlocal, index, axis = 0), np.delete(Ylocal, index))
    pred = rf.predict(np.array([Xlocal[index, :]]))
    return(pred[0])

import numpy as np
np.random.seed(0)
n = 200
p = 20
X = np.random.normal(0, 1, size = (n, p))
Y = X[: , 0] + pow(abs(X[:,1] * X[:,2]), 0.5) + X[:,1] - X[:,2] + 
    np.random.normal(0, 1, n)

mydict = dict(X = X, Y = Y, looFit = looFit)
dview.push(mydict)
```

#### 3.1.4 Parallelized machine learning example: execution

Now let's set up a "load-balanced view". With this type of interface, one submits the tasks and the controller decides how to divide up the tasks, ideally achieving good load balancing. A load-balanced computation is one that keeps all the workers busy throughout the computation

```python
lview = c.load_balanced_view()
lview.block = True

# need a wrapper function because map() only operates on one argument
def wrapper(i):
    return(looFit(i, Y, X))

# Now run the fitting, predicting on each held-out observation:
pred = lview.map(wrapper, range(n))
# Check a few predictions:
pred[0:3]
```

#### 3.1.5 Starting the workers outside Python

One can also start the workers outside of Python. This was required in older versions of ipyparallel, before version 7.

```bash
# In the bash shell:
export NWORKERS=4
ipcluster start -n ${NWORKERS} &
```

Now in Python, we can connect to the running workers:

```python
# In python
import os
import ipyparallel as ipp
c = ipp.Client()
c.wait_for_engines(n = int(os.environ['NWORKERS']))
c.ids
# Now do your parallel computations
```

Finally, stop the workers.

```bash
ipcluster stop
```

### 3.2 Using multiple machines or cluster nodes

One can use ipyparallel in a context with multiple nodes, though the setup to get the worker processes started is a bit more involved when you have multiple nodes. 

If we are using the SLURM scheduling software, here's how we start up the worker processes:

```bash
# In the bash shell (e.g., in your Slurm job script)
ipcontroller --ip='*' &
sleep 60
# Next start as many ipengines (workers) as we have Slurm tasks. 
# This works because srun is a Slurm command, 
# so it knows it is running within a Slurm allocation
srun ipengine &
```

At this point you should be able to connect to the running cluster using the syntax seen for single-node usage.

> **Warning**: Be careful to set the sleep period long enough that the controller starts before trying to start the workers and the workers start before trying to connect to the workers from within Python.

After doing your computations and quitting your main Python session, shut down the cluster of workers:
```bash
ipcluster stop
```

To start the engines in a context outside of using Slurm (provided all machines share a filesystem), you should be able ssh to each machine and run `ipengine & ` for as many worker processes as you want to start as follows. In some, but not all cases (depending on how the network is set up) you may not need the `--location` flag, but if you do, it should be set to the name of the machine you're working on, e.g., by using the HOST environment variable. Here we start all the workers on a single other machine, "other_host":

```bash
ipcontroller --ip='*' --location=${HOST} &
sleep 60
NWORKERS=4
ssh other_host "for (( i = 0; i < ${NWORKERS}; i++ )); do ipengine &; done"
```

## 4 Dask and Ray

Dask and Ray are powerful packages for parallelization that allow one to parallelize tasks in similar fashion to ipyparallel. But they also provide additional useful functionality: Dask allows one to work with large datasets that are split up across multiple processes on (potentially) multiple nodes, providing Spark/Hadoop-like functionality. Ray allows one to develop complicated apps that execute in parallel using the notion of *actors*.

For more details on using distributed dataset with Dask, see [this Dask tutorial](https://berkeley-scf.github.io/tutorial-dask-future/python-dask.html). For more details on Ray's actors, please see the [Ray documentation](https://www.ray.io/docs).

### 4.1 Parallel looping in Dask

There are various ways to do parallel loops in Dask, as discussed in detail in [this Dask tutorial](https://berkeley-scf.github.io/tutorial-dask-future/python-dask.html).

Here's an example of doing it with "delayed" calculations set up via list comprehension. First we'll start workers on a single machine. One can also start workers on multiple machines, as discussed in the tutorial linked to just above.

```python
import dask.multiprocessing
dask.config.set(scheduler='processes', num_workers = 4)
```

Now we'll execute a set of tasks in parallel by wrapping the function of interest in `dask.delayed` to set up lazy evaluation that will be done in parallel using the workers already set up with the 'processes' scheduler above.

```python
def calc_mean(i, n):
    import numpy as np
    rng = np.random.default_rng(i)
    data = rng.normal(size = n)
    return([np.mean(data), np.std(data)])

n = 1000
p = 10
futures = [dask.delayed(calc_mean)(i, n) for i in range(p)]
futures  # This is an array of placeholders for the tasks to be carried out.
# [Delayed('calc_mean-b07564ff-149a-4db7-ac3c-1cc89b898fe5'), 
# Delayed('calc_mean-f602cd67-97ad-4293-aeb8-e58be55a89d6'), 
# Delayed('calc_mean-d9448f54-b1db-46aa-b367-93a46e1c202a'), ...

# Now ask for the output to trigger the lazy evaluation.
results = dask.compute(futures)
```

Execution only starts when we call `dask.compute`.

Note that we set a separate seed for each task to try to ensure indepenedent random numbers between tasks, but Section 5 discusses better ways to do this.

### 4.2 Parallel looping in Ray

We'll start up  workers on a single machine. To run across multiple workers, see [this tutorial](https://berkeley-scf.github.io/tutorial-dask-future/python-ray.html) or the [Ray documentation](https://www.ray.io/docs).

```python
import ray
ray.init(num_cpus = 4)
```

To run a computation in parallel, we decorate the function of interest with the `remote` tag:

```python
@ray.remote
def calc_mean(i, n):
    import numpy as np
    rng = np.random.default_rng(i)
    data = rng.normal(size = n)
    return([np.mean(data), np.std(data)])

n = 1000
p = 10
futures = [calc_mean.remote(i, n) for i in range(p)]
futures  # This is an array of placeholders for the tasks to be carried out.
# [ObjectRef(a67dc375e60ddd1affffffffffffffffffffffff0100000001000000), 
# ObjectRef(63964fa4841d4a2effffffffffffffffffffffff0100000001000000), ...

# Now trigger the computation
ray.get(futures)
```

## 5 Random number generation (RNG) in parallel 

### 5.1 Overview

The key thing when thinking about random numbers in a parallel context
is that you want to avoid having the same 'random' numbers occur on
multiple processes. On a computer, random numbers are not actually
random but are generated as a sequence of pseudo-random numbers designed
to mimic true random numbers. The sequence is finite (but very long)
and eventually repeats itself. When one sets a seed, one is choosing
a position in that sequence to start from. Subsequent random numbers
are based on that subsequence. All random numbers can be generated
from one or more random uniform numbers, so we can just think about
a sequence of values between 0 and 1. 

The worst thing that could happen is that one sets things up in such
a way that every process is using the same sequence of random numbers.
This could happen if you mistakenly set the same seed in each process,
e.g., using `rng = np.random.default_rng(1)` or `np.random.seed(1)` in Python for every worker.

The naive approach is to use a different seed for each process. E.g.,
if your processes are numbered `id = 1,2,...,p`  with a variable *id* that is  unique
to a process, setting the seed to be the value of *id* on each process. This is likely
not to cause problems, but raises the danger that two (or more) subsequences
might overlap. For an algorithm with dependence on the full subsequence,
such as an MCMC, this probably won't cause big problems (though you
likely wouldn't know if it did), but for something like simple simulation
studies, some of your 'independent' samples could be exact replicates
of a sample on another process. Given the period length of the default
generator in Python, this is actually quite unlikely, but it is a bit sloppy.

To avoid this problem, the key is to use an algorithm
that ensures sequences that do not overlap.

### 5.2 Parallel RNG in practice


In recent versions of numpy there has been attention paid to this problem and there are now [multiple approaches to getting high-quality random number generation for parallel code](https://numpy.org/doc/stable/reference/random/parallel.html).

One approach is to generate one random seed per task such that the blocks of random numbers avoid overlapping with high probability, as implemented in numpy's `SeedSequence` approach.

Here we use that approach within the context of an ipyparallel load-balanced view.

```python
import numpy as np
import ipyparallel as ipp
n = 4
cluster = ipp.Cluster(n = n)
cluster.start_cluster_sync()

c = cluster.connect_client_sync()
c.wait_for_engines(n)
c.ids

lview = c.load_balanced_view()
lview.block = True

n = 1000
p = 10

seed = 1
ss = np.random.SeedSequence(seed)
child_seeds = ss.spawn(p)

def calc_mean(i, n, seed_i):
    import numpy as np
    rng = np.random.default_rng(seed_i)
    data = rng.normal(size = n)
    return([np.mean(data), np.std(data)])

# need a wrapper function because map() only operates on one argument
def wrapper(i):
    return(calc_mean(i, n, child_seeds[i]))

dview = c[:]
dview.block = True 
mydict = dict(calc_mean = calc_mean, n = n, child_seeds = child_seeds)
dview.push(mydict)

results = lview.map(wrapper, range(p))
```

A second approach is to advance the state of the random number generator as if a large number of random numbers had been drawn. 

```python
seed = 1
pcg64 = np.random.PCG64(seed)

def calc_mean(i, n, rng):
    import numpy as np
    rng = np.random.Generator(pcg64.jumped(i))  ## jump in large steps, one jump per task
    data = rng.normal(size = n)
    return([np.mean(data), np.std(data)])

# need a wrapper function because map() only operates on one argument
def wrapper(i):
    return(calc_mean(i, n, rng))

dview = c[:]
dview.block = True 
mydict = dict(calc_mean = calc_mean, n = n, rng = rng)
dview.push(mydict)

results = lview.map(wrapper, range(p))
```

Note that above, I've done everything at the level of the computational tasks. One could presumably
do this at the level of the workers, but one would need to figure out how to maintain the state of the generator from one task to the next for any given worker.




