# Parallel processing in Python

# Parallel processing in Python

## 1 Overview

Python provides a variety of functionality for parallelization,
including threaded operations (in particular for linear algebra),
parallel looping and map statements, and parallelization across multiple
machines. For the CPU, this material focuses on Python’s ipyparallel
package and JAX, with some discussion of Dask and Ray. For the GPU, the
material focuses on PyTorch and JAX, with a bit of discussion of CuPy.

Note that all of the looping-based functionality discussed here applies
*only* if the iterations/loops of your calculations can be done
completely separately and do not depend on one another. This scenario is
called an *embarrassingly parallel* computation. So coding up the
evolution of a time series or a Markov chain is not possible using these
tools. However, bootstrapping, random forests, simulation studies,
cross-validation and many other statistical methods can be handled in
this way.

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
provided your installed Python is linked against the threaded BLAS
installed on your machine.

To use a fast, threaded BLAS, one approach is to use the
Anaconda/Miniconda Python distribution. When you install numpy and
scipy, these should be [automatically
linked](https://www.anaconda.com/blog/scikit-learn-speed-up-with-intel-and-anaconda)
against a fast, threaded BLAS (MKL). More generally, simply installing
numpy from PyPI [should make use of
OpenBLAS](https://numpy.org/install/).

### 2.2 Example syntax

Threading in Python is limited to linear algebra (provided Python is
linked against a threaded BLAS, except if using Dask or JAX or various
other packages). Python has something called the [Global Interpreter
Lock](https://wiki.python.org/moin/GlobalInterpreterLock) that
interferes with threading in Python (but not in threaded linear algebra
packages called by Python).

Here’s some linear algebra in Python that will use threading if *numpy*
is linked against a threaded BLAS, though I don’t compare the timing for
different numbers of threads here.

``` python
import numpy as np
n = 5000
x = np.random.normal(0, 1, size=(n, n))
x = x.T @ x
U = np.linalg.cholesky(x)
```

If you watch the Python process via the top command, you should see CPU
usage above 100% if Python is linking to a threaded BLAS.

### 2.3 Fixing the number of threads (cores used)

In general, threaded code will detect the number of cores available on a
machine and make use of them. However, you can also explicitly control
the number of threads available to a process.

For most threaded code (that based on the openMP protocol), the number
of threads can be set by setting the OMP_NUM_THREADS environment
variable. Note that under some circumstances you may need to use
VECLIB_MAXIMUM_THREADS if on an (older, Intel-based) Mac or
MKL_NUM_THREADS if numpy/scipy are linked against MKL.

For example, to set it for four threads in bash, do this before starting
your Python session.

``` bash
export OMP_NUM_THREADS=4
```

Alternatively, you can set OMP_NUM_THREADS as you invoke your job, e.g.,

``` bash
OMP_NUM_THREADS=4 python job.py > job.out
```

## 3 Basic parallelized loops/maps/apply using ipyparallel

### 3.1 Parallel looping on one machine

#### 3.1.1 Starting the workers

First we’ll cover IPython Parallel (i.e., the `ipyparallel` package)
functionality, which allows one to parallelize on a single machine
(discussed here) or across multiple machines (see next section). In
later sections, I’ll discuss other packages that can be used for
parallelization.

First we need to start our workers. As of ipyparallel version 7, we can
start the workers from within Python.

``` python
## In newer versions of ipyparallel (v. 7 and later)
import ipyparallel as ipp
# Check the version:
ipp.__version__
n = 4
cluster = ipp.Cluster(n = n)
c = cluster.start_and_connect_sync()
```

    Starting 4 engines with <class 'ipyparallel.cluster.launcher.LocalEngineSetLauncher'>

      0%|          | 0/4 [00:00<?, ?engine/s]

#### 3.1.2 Testing our workers

Let’s verify that things seem set up ok and we can interact with all our
workers:

``` python
## Check that we have the number of workers expected:
c.ids
```

    [0, 1, 2, 3]

``` python
## Set up a direct view to interface with all the workers
dview = c[:]
dview
```

    <DirectView [0, 1, 2, 3]>

``` python
## Set blocking so that we wait for the result of the parallel execution
dview.block = True 
dview.apply(lambda : "Hello, World")
```

    ['Hello, World', 'Hello, World', 'Hello, World', 'Hello, World']

`dview` stands for a ‘direct view’, which is an interface to our cluster
that allows us to ‘manually’ send tasks to the workers.

#### 3.1.3 Parallelized machine learning example: setup

Now let’s see an example of how we can use our workers to run code in
parallel.

We’ll carry out a statistics/machine learning prediction method (random
forest regression) with leave-one-out cross-validation, parallelizing
over different held out data.

First let’s set up packages, data and our main function on the workers:

``` python
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
Y = X[: , 0] + pow(abs(X[:,1] * X[:,2]), 0.5) + X[:,1] - X[:,2] + \
    np.random.normal(0, 1, n)

mydict = dict(X = X, Y = Y, looFit = looFit)
dview.push(mydict)
```

    [None, None, None, None]

#### 3.1.4 Parallelized machine learning example: execution

Now let’s set up a “load-balanced view”. With this type of interface,
one submits the tasks and the controller decides how to divide up the
tasks, ideally achieving good load balancing. A load-balanced
computation is one that keeps all the workers busy throughout the
computation

``` python
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

    [2.0945225368269256, -0.8696741139958911, -0.32442762057816776]

#### 3.1.5 Starting the workers outside Python

One can also start the workers outside of Python. This was required in
older versions of ipyparallel, before version 7.

``` bash
# In the bash shell:
export NWORKERS=4
ipcluster start -n ${NWORKERS} &
```

Now in Python, we can connect to the running workers:

``` python
# In python
import os
import ipyparallel as ipp
c = ipp.Client()
c.wait_for_engines(n = int(os.environ['NWORKERS']))
c.ids
# Now do your parallel computations
```

Finally, stop the workers.

``` bash
ipcluster stop
```

### 3.2 Using multiple machines or cluster nodes

One can use ipyparallel in a context with multiple nodes, though the
setup to get the worker processes started is a bit more involved when
you have multiple nodes.

If we are using the SLURM scheduling software, here’s how we start up
the worker processes:

``` bash
# In the bash shell (e.g., in your Slurm job script)
ipcontroller --ip='*' &
sleep 60
# Next start as many ipengines (workers) as we have Slurm tasks. 
# This works because srun is a Slurm command, 
# so it knows it is running within a Slurm allocation
srun ipengine &
```

At this point you should be able to connect to the running cluster using
the syntax seen for single-node usage.

> **Warning**: Be careful to set the sleep period long enough that the
> controller starts before trying to start the workers and the workers
> start before trying to connect to the workers from within Python.

After doing your computations and quitting your main Python session,
shut down the cluster of workers:

``` bash
ipcluster stop
```

To start the engines in a context outside of using Slurm (provided all
machines share a filesystem), you should be able ssh to each machine and
run `ipengine &` for as many worker processes as you want to start as
follows. In some, but not all cases (depending on how the network is set
up) you may not need the `--location` flag, but if you do, it should be
set to the name of the machine you’re working on, e.g., by using the
HOST environment variable. Here we start all the workers on a single
other machine, “other_host”:

``` bash
ipcontroller --ip='*' --location=${HOST} &
sleep 60
NWORKERS=4
ssh other_host "for (( i = 0; i < ${NWORKERS}; i++ )); do ipengine &; done"
```

## 4 Dask and Ray

Dask and Ray are powerful packages for parallelization that allow one to
parallelize tasks in similar fashion to ipyparallel. But they also
provide additional useful functionality: Dask allows one to work with
large datasets that are split up across multiple processes on
(potentially) multiple nodes, providing Spark/Hadoop-like functionality.
Ray allows one to develop complicated apps that execute in parallel
using the notion of *actors*.

For more details on using distributed dataset with Dask, see [this Dask
tutorial](https://berkeley-scf.github.io/tutorial-dask-future/python-dask.html).
For more details on Ray’s actors, please see the [Ray
documentation](https://www.ray.io/docs).

### 4.1 Parallel looping in Dask

There are various ways to do parallel loops in Dask, as discussed in
detail in [this Dask
tutorial](https://berkeley-scf.github.io/tutorial-dask-future/python-dask.html).

Here’s an example of doing it with “delayed” calculations set up via
list comprehension. First we’ll start workers on a single machine. One
can also start workers on multiple machines, as discussed in the
tutorial linked to just above.

``` python
import dask.multiprocessing
dask.config.set(scheduler='processes', num_workers = 4)
```

Now we’ll execute a set of tasks in parallel by wrapping the function of
interest in `dask.delayed` to set up lazy evaluation that will be done
in parallel using the workers already set up with the ‘processes’
scheduler above.

``` python
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

Note that we set a separate seed for each task to try to ensure
indepenedent random numbers between tasks, but Section 5 discusses
better ways to do this.

### 4.2 Parallel looping in Ray

We’ll start up workers on a single machine. To run across multiple
workers, see [this
tutorial](https://berkeley-scf.github.io/tutorial-dask-future/python-ray.html)
or the [Ray documentation](https://www.ray.io/docs).

``` python
import ray
ray.init(num_cpus = 4)
```

To run a computation in parallel, we decorate the function of interest
with the `remote` tag:

``` python
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
e.g., using `rng = np.random.default_rng(1)` or `np.random.seed(1)` in
Python for every worker.

The naive approach is to use a different seed for each process. E.g., if
your processes are numbered `id = 1,2,...,p` with a variable *id* that
is unique to a process, setting the seed to be the value of *id* on each
process. This is likely not to cause problems, but raises the danger
that two (or more) subsequences might overlap. For an algorithm with
dependence on the full subsequence, such as an MCMC, this probably won’t
cause big problems (though you likely wouldn’t know if it did), but for
something like simple simulation studies, some of your ‘independent’
samples could be exact replicates of a sample on another process. Given
the period length of the default generator in Python, this is actually
quite unlikely, but it is a bit sloppy.

To avoid this problem, the key is to use an algorithm that ensures
sequences that do not overlap.

### 5.2 Parallel RNG in practice

In recent versions of numpy there has been attention paid to this
problem and there are now [multiple approaches to getting high-quality
random number generation for parallel
code](https://numpy.org/doc/stable/reference/random/parallel.html).

One approach is to generate one random seed per task such that the
blocks of random numbers avoid overlapping with high probability, as
implemented in numpy’s `SeedSequence` approach.

Here we use that approach within the context of an ipyparallel
load-balanced view.

``` python
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

A second approach is to advance the state of the random number generator
as if a large number of random numbers had been drawn.

``` python
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

Note that above, I’ve done everything at the level of the computational
tasks. One could presumably do this at the level of the workers, but one
would need to figure out how to maintain the state of the generator from
one task to the next for any given worker.

## 6 Using the GPU via PyTorch

Python is the go-to language used to run computations on a GPU. Some of
the packages that can easily offload computations to the GPU include
PyTorch, Tensorflow, JAX, and CuPy. (Of course PyTorch and Tensorflow
are famously used for deep learning, but they’re also general numerical
computing packages.) We’ll discuss some of these.

There are a couple key things to remember about using a GPU:

-   The GPU memory is separate from CPU memory, and transferring data
    from the CPU to GPU (or back) is often more costly than doing the
    computation on the GPU.
    -   If possible, generate the data on the GPU or keep the data on
        the GPU when carrying out a sequence of operations.
-   By default GPU calculations are often doing using 32-bit (4-byte)
    floating point numbers rather than the standard of 64-bit (8-byte)
    when on the CPU.
    -   This can affect speed comparisons between CPU and GPU if one
        doesn’t compare operations with the same types of floating point
        numbers.
-   GPU operations are often asynchronous – they’ll continue in the
    background after they start, returning control of your Python
    session to you and potentially making it seem like the computation
    happened more quickly than it did.
    -   In the examples below, note syntax that ensures the operation is
        done before timing concludes (e.g., `cuda.synchronize` for
        PyTorch and `block_until_ready` for JAX).

Note that for this section, I’m pasting in the output when running the
code separately on a machine with a GPU because this document is
generated on a machine without a GPU.

### 6.1 Matrix multiplication

By default PyTorch will use 32-bit numbers.

``` python
import torch
import time

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

gpu = torch.device("cuda:0")

n = 7000

def matmul_wrap(x, y):
    z = torch.matmul(x, y)
    return(z)
    
## Generate data on the CPU.    
x = torch.randn(n,n)
y = torch.randn(n,n)

## Copy the objects to the GPU.
x_gpu = x.cuda() # or: `x.to("cuda")`
y_gpu = y.cuda()
    
torch.set_num_threads(1)
    
t0 = time.time()
z = matmul_wrap(x, y)
print(time.time() - t0)  # 6.8 sec.

start.record()
z_gpu = matmul_wrap(x_gpu, y_gpu)
torch.cuda.synchronize()
end.record()
print(start.elapsed_time(end))  # 70 milliseconds (ms)
```

So we achieved a speedup of about 100-fold over a single CPU core using
an A100 GPU in this case.

Let’s consider the time for copying data to the GPU:

``` python
x = torch.randn(n,n)
start.record()
x_gpu = x.cuda()
torch.cuda.synchronize()
end.record()
print(start.elapsed_time(end))  # 60 ms
```

This suggests that the time in copying the data is similar to that for
doing the matrix multiplication.

We can generate data on the GPU like this:

``` python
x_gpu = torch.randn(n,n, device=gpu)
```

### 6.2 Vectorized calculations (and loop fusion)

Here we’ll consider using the GPU for vectorized calculations. We’ll
compare using numpy, CPU-based PyTorch, and GPU-based PyTorch, again
with 32-bit numbers.

``` python
import torch
import numpy as np
import time

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

gpu = torch.device("cuda:0")

def myfun_np(x):
    y = np.exp(x) + 3 * np.sin(x)
    return(y)

def myfun_torch(x):
    y = torch.exp(x) + 3 * torch.sin(x)
    return(y)
    
    
n = 250000000
x = torch.randn(n)
x_gpu = x.cuda() # or: `x.to("cuda")`
tmp = np.random.normal(size = n)
x_np = tmp.astype(np.float32)  # for fair comparison

## numpy
t0 = time.time()
y_np = myfun_np(x_np)
time.time()-t0   # 1.2 sec.

## CPU-based torch (1 thread)
torch.set_num_threads(1)
start.record()
y = myfun_torch(x)
end.record()
print(start.elapsed_time(end))  # 2200 ms (2.2 sec.)

## GPU-based torch
start.record()
y_gpu = myfun_torch(x_gpu)
torch.cuda.synchronize()
end.record()
print(start.elapsed_time(end))   # 9 ms
```

So using the GPU speeds things up by 150-fold (compared to numpy) and
250-fold (compared to CPU-based PyTorch).

One can also have PyTorch “fuse” the operations in the loop, which
avoids having the different vectorized operations in `myfun` being done
in separate loops under the hood. For an overview of loop fusion, see
[this discussion](./parallel-julia#4-loops-and-fused-operations) in the
context of Julia.

To fuse the operations, we need to have the function in a module. In
this case I defined `myfun_torch` in `myfun_torch.py`, and we need to
compile the code using `torch.jit.script`.

``` python
from myfun_torch import myfun_torch as myfun_torch_tmp
myfun_torch_compiled = torch.jit.script(myfun_torch_tmp)

## CPU plus loop fusion
start.record()
y = myfun_torch_compiled(x)
end.record()
print(start.elapsed_time(end))   # 1000 ms (1 sec.)

## GPU plus loop fusion
start.record()
y_gpu = myfun_torch_compiled(x_gpu)
torch.cuda.synchronize()
end.record()
print(start.elapsed_time(end))   # 3.5 ms
```

So that seems to give a 2-3 fold speedup compared to without loop
fusion.

### 6.3 Using Apple’s M2 GPU

One can also use PyTorch to run computations on the GPU that comes with
Apple’s M2 chips.

The “backend” is called “MPS”, where “M” stands for “Metal”, which is
what Apple calls its GPU framework.

``` python
import torch
import time

start = torch.mps.Event(enable_timing=True)
end = torch.mps.Event(enable_timing=True)

mps_device = torch.device("mps")

n = 10000
x = torch.randn(n,n)
y = torch.randn(n,n) 

x_mps = x.to("mps")
y_mps = y.to("mps")
    
## On the CPU    
torch.set_num_threads(1)

t0 = time.time()
z = matmul_wrap(x, y)
print(time.time() - t0)   # 1.8 sec (1800 ms)

## On the M2 GPU
start.record()
z_mps = matmul_wrap(x_mps, y_mps)
torch.mps.synchronize()
end.record()
print(start.elapsed_time(end)) # 950 ms
```

So there is about a two-fold speed up, which isn’t impressive compared
to the speedup on a standard GPU.

Let’s see how much time is involved in transferring the data.

``` python
x = torch.randn(n,n)

start.record()
x_mps = x.to("mps")
torch.mps.synchronize()
end.record()
print(start.elapsed_time(end))  # 35 ms.
```

So it looks like the transfer time is pretty small compared to the
computation time (and to the savings involved in using the M2 GPU).

We can generate data on the GPU like this:

``` python
x_mps = torch.randn(n,n, device=mps_device)
```

## 7 Using JAX (for CPU and GPU)

You can think of JAX as a version of numpy enabled to use the GPU (or
automatically parallelize on CPU threads) and provide automatic
differentiation.

One can also use just-in-time (JIT) compilation with JAX. Behind the
scenes, the instructions are compiled to machine code for different
backends (e.g., CPU and GPU) using XLA.

### 7.1 Vectorized calculations (and loop fusion)

Let’s first consider running a vectorized calculation using JAX on the
CPU, which will use multiple threads, each thread running on a separate
CPU core on our computer.

``` python
import time
import numpy as np
import jax.numpy as jnp

def myfun_np(x):
    y = np.exp(x) + 3 * np.sin(x)
    return(y)
    
def myfun_jnp(x):
    y = jnp.exp(x) + 3 * jnp.sin(x)
    return(y)

n = 250000000

x = np.random.normal(size = n).astype(np.float32)  # for consistency
x_jax = jnp.array(x)  # 32-bit by default
print(x_jax.platform())
```

    cpu

``` python
t0 = time.time()
z = myfun_np(x)
t1 = time.time() - t0

t0 = time.time()
z_jax = myfun_jnp(x_jax).block_until_ready()
t2 = time.time() - t0

print(f"numpy time: {round(t1,3)}\njax time: {round(t2,3)}")
```

    numpy time: 4.49
    jax time: 1.643

There’s a nice speedup compared to numpy.

Since JAX will often execute computations asynchronously (in particular
when using the GPU), the `block_until_ready` invocation ensures that the
computation finishes before we stop timing.

By default the JAX floating point type is 32-bit so we forced the use of
32-bit numbers for numpy for comparability. One could have JAX use
64-bit numbers like this:

``` python
import jax.config
jax.config.update("jax_enable_x64", True)  
```

Next let’s consider JIT compiling it, which should [fuse the vectorized
operations](./parallel-julia#4-loops-and-fused-operations) and avoid
temporary objects. The JAX docs have a [nice
discussion](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#to-jit-or-not-to-jit)
of when JIT compilation will be beneficial.

``` python
import jax
myfun_jnp_jit = jax.jit(myfun_jnp)

t0 = time.time()
z_jax_jit = myfun_jnp_jit(x_jax).block_until_ready()
t3 = time.time() - t0
print(f"jitted jax time: {round(t3,3)}")
```

    jitted jax time: 0.793

So that gives another almost 2x speedup.

### 7.2 Linear algebra

Linear algebra in JAX will use multiple threads ([as discussed for
numpy](./parallel-python#2-threading)). Here we’ll compare 64-bit
calculation, since matrix decompositions sometimes need more precision.

``` python
n = 7000
x = np.random.normal(0, 1, size=(n, n))

t0 = time.time()
mat = x.T @ x
print("numpy time:")
print(round(time.time() - t0,3))

t0 = time.time()
U = np.linalg.cholesky(mat) 
print(round(time.time() - t0,3))
```

    numpy time:
    5.261
    3.024

``` python
import jax.config
jax.config.update("jax_enable_x64", True)  

x_jax = jnp.array(x, dtype = jnp.float64)
print(f"JAX dtype is {x_jax.dtype}")

t0 = time.time()
mat_jax = jnp.matmul(x_jax.transpose(), x_jax)
print("jax time:")
print(round(time.time() - t0,3))

t0 = time.time()
U_jax = jnp.linalg.cholesky(mat_jax)
print(round(time.time() - t0,3))
```

    JAX dtype is float64
    jax time:
    8.288
    1.835

So here the matrix multiplication is slower using JAX with 64-bit
numbers but the Cholesky is a bit faster. If one uses 32-bit numbers,
JAX is faster for both (not shown).

In general, the JAX speedups are not huge, which is not surprising given
both approaches are using multiple threads to carry out the linear
algebra. At the least it indicates one can move a numpy workflow to JAX
without worrying about losing the threaded BLAS speed of numpy.

### 7.3 Using the GPU with JAX

Getting threaded CPU computation automatically is nice, but the real
benefit of JAX comes in offloading computations to the GPU (and in
providing automatic differentiation, not discussed in this tutorial). If
a GPU is available and a [GPU-enabled JAX is
installed](https://jax.readthedocs.io/en/latest/installation.html), JAX
will generally try to use the GPU.

Note my general comments about using the GPU in the [PyTorch
section](./#6-using-the-gpu-via-pytorch).

Note that for this section, I’m pasting in the output when running the
code separately on a machine with a GPU because this document is
generated on a machine without a GPU.

We’ll just repeat the experiments we ran earlier comparing numpy- and
JAX-based calculations, but on a machine with an A100 GPU.

``` python
import time
import numpy as np
import jax.numpy as jnp

def myfun_np(x):
    y = np.exp(x) + 3 * np.sin(x)
    return(y)
    
def myfun_jnp(x):
    y = jnp.exp(x) + 3 * jnp.sin(x)
    return(y)

n = 250000000

x = np.random.normal(size = n).astype(np.float32)  # for consistency
x_jax = jnp.array(x)  # 32-bit by default
print(x_jax.platform())    # gpu

t0 = time.time()
z = myfun_np(x)
print(time.time() - t0)    # 1.15 s.

t0 = time.time()
z_jax = myfun_jnp(x_jax).block_until_ready()
print(time.time() - t0)    # 0.0099 s.
```

So that gives a speedup of more than 100x.

``` python
import jax
myfun_jnp_jit = jax.jit(myfun_jnp)

t0 = time.time()
z_jax_jit = myfun_jnp_jit(x_jax).block_until_ready()  # 0.0052 s.
print(time.time() - t0)
```

JIT compilation helps a bit (about 2x).

Finally, here’s the linear algebra example on the GPU.

``` python
n = 7000
x = np.random.normal(0, 1, size=(n, n)).astype(np.float32) # for consistency

t0 = time.time()
mat = x.T @ x
print(time.time() - t0)    # 3.7 s.

t0 = time.time()
U = np.linalg.cholesky(mat)  # 3.3 s.
print(time.time() - t0)
```

``` python
x_jax = jnp.array(x)

t0 = time.time()
mat_jax = jnp.matmul(x_jax.transpose(), x_jax).block_until_ready()
print(time.time() - t0)    # 0.025 sec.

t0 = time.time()
U_jax = jnp.linalg.cholesky(mat_jax).block_until_ready()
print(time.time() - t0)   # 0.08 s.
```

Again we get a very impressive speedup.

### 7.4 Some comments

As discussed elsewhere in this tutorial, it takes time to transfer data
to and from the GPU, so it’s best to generate values on the GPU and keep
objects on the GPU when possible.

Also, JAX objects are designed to be manipulated as objects, rather than
manipulating individual values.

``` python
x_jax[0,0] = 3.17
```

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/scratch/users/paciorek/conda/envs/jax-test/lib/python3.11/site-packages/jax/_src/numpy/array_methods.py", line 278, in _unimplemented_setitem
        raise TypeError(msg.format(type(self)))
    TypeError: '<class 'jaxlib.xla_extension.ArrayImpl'>' object does not support item assignment. JAX arrays are immutable. Instead of ``x[idx] = y``, use ``x = x.at[idx].set(y)`` or another .at[] method: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html

### 7.5 vmap and for vectorized map operations

We can use JAX’s `vmap` to automatically vectorize a map operation.
Unlike numpy’s `vectorize` or `apply_along_axis`, which are just handy
syntax (“syntactic sugar”) and don’t actually speed anything up (because
the looping is still done in Python), `vmap` actually vectorizes the
loop. Behind the scenes it generates a vectorized version of the code
that can run in parallel on CPU or GPU.

In general, one would use this to automatically iterate over the
dimension(s) of one or more arrays. This is convenient from a coding
perspective (compared to explicitly writing a loop) and potentially
speeds up the computation based on parallelization and by avoiding the
overhead of looping at the Python level.

Here we’ll standardize each column of an array using `vmap` rather than
writing a loop over the columns.

``` python
import jax
import jax.numpy as jnp
import time

nr = 10000
nc = 10000
x = np.random.normal(size = (nr,nc)).astype(np.float32)  # for consistency
x_jax = jnp.array(x) 

def f(x):
    ## Standardize a vector by its range.
    return x / (np.max(x) - np.min(x))

def f_jax(x):
    return x / (jnp.max(x) - jnp.min(x))

# Standardize each column.

t0 = time.time()
out = np.apply_along_axis(f, 0, x)  
t1 = time.time() - t0

# JAX vmap numbers axes in reverse order of numpy, apparently.
f_jax_vmap = jax.vmap(f_jax, in_axes=1, out_axes=1)

t0 = time.time()
out_jax = f_jax_vmap(x_jax).block_until_ready()     
t2 = time.time() - t0
print(f"numpy time: {round(t1,3)}\njax vmap time: {round(t2,3)}")
```

    numpy time: 3.654
    jax vmap time: 1.566

That gives a nice speedup. Let’s also try JIT’ing it. That gives a
further speedup.

``` python
f_jax_vmap_jit = jax.jit(f_jax_vmap)

t0 = time.time()
out_jax_jit = f_jax_vmap_jit(x_jax).block_until_ready()
t3 = time.time() - t0
print(f"jitted jax vmap time: {round(t3,3)}")
```

    jitted jax vmap time: 0.322

It would make sense to explore the benefits of using a GPU here, though
I haven’t done so.

`vmap` has a lot of flexibility to operate on various axes of its input
arguments (and structure the output axes). Suppose we want to do the
same standardization but using the columns of a different array as what
to standardize based on.

``` python
y = np.random.normal(size = (nr,nc)).astype(np.float32)
y_jax = jnp.array(y) 

def f2_jax(x, y):
    return x / (jnp.max(y) - jnp.min(y))

out2 = jax.vmap(f2_jax, in_axes=(1,1), out_axes=1)(x_jax, y_jax)
f2_jax_jit = jax.jit(jax.vmap(f2_jax, in_axes=(1,1), out_axes=1)) 
out3 = f2_jax_jit(x_jax, y_jax)
```

Finally, note that `pmap` is a function with a similar-sounding name
that allows one to parallelize a map operation over multiple devices
(e.g., multiple GPUs).

## 8 Using CuPy

CuPy is another package allowing one to execute numpy-type calculations
on the GPU (Nvidia only). It has some similarity to JAX.

Here’s a basic illustration, where we get a 175x speedup for generating
a random matrix and matrix multiplication when using an A100 GPU.

``` python
import cupy
import numpy as np
import time

def matmul_np(n):
    x = np.random.normal(size=(n,n))
    z = np.matmul(x,x)
    return(z)

def matmul_cupy(n):
    x = cupy.random.normal(size=(n,n))
    z = cupy.matmul(x,x)
    return(z)


n = 7000

t0 = time.time()
z = matmul_np(n)
print(time.time() - t0)   # 8.8 s.

t0 = time.time()
z_cupy = matmul_cupy(n)
cupy.cuda.stream.get_current_stream().synchronize()
print(time.time() - t0)   # .05 s.
```

You can also use `cupy.RawKernel` to execute a GPU kernel written in
CUDA C/C++ directly from Python. That’s a bit beyond our scope here, so
I won’t show an example.
