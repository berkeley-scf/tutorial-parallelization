---
layout: default
title: Parallel processing in Python
---

# 1. Threading, particularly for linear algebra

# 2.1) What is the BLAS?

The BLAS is the library of basic linear algebra operations (written in
Fortran or C). A fast BLAS can greatly speed up linear algebra
relative to the default BLAS on a machine. Some fast BLAS libraries
are 
 - Intel's *MKL*; may be available for educational use for free
 - *OpenBLAS* (formerly *GotoBLAS*); open source and free
 - AMD's *ACML*; free
 - *vecLib* for Macs; provided with your Mac


In addition to being fast when used on a single core, all of these BLAS libraries are
threaded - if your computer has multiple cores and there are free
resources, your linear algebra will use multiple cores, provided your
program is linked against the threaded BLAS installed on your machine and provided
OMP_NUM_THREADS is not set to one. (Macs make use of
VECLIB_MAXIMUM_THREADS rather than OMP_NUM_THREADS.)

On BCE, both R and (to some degree) Python are linked against OpenBLAS as of BCE-fall-2015. 


## 2.2) Using threading 


### 2.2.2) Python

As with R, threading in Python is limited to linear algebra, provided Python is linked against a threaded BLAS.  Python has something
called the [Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock)
that interferes with threading in Python (but not in threaded linear algebra packages called by Python). 

Here's some linear algebra in  Python that will use threading if *numpy* is linked against a threaded BLAS, though I don't compare the timing for different numbers of threads here. 

```{r, py-linalg, engine='python', eval=FALSE}
```


## 2.3) Fixing the number of threads (cores used)

In general, threaded code will
detect the number of cores available on a machine and make use of
them. However, you can also explicitly control the number of threads
available to a process. 

### 2.3.1) R, Python, C, etc.

For most threaded code (that based on the openMP protocol), the number
of threads can be set by setting the OMP_NUM_THREADS environment
variable (VECLIB_MAXIMUM_THREADS on a Mac). E.g., to set it for four
threads in bash:

```export OMP_NUM_THREADS=4```

Do this before starting your R or Python session or before running your compiled executable. 

Alternatively, you can set OMP_NUM_THREADS as you invoke your job, e.g., here with R:

```OMP_NUM_THREADS=4 R CMD BATCH --no-save job.R job.out```



## 2.4) Important warnings about use of threaded BLAS

### 2.4.1) Speed and threaded BLAS

In many cases, using multiple threads for linear algebra operations
will outperform using a single thread, but there is no guarantee that
this will be the case, in particular for operations with small matrices
and vectors. Testing with openBLAS suggests that sometimes a job may
take more time when using multiple threads; this seems to be less
likely with ACML. This presumably occurs because openBLAS is not doing
a good job in detecting when the overhead of threading outweights
the gains from distributing the computations. You can compare speeds
by setting OMP_NUM_THREADS to different values. In cases where threaded
linear algebra is slower than unthreaded, you would want to set OMP_NUM_THREADS
to 1. 

More generally, if you are using the parallel tools in Section 3 to 
simultaneously carry out many independent calculations (tasks), it is
likely to be more effective to use the fixed number of cores available on your machine
 so as to split up the tasks, one per core, without taking advantage of the threaded BLAS (i.e., restricting
each process to a single thread). 




## 2.5) Using an optimized BLAS on your own machine(s)

To use an optimized BLAS with R, talk to your systems administrator, see [Section A.3 of the R Installation and Administration Manual](https://cran.r-project.org/manuals.html), or see [these instructions to use *vecLib* BLAS from Apple's Accelerate framework on your own Mac](http://statistics.berkeley.edu/computing/blas) or see [these instructions for Windows](https://www.r-bloggers.com/building-r-4-for-windows-with-openblas/).

It's also possible to use an optimized BLAS with Python's numpy and scipy packages, on either Linux or using the Mac's *vecLib* BLAS. Details will depend on how you install Python, numpy, and scipy. 

 
# 3) Basic parallelized loops/maps/apply

All of the functionality discussed here applies *only* if the iterations/loops of your calculations can be done completely separately and do not depend on one another. This scenario is called an *embarrassingly parallel* computation.  So coding up the evolution of a time series or a Markov chain is not possible using these tools. However, bootstrapping, random forests, simulation studies, cross-validation
and many other statistical methods can be handled in this way.


## 3.2) Parallel looping in Python

I'll cover IPython parallel functionality, which allows one to parallelize on a single machine (discussed here) or across multiple machines (see the tutorial on distributed memory parallelization). There are a variety of other approaches one could use, of which I discuss two (the *pp* and *multiprocessing* packages) in the Appendix.

First we need to start our worker engines.

```{r, ipython-parallel-setup, engine='bash', eval=FALSE}
ipcluster start -n 4 &
sleep 45
```

Here we'll do the same random forest cross-validation as before. 

```{r, ipython-parallel, engine='python', eval=FALSE}
```

Finally we stop the worker engines:

```{r, ipython-parallel-shutdown, engine='bash', eval=FALSE}
ipcluster stop
```




# 5) Random number generation (RNG) in parallel 

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
e.g., using *set.seed(mySeed)* in R on every process.

The naive approach is to use a different seed for each process. E.g.,
if your processes are numbered `id = 1,2,...,p`  with a variable *id* that is  unique
to a process, setting the seed to be the value of *id* on each process. This is likely
not to cause problems, but raises the danger that two (or more sequences)
might overlap. For an algorithm with dependence on the full sequence,
such as an MCMC, this probably won't cause big problems (though you
likely wouldn't know if it did), but for something like simple simulation
studies, some of your 'independent' samples could be exact replicates
of a sample on another process. Given the period length of the default
generators in R, MATLAB and Python, this is actually quite unlikely,
but it is a bit sloppy.

One approach to avoid the problem is to do all your RNG on one process
and distribute the random deviates, but this can be infeasible with
many random numbers.

More generally to avoid this problem, the key is to use an algorithm
that ensures sequences that do not overlap.

## 5.2) RNG in Python

Python uses the Mersenne-Twister generator. If you're using the RNG
in *numpy/scipy*, you can set the seed using `numpy.random.seed` or `scipy.random.seed`.
The advice I'm seeing online in various Python forums is to just set
separate seeds, so it appears the Python is a bit behind R and MATLAB here.
 There is a function *random.jumpahead* that
allows you to move the seed ahead as if a given number of random numbers
had been generated, but this function will not be in Python 3.x, so
I won't suggest using it. 




## 3.2) Python

### 3.2.1) IPython parallel

One can use IPython's parallelization tools in a context with multiple nodes, though the setup to get the worker processes is a bit more involved when you have multiple nodes. For details on using IPython parallel on a single node, see the [parallel basics tutorial appendix](https://github.com/berkeley-scf/tutorial-parallel-basics). 

If we are using the SLURM scheduling software, here's how we start up the worker processes:

```{r, ipyparallel-setup, engine='bash', eval=FALSE}
ipcontroller --ip='*' &
sleep 25
# next line will start as many ipengines as we have SLURM tasks 
#   because srun is a SLURM command
srun ipengine &  
sleep 45  # wait until all engines have successfully started
```


We can then run IPython to split up our computational tasks across the engines.

```{r, ipyparallel, engine='python', eval=FALSE}
```

To finish up, we need to shut down the cluster of workers:
```{r, engine='bash', eval=FALSE}
ipcluster stop
```

To start the engines in a context outside of using slurm (provided all machines share a filesystem), you should be able ssh to each machine and run `ipengine &` for as many worker processes as you want to start as follows. In some, but not all cases (depending on how the network is set up) you may not need the `--location` flag. 

```{r, ipyparallel-setup2, engine='bash', eval=FALSE}
ipcontroller --ip='*' --location=URL_OF_THIS_MACHINE &
sleep 25
nengines=8
ssh other_host "for (( i = 0; i < ${nengines}; i++ )); do ipengine & done"
sleep 45  # wait until all engines have successfully started
```

### 3.2.2) *pp* package

Another way to parallelize across multiple nodes that uses more manual setup and doesn't integrate as well with scheduling software like SLURM is to use the pp package (also useful for parallelizing on a single machine as discussed in the [parallel basics tutorial appendix](https://github.com/berkeley-scf/tutorial-parallel-basics). 

Assuming that the pp package is installed on each node (e.g., `sudo apt-get install python-pp` on an Ubuntu machine), you need to start up a ppserver process on each node. E.g., if `$nodes` is a UNIX environment variable containing the names of the worker nodes and you want to start 2 workers per node:

```{r, pp-start, engine='bash', eval=FALSE}
nodes='smeagol radagast beren arwen'
for node in $nodes; do
# cd /tmp is because of issue with starting ppserver in home directory
# -w says how many workers to start on the node
    ssh $node "cd /tmp && ppserver -s mysecretphrase -t 120 -w 2 &" & 
done
```

Now in our Python code we create a server object and submit jobs to the server object, which manages the farming out of the tasks. Note that this will run interactively in IPython or as a script from UNIX, but there have been times where I was not able to run it interactively in the base Python interpreter. Also note that while we are illustrating this as basically another parallelized for loop, the individual jobs can be whatever calculations you want, so the  function (in this case it's always *pi.sample*) could change from job to job.

```{r, python-pp, engine='python', eval=FALSE}
```

```{r, python-pp-example, engine='bash', eval=FALSE}
python python-pp.py > python-pp.out
cat python-pp.out
```

```
['smeagol', 'radagast', 'beren', 'arwen', 'smeagol', 'radagast', 'beren', 'arwen']
Pi is roughly 3.141567
Time elapsed:  32.0389587879
```

The -t flag used when starting ppserver should ensure that the server processes are removed, but if you need to do it manually, this should work:

```{r, pp-stop, engine='bash', eval=FALSE}
for node in $nodes; do
    killall ppserver
done
```


## Other approaches to parallelization in Python

###  *pp* package

Here we create a server object and submit jobs to the server object,
which manages the farming out of the tasks. Note that this will run
interactively in IPython or as a script from UNIX, but will not run
interactively in the base Python interpreter (for reasons that are
unclear to me). Also note that while we are illustrating this as basically
another parallelized for loop, the individual jobs can be whatever
calculations you want, so the *taskFun* function could change from
job to job.

```{r, python-pp, engine='python', eval=FALSE}
```

```
[(0, 0.00030280243091597474), (1, -8.5722825540149767e-05), (2, 0.00013566614947237407), (3, 0.00061310818505479474), (4, -0.0004702706491795272), (5, 0.00024515486966970537), (6, -0.00017472300458822845), (7, -0.00025050095623507584), (8, -0.00033399772183492841), (9, -0.00049137138871004158), (10, 0.00029251318047107422), (11, 1.1956375483643322e-05), (12, -0.00010810414999124078), (13, 0.00015533121727716678), (14, -0.00092143784872822018), (15, -7.4020047531168942e-05), (16, -0.00027179897723462343), (17, -0.00020500359099047446), (18, 5.0102720605584639e-05), (19, -0.00031948846032527046), (20, -5.4961570167677311e-05), (21, -0.00057477384497516828), (22, 0.00035571929916218195), (23, 0.0003172760600221845), (24, -3.9757431343687736e-05), (25, 0.00037903275195759294), (26, -0.00010435497860874407), (27, 0.0001701626336006962), (28, -0.00069358450543517865), (29, 0.00067886194920371693), (30, -0.00051310981441539557), (31, -3.0022955955111069e-05), (32, -0.00063590672702952002), (33, -0.00031966078315322541), (34, -0.00015649509164027703), (35, 0.00028376009875884391), (36, 0.00018534703816611961), (37, -0.00021821998172858934), (38, 8.0842394421238762e-05), (39, -0.00014637870897851111)]
```

### *multiprocessing* package

Here we'll use the *Pool.map* method
to iterate in a parallelized fashion, as the Python analog to *foreach*
or *parfor*. *Pool.map* only supports having a single
argument to the function being used, so we'll use list of tuples,
and pass each tuple as the argument. 

```{r, python-mp, engine='python', eval=TRUE}
```


