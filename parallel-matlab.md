---
layout: default
title: Parallel processing in MATLAB
---

# Parallel processing in MATLAB

## 1 Overview

MATLAB provides a variety of functionality for parallelization, including threaded operations (linear algebra and more), parallel for loops, and parallelization across multiple machines. This material will just scratch the surface of what is possible.

## 2 Threading

Many MATLAB functions are automatically threaded (not just linear
algebra), so you don't need to do anything special in your code to
take advantage of this. 
So if you're running MATLAB and monitoring CPU usage (e.g., using
*top* on Linux or OS X), you may see a process using more than 100% of CPU. 

However
worker tasks within a *parfor* use only a single thread. 

MATLAB uses MKL for (threaded) linear algebra. 

Threading in MATLAB can be controlled
in two ways. From within your MATLAB code you can set the number of
threads, e.g., to four in this case:

```matlab
maxNumCompThreads(4);
```

To use only a single thread, you can use 1 instead of 4 above, or
you can start MATLAB with the *singleCompThread* flag:

```bash
matlab -singleCompThread ...
```


## 3 Parallel for loops on one machine

To run a loop in parallel in MATLAB, you can use the *parfor*
construction.  Before running the parfor you
need to start up a set of workers using *parpool*. MATLAB will use only  one thread
per worker. 
Here is some demo code:

```matlab
nslots = 4; % to set manually
mypool = parpool(nslots) 

n = 3000;
nIts = 500;
c = zeros(n, nIts);
parfor i = 1:nIts
     c(:,i) = eig(rand(n)); 
end

delete(mypool)

% delete(gcp) works if you forget to name your pool by assigning the output of parpool to a variable
```

MATLAB has a default limit on the number of workers in a pool, but you can modify your MATLAB settings  as follows to increase that limit (in this case to allow up to 32 workers). It should work to run the following code once in a MATLAB session, which will modify the settings for future MATLAB sessions.

```
cl = parcluster();
cl.NumWorkers = 32;
saveProfile(cl);
```

By default MATLAB uses a single thread (core) per worker. However you can also use multiple threads. Here is how you can set that up.

```
cl = parcluster();
cl.NumThreads = 2;  % 2 threads per worker
cl.parpool(4);      % 4 workers
```

## 4 Parallel random number generation

MATLAB also uses the Mersenne-Twister. We can set the seed as: `rng(seed)`,
with seed being a non-negative integer. 

Happily, like in R, we can set up independent streams, using either of
the Combined Multiple Recursive ('mrg32k3a') and the Multiplicative
Lagged Fibonacci ('mlfg6331_64') generators. Here's an example, where
we create the second of the 5 streams, as if we were using this code
in the second of our parallel processes. The `'Seed',0` part
is not actually needed as that is the default.

```
thisStream = 2;
totalNumStreams = 5;
seed = 0;
cmrg1 = RandStream.create('mrg32k3a', 'NumStreams', totalNumStreams, 
   'StreamIndices', thisStream, 'Seed', seed); 
RandStream.setGlobalStream(cmrg1);
randn(5, 1)
```


## 5 Manually parallelizing individual tasks

In addition to using *parfor* in MATLAB, you can also explicitly program parallelization, managing the individual
parallelized tasks. Here is some template code for doing this. We'll
submit our jobs to a pool of workers so that we have control over
how many jobs are running at once. Note that here I submit 6 jobs
that call the same function, but the different jobs could call different
functions and have varying inputs and outputs. MATLAB will run as
many jobs as available workers in the pool and will queue the remainder,
starting them as workers in the pool become available. Here is
some demo code

```matlab
feature('numThreads', 1); 
ncores = 4;
pool = parpool(ncores); 
% assume you have test.m with a function, test, taking two inputs 
% (n and seed) and returning 1 output
n = 10000000;
job = cell(1,6); 
job{1} = parfeval(pool, @test, 1, n, 1);  
job{2} = parfeval(pool, @test, 1, n, 2);  
job{3} = parfeval(pool, @test, 1, n, 3);  
job{4} = parfeval(pool, @test, 1, n, 4);  
job{5} = parfeval(pool, @test, 1, n, 5);  
job{6} = parfeval(pool, @test, 1, n, 6);  

% wait for outputs, in order
output = cell(1, 6);
for idx = 1:6
  output{idx} = fetchOutputs(job{idx});
end 

% alternative way to loop over jobs:
for idx = 1:6
  jobs(idx) = parfeval(pool, @test, 1, n, idx); 
end 

% wait for outputs as they finish
output = cell(1, 6);
for idx = 1:6
  [completedIdx, value] = fetchNext(jobs);
  output{completedIdx} = value;
end 

delete(pool);
```

And if you want to run threaded code in a given job, you can do that
by setting the number of threads within the function called by *parfeval*.
See the *testThread.m* file  for the *testThread*
function.

```matlab
ncores = 8;
n = 5000;
nJobs = 4;
pool = parpool(nJobs);
% pass number of threads as number of slots divided by number of jobs
% testThread function should then do: 
% feature('numThreads', nThreads);
% where nThreads is the name of the relevant argument to testThread
jobt1 = parfeval(pool, @testThread, 1, n, 1, nCores/nJobs);
jobt2 = parfeval(pool, @testThread, 1, n, 2, nCores/nJobs);
jobt3 = parfeval(pool, @testThread, 1, n, 3, nCores/nJobs);
jobt4 = parfeval(pool, @testThread, 1, n, 4, nCores/nJobs);

output1 = fetchOutputs(jobt1);
output2 = fetchOutputs(jobt2);
output3 = fetchOutputs(jobt3);
output4 = fetchOutputs(jobt4);

delete(pool);
```


## 6 Using MATLAB across multiple nodes

To use MATLAB across multiple nodes, you need to have the MATLAB Parallel Server, which often requires an additional license. If it is installed, one can set up MATLAB so that *parfor* will distribute its work across multiple nodes. Details may vary depending on how Parallel Server is set up on your system. 

