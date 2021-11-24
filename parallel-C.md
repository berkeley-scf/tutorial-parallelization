---
layout: default
title: Parallel processing in C/C++
---

# 1 Overview

Some long-standing tools for parallelizing C, C++, and Fortran code are `openMP` for writing threaded code to run in parallel on one machine and `MPI` for writing code that passages message to run in parallel across (usually) multiple nodes.

# 2 Using *OpenMP* threads for basic shared memory programming in C

It's straightforward to write threaded code in C and C++ (as well as Fortran) to exploit multiple cores. The basic approach is to use the OpenMP protocol. 

## 2.1 Basics of OpenMP

Here's how one would parallelize a loop in C/C++ using an OpenMP compiler directive. In this case we are parallelizing the outer loop; the iterations of the outer loop are done in parallel, while the iterations of the inner loop are done serially within a thread. As with *foreach* in R, you only want to do this if the iterations do not depend on each other. The code is available as a C++ program (but the core of the code is just C code) in *testOpenMP.cpp*.

```c
// see testOpenMP.cpp
#include <iostream>
using namespace std;

// compile with:  g++ -fopenmp -L/usr/local/lib  
//                  testOpenMP.cpp -o testOpenMP 

int main(){
  int nReps = 20;
  double x[nReps];
  #pragma omp parallel for
  for (int i=0; i<nReps; i++){
    x[i] = 0.0;
    for ( int j=0; j<1000000000; j++){
      x[i] = x[i] + 1.0;
    }
    cout << x[i] << endl;
  }
  return 0;
}
```

We would compile this program as follows

```bash
$ g++ -fopenmp testOpenMP.cpp -o testOpenMP
```

The main thing to be aware of in using OpenMP is not having different threads overwrite variables used by other threads. In the example above, variables declared within the `#pragma` directive will be recognized as variables that are private to each thread. In fact, you could declare `int i` before the compiler directive and things would be fine because OpenMP is smart enough to deal properly with the primary looping variable. But big problems would ensue if you had instead written the following code:

```c
int main(){
  int nReps = 20;
  int j;  // DON'T DO THIS !!!!!!!!!!!!!
  double x[nReps];
  #pragma omp parallel for
  for (int i=0; i<nReps; i++){
    x[i] = 0.0;
    for (j=0; j<1000000000; j++){
      x[i] = x[i] + 1.0;
    }
    cout << x[i] << endl;
  }
  return 0;
}
```

Note that we do want *x* declared before the compiler directive because we want all the threads to write to a common *x* (but, importantly, to different components of *x*). That's the point!

We can also be explicit about what is shared and what is private to each thread:

```c
int main(){
  int nReps = 20;
  int i, j;
  double x[nReps];
  #pragma omp parallel for private(i,j) shared(x, nReps)
  for (i=0; i<nReps; i++){
    x[i] = 0.0;
    for (j=0; j<1000000000; j++){
      x[i] = x[i] + 1.0;
    }
    cout << x[i] << endl;
  }
  return 0;
}
```

## 2.2 Calling OpenMP-based C code from R

The easiest path here is to use the *Rcpp* package. In this case, you can write your C++ code with OpenMP pragma statemetns as in the previous subsection. You'll need to make sure that the *PKG_CXXFLAGS* and *PKG_LIBS* environment variables are set to include `-f openmp` so the compilation is done correctly. More details/examples linked to from [this Stack overflow post](http://stackoverflow.com/questions/22748358/rcpp-with-openmp).

## 2.3 More advanced use of *OpenMP* in C

The goal here is just to give you a sense of what is possible with OpenMP. 

The OpenMP API provides three components: compiler directives that parallelize your code (such as `#pragma omp parallel for`), library functions (such as `omp_get_thread_num()`), and environment variables (such as `OMP_NUM_THREADS`)

OpenMP constructs apply to structured blocks of code. Blocks may be executed in parallel or sequentially, depending on how one uses the OpenMP pragma statements. One can also force execution of a block to wait until particular preceding blocks have finished, using a *barrier*. 

Here's a basic "Hello, world" example that illustrates how it works (the full program is in *helloWorldOpenMP.cpp*):

```c
// see helloWorldOpenMP.cpp
#include <stdio.h>
#include <omp.h> // needed when using any openMP functions 
//                               such as omp_get_thread_num()

void myFun(double *in, int id){
// this is the function that would presumably do the heavy computational stuff
}

int main()
{
   int nthreads, myID;
   double* input;
   /* make the values of nthreads and myid private to each thread */
   #pragma omp parallel private (nthreads, myID)
   { // beginning of block
      myID = omp_get_thread_num();
      printf("Hello, I am thread %d\n", myID);
      myFun(input, myID);  // do some computation on each thread
      /* only main node print the number of threads */
      if (myid == 0)
      {
         nthreads = omp_get_num_threads();
         printf("I'm the boss and control %i threads. How come they're in front of me?\n", nThreads);
      }
   } // end of block
   return 0;
} 
```

The *parallel* directive starts a team of threads, including the main thread, which is a member of the team and has thread number 0. The number of threads is determined in the following ways - here the first two options specify four threads:

1. #pragma omp parallel NUM_THREADS (4) // set 4 threads for this parallel block

2. omp_set_num_threads(4) // set four threads in general

3. the value of the OMP_NUM_THREADS environment variable

4. a default - usually the number of cores on the compute node

Note that in `#pragma omp parallel for`, there are actually two instructions, `parallel` starts a team of threads, and `for` farms out the iterations to the team. In our parallel for invocation, we could have done it more explicitly as:

```c
#pragma omp parallel
#pragma omp for
```

We can also explicitly distribute different chunks of code amongst different threads as seen here and in the full program in *sectionsOpenMP.cpp*.

```c
// see sectionsOpenMP.cpp
#pragma omp parallel // starts a new team of threads
{
   Work0(); // this function would be run by all threads. 
   #pragma omp sections // divides the team into sections 
   { 
      // everything herein is run only once. 
      #pragma omp section 
      { Work1(); } 
      #pragma omp section 
      { 
         Work2(); 
         Work3(); 
      } 
      #pragma omp section 
      { Work4(); } 
   }
} // implied barrier
```

Here Work1, {Work2 + Work3} and Work4 are done in parallel, but Work2 and Work3 are done in sequence (on a single thread).

If one wants to make sure that all of a parallized calculation is complete before any further code is executed you can insert `#pragma omp barrier`. 

Note that a `#pragma for` statement includes an implicit barrier as does the end of any block specified with `#pragma omp parallel`.

You can use `nowait` if you explicitly want to prevent threads from waiting at an implicit barrier: e.g., `#pragma omp parallel sections nowait` or `#pragma omp parallel for nowait`

One should be careful about multiple threads writing to the same variable at the same time (this is an example of a race condition). In the example below, if one doesn't have the `#pragma omp critical` directive two threads could read the current value of *result* at the same time and then sequentially write to *result* after incrementing their local copy, which would result in one of the increments being lost. A way to avoid this is with the *critical* directive (for single lines of code you can also use `atomic` instead of `critical`), as seen here and in the full program in *criticalOpenMP.cpp*:

```c
// see criticalOpenMP.cpp
double result = 0.0;
double tmp;
#pragma omp parallel for private (tmp, i) shared (result)
for (int i=0; i<n; i++){
   tmp = myFun(i);
   #pragma omp critical
   result += tmp;
}
```

You should also be able to use syntax like the following for the parallel for declaration (in which case you shouldn't need the `#pragma omp critical`):

```c
#pragma omp parallel for reduction(+:result)
```

I believe that doing this sort of calculation where multiple threads write to the same variable may be rather inefficient given time lost in waiting to have access to result, but presumably this would depend on how much time is spent in *myFun()* relative to the reduction operation.

# 3 MPI

## 3.1 MPI Overview

There are multiple MPI implementations, of which *openMPI* and
*mpich* are very common. *openMPI* is quite common, and we'll use that.

In MPI programming, the same code runs on all the machines. This is
called SPMD (single program, multiple data). As we saw a bit with the pbdR code, one
invokes the same code (same program) multiple times, but the behavior
of the code can be different based on querying the rank (ID) of the
process. Since MPI operates in a distributed fashion, any transfer
of information between processes must be done explicitly via send
and receive calls (e.g., *MPI_Send*, *MPI_Recv*, *MPI_Isend*,
and *MPI_Irecv*). (The ``MPI_'' is for C code; C++ just has
*Send*, *Recv*, etc.)

The latter two of these functions (*MPI_Isend* and *MPI_Irecv*)
are so-called non-blocking calls. One important concept to understand
is the difference between blocking and non-blocking calls. Blocking
calls wait until the call finishes, while non-blocking calls return
and allow the code to continue. Non-blocking calls can be more efficient,
but can lead to problems with synchronization between processes. 

In addition to send and receive calls to transfer to and from specific
processes, there are calls that send out data to all processes (*MPI_Scatter*),
gather data back (*MPI_Gather*) and perform reduction operations
(*MPI_Reduce*).

Debugging MPI code can be tricky because communication
can hang, error messages from the workers may not be seen or readily
accessible, and it can be difficult to assess the state of the worker
processes. 

## 3.2 Basic syntax for MPI in C


Here's a basic hello world example  The code is also in *mpiHello.c*.

```c
// see mpiHello.c
#include <stdio.h> 
#include <math.h> 
#include <mpi.h>

int main(int argc, char* argv) {     
	int myrank, nprocs, namelen;     
	char process_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(&argc, &argv);     
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);   
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);          
	MPI_Get_processor_name(process_name, &namelen);            
	printf("Hello from process %d of %d on %s\n", 
		myrank, nprocs, process_name);
    MPI_Finalize();     
	return 0; 
} 
```

There are C (*mpicc*) and C++ (*mpic++*) compilers for MPI programs (*mpicxx* and *mpiCC* are synonyms).
I'll use the MPI C++ compiler
even though the code is all plain C code.

Then we'll run the executable via `mpirun`. Here the code will just run on my single machine, called `arwen`.
See Section 3.3 for details on how to run on multiple machines.


```bash
mpicxx mpiHello.c -o mpiHello
mpirun -np 4 mpiHello
```

Here's the output we would expect:

```
## Hello from processor 0 of 4 on arwen
## Hello from processor 1 of 4 on arwen
## Hello from processor 2 of 4 on arwen
## Hello from processor 3 of 4 on arwen
```

To actually write real MPI code, you'll need to go learn some of the
MPI syntax. See *quad_mpi.c* and *quad_mpi.cpp*, which
are example C and C++ programs (for approximating an integral via
quadrature) that show some of the basic MPI functions. Compilation
and running are as above:

```bash
mpicxx quad_mpi.cpp -o quad_mpi
mpirun -machinefile .hosts -np 4 quad_mpi
```

And here's the output we would expect:

```
23 November 2021 03:28:25 PM

QUAD_MPI
  C++/MPI version
  Estimate an integral of f(x) from A to B.
  f(x) = 50 / (pi * ( 2500 * x * x + 1 ) )

  A = 0
  B = 10
  N = 999999999
  EXACT =       0.4993633810764567

  Use MPI to divide the computation among 4 total processes,
  of which one is the main process and does not do core computations.
  Process 1 contributed MY_TOTAL = 0.49809
  Process 2 contributed MY_TOTAL = 0.00095491
  Process 3 contributed MY_TOTAL = 0.000318308

  Estimate =       0.4993634591634721
  Error = 7.808701535383378e-08
  Time = 10.03146505355835
  Process 2 contributed MY_TOTAL = 0.00095491

QUAD_MPI:
  Normal end of execution.

23 November 2021 03:28:36 PM
```

## 3.3 Starting MPI-based jobs

MPI-based executables require that you start your process(es) in a special way via the *mpirun* command. Note that *mpirun*, *mpiexec* and *orterun* are synonyms under *openMPI*. 

The basic requirements for starting such a job are that you specify the number of processes you want to run and that you indicate what machines those processes should run on. Those machines should be networked together such that MPI can ssh to the various machines without any password required.

### 3.3.1 Running an MPI job with machines specified manually


There are two ways to tell *mpirun* the machines on which to run the worker processes.

First, we can pass the machine names directly, replicating the name
if we want multiple processes on a single machine. In the example here, these are machines accessible to me, and you would need to replace those names with the names of machines you have access to. You'll need to [set up SSH keys](http://statistics.berkeley.edu/computing/sshkeys) so that you can access the machines without a password.


```bash
mpirun --host gandalf,radagast,arwen,arwen -np 4 hostname
```

Alternatively, we can create a file with the relevant information.

```bash
echo 'gandalf slots=1' > .hosts
echo 'radagast slots=1' >> .hosts
echo 'arwen slots=2' >> .hosts
mpirun -machinefile .hosts -np 4 hostname
```

One can also just duplicate a given machine name as many times as desired, rather than using `slots`.

### 3.3.2 Running an MPI job within a Slurm job

If you are running your code as part of a job submitted to Slurm, you generally won't need to pass the *machinefile* or *np* arguments as MPI will get that information from Slurm. So you can simply run your executable, in this case first checking which machines mpirun is using:

```bash
mpirun hostname
mpirun quad_mpi
```

### 3.3.3 Additional details

To limit the number of threads for each process, we can tell *mpirun*
to export the value of *OMP_NUM_THREADS* to the processes. E.g., calling a C program, *quad_mpi*:

```bash
export OMP_NUM_THREADS=2
mpirun -machinefile .hosts -np 4 -x OMP_NUM_THREADS quad_mpi
```

There are additional details involved in carefully controlling how processes are allocated to nodes, but the default arguments for mpirun should do a reasonable job in many situations. 


