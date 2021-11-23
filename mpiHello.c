#include <stdio.h>
#include <math.h>
#include <mpi.h>

// mpicxx mpiHello.c -o mpiHello

int main(int argc, char** argv) {
    int myrank, nprocs, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);     
    MPI_Get_processor_name(processor_name, &namelen);       
    printf("Hello from processor %d of %d on %s\n", myrank, nprocs, processor_name);

    MPI_Finalize();
    return 0;
}
