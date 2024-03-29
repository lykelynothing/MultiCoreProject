#ifndef COLLECTIVES_H
#define COLLECTIVES_H

#include <mpi.h>

int MPI_Allreduce(const void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

#endif
