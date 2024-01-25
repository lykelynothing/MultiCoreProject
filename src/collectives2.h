#ifndef "COLLECTIVES2_H"
#define "COLLECTIVES2_H"

#include <mpi.h>

/*int MPI_Allreduce_ring(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
*/
void ScatterUniform(struct unif_quant* in, size_t size, MPI_Comm comm);

#endif

