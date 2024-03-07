#include <mpi.h>

#include "recursive_halving_reduce.h"
#include "ring_reduce.h"
#include "tools.h"

/* Custom MPI_Allreduce that will intercept any calls to it.
 * Will look for the environment variable "QUANT_ALGO" and choose
 * the quantization algorithm accordingly (if QUANT_ALGO = NON_LINEAR
 * then also environment variable NON_LINEAR_TYPE will be checked).
 * Once the sendbuf is quantized, it executes a normal Allreduce collective
 * through PMPI_Allreduce.*/
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  SEND send_algo;
  QUANT quant_algo;
  GetEnvVariables(&send_algo, &quant_algo);
  int my_rank, comm_sz;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_sz);

  switch (send_algo) {
  case REC_HALVING: {
    if (quant_algo != HOMOMORPHIC && quant_algo != KNOWN_RANGE)
      // this will handle automatically both BITS_VAR cases
      return RecursiveHalvingSend(my_rank, comm_sz, count, quant_algo,
                                  (float *)sendbuf, (float *)recvbuf);
    else {
      if (BITS == 16)
        return RecursiveHalvingSendHomomorphic_16(my_rank, comm_sz, count,
                                                  quant_algo, (float *)sendbuf,
                                                  (float **)&(recvbuf));
      else if (BITS == 8)
        return RecursiveHalvingSendHomomorphic(my_rank, comm_sz, count,
                                               quant_algo, (float *)sendbuf,
                                               (float **)&(recvbuf));
    }
  }
  case RING: {
    if (BITS == 16)
      return RingAllreduce_16(my_rank, comm_sz, (float *)sendbuf, count,
                              (float **)&(recvbuf), quant_algo);
    else if (BITS == 8)
      return RingAllreduce(my_rank, comm_sz, (float *)sendbuf, count,
                           (float **)&(recvbuf), quant_algo);
  }
  default: {
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  }
  }
}
