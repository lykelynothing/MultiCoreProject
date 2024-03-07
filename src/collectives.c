#include <mpi.h>

#include "recursive_halving_reduce.h"
#include "ring_reduce.h"
#include "tools.h"

/* Custom MPI_Allreduce that will intercept any calls to it.
 * Will look for the environment variable "QUANT_ALGO" and choose
 * the quantization algorithm and the send structure accordingly
 * (if QUANT_ALGO = NON_LINEAR then also environment variable
 * NON_LINEAR_TYPE will be checked).
 * All reduction calls are put in return for error handling (that
 * we do not really do, but is partially set up)*/
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  // Environmental variables are check and put into an enum type object
  //(defined in tools.h), and passed as parameters for reduction functions
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
