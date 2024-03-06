#include <mpi.h>
#include <stdlib.h>

#include "homomorphic_quantizer.h"
#include "tools.h"
#include "tools_collectives.h"

/* Each active node will receive the entire vector of
 * quantized data, dequantize it, sum it with its own vector,
 * requantize the partial sum vector and send it to the next process.
 * For now assumes comm_sz is always divisible by 2. */
int RecursiveHalvingSend(int my_rank, int comm_sz, int dim, QUANT algo,
                         float *my_numbers, float *recv_buf) {
  int remaining = comm_sz;
  int half;
  // used to store quantized received bits
  void *struct_ptr = Allocate(algo, dim);

  float *dequantized = malloc(sizeof(float) * dim);
  float *partial_sum = my_numbers;
  int sent = 0;

  // need to send struct as well
  while (remaining != 1) {
    half = remaining / 2;
    if (my_rank < half) {
      int source = half + my_rank;
      // receive struct and quantized array

      Receive(algo, dim, source, struct_ptr);

      // all will now be contained inside struct_ptr
      DequantizeVector(struct_ptr, dequantized, algo, dim);
      // sum my array with received dequantized one
      for (int i = 0; i < dim; i++)
        partial_sum[i] += dequantized[i];
    } else if (sent == 0) {
      int dest = my_rank % half;
      // first quantize
      Quantize(partial_sum, dim, algo, struct_ptr);
      // now quantize uses same struct_ptr
      Send(struct_ptr, algo, dim, dest);
      sent = 1;
    }
    remaining = remaining / 2;
  }
  if (my_rank == 0) {
    for (int i = 0; i < dim; i++)
      recv_buf[i] = my_numbers[i];
  }
  PMPI_Bcast(recv_buf, dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

  Free(algo, struct_ptr);
  free(dequantized);
  return MPI_SUCCESS;
}

/* Like Normal Recursive halving but this time the quantization and
 * dequantization only happen once i.e. before and after all reductions
 * respectively have been executed (uses homomorphic quant algo)*/
int RecursiveHalvingSendHomomorphic(int my_rank, int comm_sz, int count,
                                    QUANT algo, float *sendbuf,
                                    float **recvbuf) {
  int remaining = comm_sz;
  int half;

  void *void_ptr = Allocate(algo, count);
  void_ptr = Quantize(sendbuf, count, algo, void_ptr);
  struct unif_quant *struct_ptr = (struct unif_quant *)void_ptr;

  void *tmp_bf = Allocate(algo, count);
  struct unif_quant *rcv_bf = (struct unif_quant *)tmp_bf;

  int sent = 0;
  while (remaining != 1) {
    half = remaining / 2;
    if (my_rank < half) {
      int source = half + my_rank;
      Receive(algo, count, source, rcv_bf);
      for (int i = 0; i < count; i++) {
        struct_ptr->vec[i] += rcv_bf->vec[i];
      }
    } else if (sent == 0) {
      int dest = my_rank % half;
      Send(struct_ptr, algo, count, dest);
      sent = 1;
    }
    remaining = remaining / 2;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Bcast(struct_ptr->vec, count, MPI_UINT8_T, 0, MPI_COMM_WORLD);

  // TODO: use deallocate
  *recvbuf =
      HomomorphicDequantization(struct_ptr->vec, struct_ptr->min,
                                struct_ptr->max, comm_sz, count, 1, *recvbuf);

  Free(algo, struct_ptr);
  Free(algo, tmp_bf);

  return MPI_SUCCESS;
}

/* Same as before but for for uint16*/
int RecursiveHalvingSendHomomorphic_16(int my_rank, int comm_sz, int count,
                                       QUANT algo, float *sendbuf,
                                       float **recvbuf) {
  int remaining = comm_sz;
  int half;

  void *void_ptr = Allocate(algo, count);
  void_ptr = Quantize(sendbuf, count, algo, void_ptr);
  struct unif_quant_16 *struct_ptr = (struct unif_quant_16 *)void_ptr;

  void *tmp_bf = Allocate(algo, count);
  struct unif_quant_16 *rcv_bf = (struct unif_quant_16 *)tmp_bf;

  int sent = 0;
  while (remaining != 1) {
    half = remaining / 2;
    if (my_rank < half) {
      int source = half + my_rank;
      Receive(algo, count, source, rcv_bf);
      for (int i = 0; i < count; i++) {
        struct_ptr->vec[i] += rcv_bf->vec[i];
      }
    } else if (sent == 0) {
      int dest = my_rank % half;
      Send(struct_ptr, algo, count, dest);
      sent = 1;
    }
    remaining = remaining / 2;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Bcast(struct_ptr->vec, count, MPI_UINT16_T, 0, MPI_COMM_WORLD);

  // TODO: use deallocate
  *recvbuf = HomomorphicDequantization_16(struct_ptr->vec, struct_ptr->min,
                                          struct_ptr->max, comm_sz, count, 1,
                                          *recvbuf);

  Free(algo, struct_ptr);
  Free(algo, tmp_bf);

  return MPI_SUCCESS;
}
