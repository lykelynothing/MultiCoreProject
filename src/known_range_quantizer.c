#include <math.h>
#include <mpi.h>
#include <omp.h>

#include "tools.h"

/* Work exactly like a normal Homomorphic Quantization but here the
 * quantization interval is known in advance and equal for all processes */
struct unif_quant *KnownRangeQuantization(float *input, size_t input_size,
                                          MPI_Comm comm, void *struct_ptr) {
  int my_rank, comm_sz;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_sz);
  struct unif_quant *out = (struct unif_quant *)struct_ptr;

  // this is the default range set in main for the array generator
  out->min = -500;
  out->max = 500;
  // make the REPR_RANGE smaller to allow for reduction additions
  int pow = 1;
  while (pow < comm_sz) {
    pow <<= 1;
  }

  // calculate the steps of the quantization
  float hom_repr_range = REPR_RANGE / pow;
  float range = out->max - out->min;
  float min = out->min;
  float step = (hom_repr_range - 1) / range;
#pragma omp parallel for
  for (int i = 0; i < input_size; i++) {
    float quant = round((input[i] - min) * step);
    out->vec[i] = (uint8_t)quant;
  }
  return out;
}

// Same as before buf for UINT16
struct unif_quant_16 *KnownRangeQuantization_16(float *input, size_t input_size,
                                                MPI_Comm comm,
                                                void *struct_ptr) {
  int my_rank, comm_sz;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_sz);
  struct unif_quant_16 *out = (struct unif_quant_16 *)struct_ptr;

  out->min = -500;
  out->max = 500;

  int pow = 1;
  while (pow < comm_sz) {
    pow <<= 1;
  }

  float hom_repr_range = REPR_RANGE / pow;
  float range = out->max - out->min;
  float min = out->min;
  float step = (hom_repr_range - 1) / range;
#pragma omp parallel for
  for (int i = 0; i < input_size; i++) {
    float quant = round((input[i] - min) * step);
    out->vec[i] = (uint16_t)quant;
  }
  return out;
}
