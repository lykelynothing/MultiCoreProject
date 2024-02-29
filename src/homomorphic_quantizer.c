#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#include "tools.h"

/* Work exactly like a normal Uniform Ranged Quantization but here the
 * quantization interval is shared between all the processes. To allow for sum
 * of quantized values, we need to make the REPR_RANGE smaller. The this is done
 * by dividing the REPR_RANGE by pow=2^n such that >=comm_sz */
struct unif_quant *HomomorphicQuantization(float *input, size_t input_size,
                                           MPI_Comm comm, void *struct_ptr) {
  int my_rank, comm_sz;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_sz);
  struct unif_quant *out = (struct unif_quant *)struct_ptr;
  float min_max[2];

  MinMax(input, input_size, &min_max[0], &min_max[1], 1);
  // Standard call of MPI_Allreduce with MPI_Max on an array of -min and max to
  // get the smallest min and biggest between the vectors
  PMPI_Allreduce(MPI_IN_PLACE, min_max, 2, MPI_FLOAT, MPI_MAX, comm);

  out->min = -min_max[0];
  out->max = min_max[1];
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
struct unif_quant_16 *HomomorphicQuantization_16(float *input,
                                                 size_t input_size,
                                                 MPI_Comm comm,
                                                 void *struct_ptr) {
  int my_rank, comm_sz;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_sz);
  struct unif_quant_16 *out = (struct unif_quant_16 *)struct_ptr;
  float min_max[2];
  MinMax(input, input_size, &min_max[0], &min_max[1], 1);
  PMPI_Allreduce(MPI_IN_PLACE, min_max, 2, MPI_FLOAT, MPI_MAX, comm);
  out->min = -min_max[0];
  out->max = min_max[1];
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

/* For the dequantization step, the new minimum will be
 * comm_sz*quantization_minimum because this dequantization will be done on
 * reduced data (hence each element will be the sum of comm_sz quantized
 * elements).*/
float *HomomorphicDequantization(uint8_t *quantized, float min, float max,
                                 int comm_sz, size_t input_size,
                                 int reduction_flag, float *out) {
  // same custom REPR_RANGE used in quantization
  int pow = 1;
  while (pow < comm_sz) {
    pow <<= 1;
  }
  float hom_repr_range = REPR_RANGE / pow;
  float range = max - min;
  float step = range / (hom_repr_range - 1);

  // new min due to reduction
  float new_min;
  if (reduction_flag == 1)
    new_min = comm_sz * min;
  else
    new_min = min;

#pragma omp parallel for
  for (size_t i = 0; i < input_size; i++)
    out[i] = ((float)quantized[i]) * step + new_min;

  return out;
}

float *HomomorphicDequantization_16(uint16_t *quantized, float min, float max,
                                    int comm_sz, size_t input_size,
                                    int reduction_flag, float *out) {
  int pow = 1;
  while (pow < comm_sz) {
    pow <<= 1;
  }
  float hom_repr_range = REPR_RANGE / (float)pow;
  float range = max - min;
  float step = range / (hom_repr_range - 1);
  float new_min;
  if (reduction_flag == 1)
    new_min = comm_sz * min;
  else
    new_min = min;
#pragma omp parallel for
  for (size_t i = 0; i < input_size; i++)
    out[i] = ((float)quantized[i]) * step + new_min;
  return out;
}
