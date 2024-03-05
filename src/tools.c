#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tools.h"

int BITS;
int REPR_RANGE;

/* Function used to get environmental variables that will be used for the choice
 * of quantization algorithm, quantization precision (number of bits used) and
 * type of reduction. */
void GetEnvVariables(SEND *send_algo, QUANT *quant_algo) {
  char *quant_env_var;
  char *send_algo_var;
  char *bits_env_var;

  bits_env_var = getenv("BITS_VAR");
  quant_env_var = getenv("QUANT_ALGO");
  send_algo_var = getenv("SEND_ALGO");

  // TODO: error handling when var isn't found
  if ((send_algo_var == NULL) | (quant_env_var == NULL) |
      (bits_env_var == NULL)) {
    printf("\nError : at least one environmental variable was not found \n");
    printf("BITS_VAR: %s\n", bits_env_var);
    printf("QUANT_ALGO: %s\n", send_algo_var);
    printf("SEND_ALGO: %s\n", quant_env_var);
    return;
  }

  if (strcmp(send_algo_var, "REC_HALVING") == 0)
    *send_algo = REC_HALVING;
  else if (strcmp(send_algo_var, "RING") == 0)
    *send_algo = RING;
  else {
    printf("\nERROR!! Invalid SEND_ALGO.\n export SEND_ALGO = 1 (ring) | 0 "
           "(recursive halving) | -1 (no quantization)\n");
    return;
  }

  if (strcmp(quant_env_var, "LLOYD") == 0)
    *quant_algo = LLOYD;
  else if (strcmp(quant_env_var, "NON_LINEAR") == 0)
    *quant_algo = NON_LINEAR;
  else if (strcmp(quant_env_var, "UNIFORM") == 0)
    *quant_algo = UNIFORM;
  else if (strcmp(quant_env_var, "HOMOMORPHIC") == 0)
    *quant_algo = HOMOMORPHIC;
  else if (strcmp(quant_env_var, "KNOWN_RANGE") == 0)
    *quant_algo = KNOWN_RANGE;
  else {
    printf("\nERROR!! Invalid QUANT_ALGO.\n export "
           "QUANT_ALGO=LLOYD|NON_LINEAR|UNFIORM\n");
    return;
  }

  if (bits_env_var != NULL) {
    int bits_env_int = atoi(bits_env_var);
    if (bits_env_int != 8 && bits_env_int != 16) {
      printf("ERROR!! Insert a BITS_VAR value of 8 or 16 \n");
      return;
    }
    BITS = bits_env_int;
    REPR_RANGE = (1 << BITS);
  } else {
    printf("\nERROR!! Invalid BITS_VAR.\n export BITS_VAR=8|16\n");
    return;
  }
}

/* Generates a random vector of float parallelizing the work	*
 * on more threads and using rand_r (a thread safe version of	*
 * time.h rand that requires an explicit seed), an upperbound	*
 * and a lowerbound.						*/
float *RandFloatGenerator(size_t lenght, float lowerbound, float upperbound) {
  float *list = malloc(lenght * sizeof(float));

  int i = 0;
  unsigned int seed;

  float range = ((float)RAND_MAX) / (upperbound - lowerbound);

#pragma omp parallel default(none)                                             \
    shared(list, lenght, range, lowerbound) private(i, seed)
  {
    seed = rand();
#pragma omp for
    for (i = 0; i < lenght; i++)
      list[i] = (float)rand_r(&seed) / range + lowerbound;
  }

  return list;
}

/* Writes scan in a parallel way (through the use of reduction	*
 * clauses) through the vector to find the minimum and maximum	*
 * values.							*
 * those values are then stored.				*
 * In the code it is often used with the uint8_vec struct to	*
 * store those value for the quantization interval setup and	*
 * for the dequantization itself.				*/
void MinMax(float *vec, size_t lenght, float *min, float *max, int hom_flag) {

  float minimum = INFINITY;
  float maximum = -INFINITY;

#pragma omp parallel for reduction(min : minimum) reduction(max : maximum)
  for (int i = 0; i < lenght; i++) {
    if (vec[i] < minimum)
      minimum = vec[i];
    if (vec[i] > maximum)
      maximum = vec[i];
  }

  if (minimum == INFINITY || maximum == -INFINITY) {
    printf("ERROR!! minimum is: %f maximum is: %f\n", minimum, maximum);
    return;
  }

  // if MinMax is used with QUANT_ALGO=HOMOMORPHIC
  // a flag is set to register the minimum as -minimum
  if (hom_flag == 1)
    *min = -minimum;
  else
    *min = minimum;
  *max = maximum;
}

float sign(float x) {
  if (x >= 0)
    return 1.0;
  else
    return -1.0;
}

// MeanSquaredError between two float arrays
float MeanSquaredError(float *v1, float *v2, size_t lenght) {
  float out = 0;

#pragma omp parallel for reduction(+ : out)
  for (int i = 0; i < lenght; i++)
    out += (v1[i] - v2[i]) * (v1[i] - v2[i]);

  out = out / (float)lenght;

  return out;
}

// MSE but each value it's normalized by its dimension
float NormalizedMSE(float *v1, float *v2, size_t lenght) {
  float out = 0;

#pragma omp parallel for reduction(+ : out)
  for (int i = 0; i < lenght; i++)
    out += (v1[i] - v2[i]) * (v1[i] - v2[i]) / (v1[i] * v1[i]);

  out = out / (float)lenght;
  return out;
}

void ProcessPrinter(void *obj, size_t lenght, int my_rank, int comm_sz,
                    TYPE t) {
  switch (t) {
  case INT: {
    int *intptr = (int *)obj;
    for (int rank = 0; rank < comm_sz; rank++) {
      if (my_rank == rank) {
        printf("my_rank = %d\n", my_rank);
        for (int ind = 0; ind < lenght; ind++)
          printf("INT[%d] = %d\t", ind, intptr[ind]);
        printf("\n");
      }
      if (comm_sz != 1)
        MPI_Barrier(MPI_COMM_WORLD);
    }
    break;
  }
  case FLOAT: {
    float *floatptr = (float *)obj;
    for (int rank = 0; rank < comm_sz; rank++) {
      if (my_rank == rank) {
        printf("my_rank = %d\n", my_rank);
        for (int ind = 0; ind < lenght; ind++)
          printf("FLOAT[%d] = %.2f\t", ind, floatptr[ind]);
        printf("\n");
      }
      if (comm_sz != 1)
        MPI_Barrier(MPI_COMM_WORLD);
    }
    break;
  }
  case UINT8: {
    uint8_t *uint8ptr = (uint8_t *)obj;
    for (int rank = 0; rank < comm_sz; rank++) {
      if (my_rank == rank) {
        printf("my_rank = %d\n", my_rank);
        for (int ind = 0; ind < lenght; ind++)
          printf("UINT8[%d] = %u\t", ind, uint8ptr[ind]);
        printf("\n");
      }
      if (comm_sz != 1)
        MPI_Barrier(MPI_COMM_WORLD);
    }
    break;
  }
  case UINT16: {
    uint16_t *uint16ptr = (uint16_t *)obj;
    for (int rank = 0; rank < comm_sz; rank++) {
      if (my_rank == rank) {
        printf("my_rank = %d\n", my_rank);
        for (int ind = 0; ind < lenght; ind++)
          printf("UINT16[%d] = %u\t", ind, uint16ptr[ind]);
        printf("\n");
      }
      if (comm_sz != 1)
        MPI_Barrier(MPI_COMM_WORLD);
    }
    break;
  }
  default: {
    printf("no type inserted\n");
    break;
  }
  }
}
