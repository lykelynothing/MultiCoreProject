#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "homomorphic_quantizer.h"
#include "known_range_quantizer.h"
#include "lloyd_max_quantizer.h"
#include "non_linear_quantizer.h"
#include "tools.h"
#include "uniform_quantizer.h"

static int type;

/* Function takes float vector and quantizes it according to
 * the kind of algorithm specified by int algo. Struct in arguments
 * will contain the quantized array */
void *Quantize(float *sendbuf, int count, QUANT algo, void *struct_ptr) {

  if (BITS == 8) {
    switch (algo) {
    case LLOYD: {
      LloydMaxQuantizer(sendbuf, count, struct_ptr);
      break;
    }
    case NON_LINEAR: {
      NonLinearQuantization(sendbuf, count, type, struct_ptr);
      break;
    }
    case UNIFORM: {
      UniformRangedQuantization(sendbuf, count, struct_ptr);
      break;
    }
    case HOMOMORPHIC: {
      HomomorphicQuantization(sendbuf, count, MPI_COMM_WORLD, struct_ptr);
      break;
    }
    case KNOWN_RANGE: {
      KnownRangeQuantization(sendbuf, count, MPI_COMM_WORLD, struct_ptr);
      break;
    }
    default: {
      printf("ERROR!! Quant algo not valid (quantize call)\n");
      return NULL;
    }
    }
  } else if (BITS == 16) {
    switch (algo) {
    case LLOYD: {
      LloydMaxQuantizer_16(sendbuf, count, struct_ptr);
      break;
    }
    case NON_LINEAR: {
      NonLinearQuantization_16(sendbuf, count, type, struct_ptr);
      break;
    }
    case UNIFORM: {
      UniformRangedQuantization_16(sendbuf, count, struct_ptr);
      break;
    }
    case HOMOMORPHIC: {
      HomomorphicQuantization_16(sendbuf, count, MPI_COMM_WORLD, struct_ptr);
      break;
    }
    case KNOWN_RANGE: {
      KnownRangeQuantization_16(sendbuf, count, MPI_COMM_WORLD, struct_ptr);
      break;
    }
    default: {
      printf("ERROR!! Quant algo not valid (quantize call)\n");
      return NULL;
    }
    }
  }

  return struct_ptr;
}

/* Writes into dequantized_1 the quantized array present into
 * struct_ptr1.vec */
void DequantizeVector(void *struct_ptr, float *dequantized, QUANT algo,
                      int dim) {
  int comm_sz;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  if (BITS == 8) {
    switch (algo) {
    case LLOYD: {
      struct lloyd_max_quant *str0 = (struct lloyd_max_quant *)struct_ptr;
      LloydMaxDequantizer(str0, dim, dequantized);
      break;
    }
    case NON_LINEAR: {
      struct non_linear_quant *str1 = (struct non_linear_quant *)struct_ptr;
      NonLinearDequantization(str1, dim, dequantized, type);
      break;
    }
    case UNIFORM: {
      struct unif_quant *str2 = (struct unif_quant *)struct_ptr;
      UniformRangedDequantization(str2, dim, dequantized);
      break;
    }
    case HOMOMORPHIC: {
      struct unif_quant *str3 = (struct unif_quant *)struct_ptr;
      HomomorphicDequantization(str3->vec, str3->min, str3->max, comm_sz, dim,
                                1, dequantized);
      break;
    }
    case KNOWN_RANGE: {
      struct unif_quant *str4 = (struct unif_quant *)struct_ptr;
      HomomorphicDequantization(str4->vec, str4->min, str4->max, comm_sz, dim,
                                1, dequantized);
      break;
    }
    default: {
      printf("ERROR!! Quant algo not valid (dequantize call)\n");
      break;
    }
    }
  } else if (BITS == 16) {
    switch (algo) {
    case LLOYD: {
      struct lloyd_max_quant_16 *str0 = (struct lloyd_max_quant_16 *)struct_ptr;
      LloydMaxDequantizer_16(str0, dim, dequantized);
      break;
    }
    case NON_LINEAR: {
      struct non_linear_quant_16 *str1 =
          (struct non_linear_quant_16 *)struct_ptr;
      NonLinearDequantization_16(str1, dim, dequantized, type);
      break;
    }
    case UNIFORM: {
      struct unif_quant_16 *str2 = (struct unif_quant_16 *)struct_ptr;
      UniformRangedDequantization_16(str2, dim, dequantized);
      break;
    }
    case HOMOMORPHIC: {
      struct unif_quant_16 *str3 = (struct unif_quant_16 *)struct_ptr;
      HomomorphicDequantization_16(str3->vec, str3->min, str3->max, comm_sz,
                                   dim, 1, dequantized);
      break;
    }
    case KNOWN_RANGE: {
      struct unif_quant_16 *str4 = (struct unif_quant_16 *)struct_ptr;
      HomomorphicDequantization_16(str4->vec, str4->min, str4->max, comm_sz,
                                   dim, 1, dequantized);
      break;
    }
    default: {
      printf("ERROR!! Quant algo not valid (dequantize call)\n");
      break;
    }
    }
  }
}

/* Calls mpi to receive struct and quantized vector. Handles
 * all kinds of struct and attaches to vec field the received
 * quantized vector (uint8). Returns pointer to received struct */
// Receives an already allocated struct
void *Receive(QUANT algo, int dim, int source, void *void_ptr) {
  if (BITS == 8) {
    switch (algo) {
      // lloyd also contains codebook
    case LLOYD: {
      struct lloyd_max_quant *str_ptr1 = (struct lloyd_max_quant *)void_ptr;
      MPI_Recv(&str_ptr1->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&str_ptr1->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      MPI_Recv(str_ptr1->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      MPI_Recv(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, source, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      break;
    }
    case NON_LINEAR: {
      struct non_linear_quant *str_ptr2 = (struct non_linear_quant *)void_ptr;
      MPI_Recv(&str_ptr2->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&str_ptr2->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(str_ptr2->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      break;
    }
    case UNIFORM: {
      struct unif_quant *str_ptr3 = (struct unif_quant *)void_ptr;
      MPI_Recv(&str_ptr3->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&str_ptr3->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(str_ptr3->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      break;
    }
    case HOMOMORPHIC: {
      struct unif_quant *str_ptr4 = (struct unif_quant *)void_ptr;
      MPI_Recv(&str_ptr4->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&str_ptr4->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(str_ptr4->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      break;
    }
    case KNOWN_RANGE: {
      struct unif_quant *str_ptr4 = (struct unif_quant *)void_ptr;
      MPI_Recv(&str_ptr4->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&str_ptr4->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(str_ptr4->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      break;
    }
    default: {
      printf("ERROR!! Quant algo not valid\n");
      break;
    }
    }
  } else if (BITS == 16) {
    switch (algo) {
    case LLOYD: {
      struct lloyd_max_quant_16 *str_ptr1 =
          (struct lloyd_max_quant_16 *)void_ptr;
      MPI_Recv(&str_ptr1->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&str_ptr1->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(str_ptr1->vec, dim, MPI_UINT16_T, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, source, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      break;
    }
    case NON_LINEAR: {
      struct non_linear_quant_16 *str_ptr2 =
          (struct non_linear_quant_16 *)void_ptr;
      MPI_Recv(&str_ptr2->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&str_ptr2->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(str_ptr2->vec, dim, MPI_UINT16_T, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      break;
    }
    case UNIFORM: {
      struct unif_quant_16 *str_ptr3 = (struct unif_quant_16 *)void_ptr;
      MPI_Recv(&str_ptr3->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&str_ptr3->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(str_ptr3->vec, dim, MPI_UINT16_T, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      break;
    }
    case HOMOMORPHIC: {
      struct unif_quant_16 *str_ptr4 = (struct unif_quant_16 *)void_ptr;
      MPI_Recv(&str_ptr4->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&str_ptr4->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(str_ptr4->vec, dim, MPI_UINT16_T, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      break;
    }
    case KNOWN_RANGE: {
      struct unif_quant_16 *str_ptr4 = (struct unif_quant_16 *)void_ptr;
      MPI_Recv(&str_ptr4->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(&str_ptr4->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(str_ptr4->vec, dim, MPI_UINT16_T, source, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      break;
    }
    default: {
      printf("ERROR!! Quant algo not valid\n");
      break;
    }
    }
  }

  return void_ptr;
}

/* Sends the struct and its vec field (and codebook with LLOYD) to dest.
 * Remeber to deallocate space outside of function. */
int Send(void *struct_ptr, QUANT algo, int dim, int dest) {
  if (BITS == 8) {
    switch (algo) {
    case LLOYD: {
      struct lloyd_max_quant *str_ptr1 = (struct lloyd_max_quant *)struct_ptr;
      MPI_Send(&str_ptr1->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr1->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr1->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, dest, 0,
               MPI_COMM_WORLD);
      break;
    }
    case NON_LINEAR: {
      struct non_linear_quant *str_ptr2 = (struct non_linear_quant *)struct_ptr;
      MPI_Send(&str_ptr2->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr2->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr2->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      break;
    }
    case UNIFORM: {
      struct unif_quant *str_ptr3 = (struct unif_quant *)struct_ptr;
      MPI_Send(&str_ptr3->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr3->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr3->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      break;
    }
    case HOMOMORPHIC: {
      struct unif_quant *str_ptr4 = (struct unif_quant *)struct_ptr;
      MPI_Send(&str_ptr4->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr4->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr4->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      break;
    }
    case KNOWN_RANGE: {
      struct unif_quant *str_ptr4 = (struct unif_quant *)struct_ptr;
      MPI_Send(&str_ptr4->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr4->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr4->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      break;
    }
    default: {
      printf("ERROR!! Quant algo not valid (send_call)\n");
      break;
    }
    }
  } else if (BITS == 16) {
    switch (algo) {
    case LLOYD: {
      struct lloyd_max_quant_16 *str_ptr1 =
          (struct lloyd_max_quant_16 *)struct_ptr;
      MPI_Send(&str_ptr1->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr1->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr1->vec, dim, MPI_UINT16_T, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, dest, 0,
               MPI_COMM_WORLD);
      break;
    }
    case NON_LINEAR: {
      struct non_linear_quant_16 *str_ptr2 =
          (struct non_linear_quant_16 *)struct_ptr;
      MPI_Send(&str_ptr2->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr2->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr2->vec, dim, MPI_UINT16_T, dest, 0, MPI_COMM_WORLD);
      break;
    }
    case UNIFORM: {
      struct unif_quant_16 *str_ptr3 = (struct unif_quant_16 *)struct_ptr;
      MPI_Send(&str_ptr3->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr3->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr3->vec, dim, MPI_UINT16_T, dest, 0, MPI_COMM_WORLD);
      break;
    }
    case HOMOMORPHIC: {
      struct unif_quant_16 *str_ptr4 = (struct unif_quant_16 *)struct_ptr;
      MPI_Send(&str_ptr4->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr4->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr4->vec, dim, MPI_UINT16_T, dest, 0, MPI_COMM_WORLD);
      break;
    }
    case KNOWN_RANGE: {
      struct unif_quant_16 *str_ptr4 = (struct unif_quant_16 *)struct_ptr;
      MPI_Send(&str_ptr4->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(&str_ptr4->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr4->vec, dim, MPI_UINT16_T, dest, 0, MPI_COMM_WORLD);
      break;
    }
    default: {
      printf("ERROR!! Quant algo not valid (send_call)\n");
      break;
    }
    }
  }
  return MPI_SUCCESS;
}

void *Allocate(QUANT algo, int count) {
  void *void_ptr;
  if (BITS == 8) {
    switch (algo) {
    case LLOYD: {
      void_ptr = malloc(sizeof(struct lloyd_max_quant));
      struct lloyd_max_quant *tmp_ptr1 = (struct lloyd_max_quant *)void_ptr;
      tmp_ptr1->vec = malloc(sizeof(uint8_t) * count);
      tmp_ptr1->codebook = malloc(sizeof(float) * count);
      break;
    }
    case NON_LINEAR: {
      void_ptr = malloc(sizeof(struct non_linear_quant));
      struct non_linear_quant *tmp_ptr2 = (struct non_linear_quant *)void_ptr;
      tmp_ptr2->vec = malloc(sizeof(uint8_t) * count);

      char *string_type_env = getenv("NON_LINEAR_TYPE");
      if (string_type_env != NULL)
        type = atoi(string_type_env);
      else {
        printf("\nERROR : Couldn't find a type env_var, aborting...\n\n");
        return NULL;
      }

      tmp_ptr2->type = type;
      break;
    }
    case UNIFORM: {
      void_ptr = malloc(sizeof(struct unif_quant));
      struct unif_quant *tmp_ptr3 = (struct unif_quant *)void_ptr;
      tmp_ptr3->vec = malloc(sizeof(uint8_t) * count);
      break;
    }
    case HOMOMORPHIC: {
      void_ptr = malloc(sizeof(struct unif_quant));
      struct unif_quant *tmp_ptr4 = (struct unif_quant *)void_ptr;
      tmp_ptr4->vec = malloc(sizeof(uint8_t) * count);
      break;
    }
    case KNOWN_RANGE: {
      void_ptr = malloc(sizeof(struct unif_quant));
      struct unif_quant *tmp_ptr4 = (struct unif_quant *)void_ptr;
      tmp_ptr4->vec = malloc(sizeof(uint8_t) * count);
      break;
    }
    default: {
      break;
    }
    }
  } else if (BITS == 16) {
    switch (algo) {
    case LLOYD: {
      void_ptr = malloc(sizeof(struct lloyd_max_quant_16));
      struct lloyd_max_quant_16 *tmp_ptr1 =
          (struct lloyd_max_quant_16 *)void_ptr;
      tmp_ptr1->vec = malloc(sizeof(uint16_t) * count);
      break;
    }
    case NON_LINEAR: {
      void_ptr = malloc(sizeof(struct non_linear_quant_16));
      struct non_linear_quant_16 *tmp_ptr2 =
          (struct non_linear_quant_16 *)void_ptr;
      tmp_ptr2->vec = malloc(sizeof(uint16_t) * count);

      char *string_type_env = getenv("NON_LINEAR_TYPE");
      if (string_type_env != NULL)
        type = atoi(string_type_env);
      else {
        printf("\nERROR : Couldn't find a type env_var, aborting...\n\n");
        return NULL;
      }

      tmp_ptr2->type = type;
      break;
    }
    case UNIFORM: {
      void_ptr = malloc(sizeof(struct unif_quant_16));
      struct unif_quant_16 *tmp_ptr3 = (struct unif_quant_16 *)void_ptr;
      tmp_ptr3->vec = malloc(sizeof(uint16_t) * count);
      break;
    }
    case HOMOMORPHIC: {
      void_ptr = malloc(sizeof(struct unif_quant_16));
      struct unif_quant_16 *tmp_ptr4 = (struct unif_quant_16 *)void_ptr;
      tmp_ptr4->vec = malloc(sizeof(uint16_t) * count);
      break;
    }
    case KNOWN_RANGE: {
      void_ptr = malloc(sizeof(struct unif_quant_16));
      struct unif_quant_16 *tmp_ptr4 = (struct unif_quant_16 *)void_ptr;
      tmp_ptr4->vec = malloc(sizeof(uint16_t) * count);
      break;
    }
    default: {
      break;
    }
    }
  }
  return void_ptr;
}

void Free(QUANT algo, void *void_ptr) {
  if (BITS == 8) {
    switch (algo) {
    case LLOYD: {
      struct lloyd_max_quant *tmp_ptr1 = (struct lloyd_max_quant *)void_ptr;
      free(tmp_ptr1->vec);
      free(tmp_ptr1->codebook);
      free(tmp_ptr1);
      break;
    }
    case NON_LINEAR: {
      struct non_linear_quant *tmp_ptr2 = (struct non_linear_quant *)void_ptr;
      free(tmp_ptr2->vec);
      free(tmp_ptr2);
      break;
    }
    case UNIFORM: {
      struct unif_quant *tmp_ptr3 = (struct unif_quant *)void_ptr;
      free(tmp_ptr3->vec);
      free(tmp_ptr3);
      break;
    }
    case HOMOMORPHIC: {
      struct unif_quant *tmp_ptr4 = (struct unif_quant *)void_ptr;
      free(tmp_ptr4->vec);
      free(tmp_ptr4);
      break;
    }
    case KNOWN_RANGE: {
      struct unif_quant *tmp_ptr4 = (struct unif_quant *)void_ptr;
      free(tmp_ptr4->vec);
      free(tmp_ptr4);
      break;
    }
    default: {
      break;
    }
    }
  } else if (BITS == 16) {
    switch (algo) {
    case LLOYD: {
      struct lloyd_max_quant_16 *tmp_ptr1 =
          (struct lloyd_max_quant_16 *)void_ptr;
      free(tmp_ptr1->vec);
      free(tmp_ptr1);
      break;
    }
    case NON_LINEAR: {
      struct non_linear_quant_16 *tmp_ptr2 =
          (struct non_linear_quant_16 *)void_ptr;
      free(tmp_ptr2->vec);
      free(tmp_ptr2);
      break;
    }
    case UNIFORM: {
      struct unif_quant_16 *tmp_ptr3 = (struct unif_quant_16 *)void_ptr;
      free(tmp_ptr3->vec);
      free(tmp_ptr3);
      break;
    }
    case HOMOMORPHIC: {
      struct unif_quant_16 *tmp_ptr4 = (struct unif_quant_16 *)void_ptr;
      free(tmp_ptr4->vec);
      free(tmp_ptr4);
      break;
    }
    case KNOWN_RANGE: {
      struct unif_quant_16 *tmp_ptr4 = (struct unif_quant_16 *)void_ptr;
      free(tmp_ptr4->vec);
      free(tmp_ptr4);
      break;
    }
    default: {
      break;
    }
    }
  }
}
