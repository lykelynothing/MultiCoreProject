#ifndef HOMOMORPHIC_QUANTIZER_H
#define HOMOMORPHIC_QUANTIZER_H

#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

#include "tools.h"

struct unif_quant* HomomorphicQuantization(float* input, size_t input_size, MPI_Comm comm, void * struct_ptr);

struct unif_quant_16* HomomorphicQuantization_16(float* input, size_t input_size, MPI_Comm comm, void * struct_ptr);

float* HomomorphicDequantization(uint8_t* quantized, float min, float max, int comm_sz, size_t input_size, int reduction_flag, float * out);

float* HomomorphicDequantization_16(uint16_t* quantized, float min, float max, int comm_sz, size_t input_size, int reduction_flag, float * out);

#endif 
