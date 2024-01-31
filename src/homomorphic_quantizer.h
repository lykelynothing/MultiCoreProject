#ifndef HOMOMORPHIC_QUANTIZER_H
#define HOMOMORPHIC_QUANTIZER_H

#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

#include "tools.h"

struct unif_quant* HomomorphicQuantization(float* input, size_t input_size, MPI_Comm comm);

float* HomomorphicDequantization(uint16_t* quantized, float min, float max, int comm_sz, size_t input_size);

#endif 
