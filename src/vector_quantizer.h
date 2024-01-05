#ifndef VECTOR_QUANTIZER_H
#define VECTOR_QUANTIZER_H

#include <stdlib.h>

#include "tools.h"

struct vector_quant * LBGVectorQuantizer(float* in, size_t input_size, size_t vector_size);

float* LBGVectorDequantizer(struct vector_quant* in, size_t input_size);

#endif
