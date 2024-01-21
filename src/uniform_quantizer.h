#ifndef UNIFORM_QUANTIZER_H
#define UNIFROM_QUANTIZER_H

#include <stdint.h>
#include <stdlib.h>

#include "tools.h"

struct unif_quant* UniformRangedQuantization(float* in, size_t input_size);

float* UniformRangedDequantization(struct unif_quant* in, size_t input_size);

#endif

