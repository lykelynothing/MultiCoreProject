#ifndef UNIFORM_QUANTIZER_H
#define UNIFORM_QUANTIZER_H

#include <stdint.h>
#include <stdlib.h>

#include "tools.h"

struct unif_quant *UniformRangedQuantization(float *in, size_t input_size,
                                             void *struct_ptr);

struct unif_quant_16 *UniformRangedQuantization_16(float *in, size_t input_size,
                                                   void *struct_ptr);

float *UniformRangedDequantization(struct unif_quant *in, size_t input_size,
                                   float *out);

float *UniformRangedDequantization_16(struct unif_quant_16 *in,
                                      size_t input_size, float *out);

#endif
