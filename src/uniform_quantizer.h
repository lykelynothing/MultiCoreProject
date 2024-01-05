#ifndef UNIFORM_QUANTIZER_H
#define UNIFROM_QUANTIZER_H

#include <stdint.h>
#include <stdlib.h>

#include "tools.h"

void UniformAffineQuantization(int8_t* out, float* in, int lenght, float scale, float offset);

void UniformAffineDequantization(float* out, int8_t* in, int lenght, float scale, float offset);

struct unif_quant* UniformRangedQuantization(float* in, size_t input_size);

float* UniformRangedDequantization(struct unif_quant* in, size_t input_size);

#endif

