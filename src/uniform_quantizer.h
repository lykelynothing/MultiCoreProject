#ifndef UNIFORM_QUANTIZER_H
#define UNIFROM_QUANTIZER_H

#include <stdint.h>
#include "tools.h"

void UniformAffineQuantization(int8_t* out, float* in, int lenght, float scale, float offset);

void UniformAffineDequantization(float* out, int8_t* in, int lenght, float scale, float offset);

struct uint8_vec* UniformRangedQuantization(float* in, size_t input_size);

float* UniformRangedDequantization(struct uint8_vec* in, size_t input_size);

#endif

