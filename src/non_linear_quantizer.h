#ifndef NON_LINEAR_QUANTIZER_H
#define NON_LINEAR_QUANTIZER_H

#include <stdlib.h>
#include "tools.h"

struct uint8_vec * NonLinearQuantization(float* in, size_t input_size, int type);

float* NonLinearDequantization(struct uint8_vec* in, size_t input_size, int type);

#endif

