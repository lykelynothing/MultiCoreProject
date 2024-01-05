#ifndef NON_LINEAR_QUANTIZER_H
#define NON_LINEAR_QUANTIZER_H

#include <stdlib.h>
#include "tools.h"

struct non_linear_quant* NonLinearQuantization(float* in, size_t input_size, int type);

float* NonLinearDequantization(struct non_linear_quant* in, size_t input_size);

#endif

