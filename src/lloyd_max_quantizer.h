#ifndef LLOYD_MAX_QUANTIZER_H
#define LLOYD_MAX_QUANTIZER_H

#include <stdlib.h>

#include "tools.h"

struct lloyd_max_quant* LloydMaxQuantizer(float* in, size_t input_size);

float* LloydMaxDequantizer(struct lloyd_max_quant* in, size_t input_size);

#endif

