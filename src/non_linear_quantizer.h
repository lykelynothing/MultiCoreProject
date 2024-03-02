#ifndef NON_LINEAR_QUANTIZER_H
#define NON_LINEAR_QUANTIZER_H

#include <stdlib.h>
#include "tools.h"

struct non_linear_quant* NonLinearQuantization(float* in, size_t input_size, int type, void* struct_ptr);

struct non_linear_quant_16* NonLinearQuantization_16(float* in, size_t input_size, int type, void* struct_ptr);

float* NonLinearDequantization(struct non_linear_quant* in, size_t input_size, float* out, int type);

float* NonLinearDequantization_16(struct non_linear_quant_16* in, size_t input_size, float* out, int type);

#endif

