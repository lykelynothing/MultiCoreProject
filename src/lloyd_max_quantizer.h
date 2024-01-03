#ifndef LLOYD_MAX_QUANTIZER_H
#define LLOYD_MAX_QUANTIZER_H

#include <stdlib.h>
#include "tools.h"

struct uint8_vec_lm * LloydMaxQuantizer(float* in, size_t input_size);

uint8_t NearestCodeword(float element, float* codebook, int codebook_len);

void UpdateCodebook(float* input, uint8_t* assignments, float* codebook, size_t input_size, size_t codebook_size);

float * LloydMaxDequantizer(struct uint8_vec_lm * in, size_t input_size, size_t codebook_size);

#endif

