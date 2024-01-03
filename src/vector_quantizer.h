#ifndef VECTOR_QUANTIZER_H
#define VECTOR_QUANTIZER_H

#include <stdlib.h>
#include <stdint.h>

#include "tools.h"

uint8_t NearestCodevector(float* vector, float* vectorbook, size_t vector_size, size_t vectorbook_size, int index);

void UpdateVectorbook(float* input, uint8_t* assignments, float* vectorbook, size_t input_size, size_t vectorbook_size, size_t vector_size);

struct uint8_vec_vq * LBGVectorQuantizer(float* in, size_t input_size, size_t vector_size);

float* LBGVectorDequantizer(struct uint8_vec_vq * in, size_t input_lenght, size_t vectorbook_size);

#endif
