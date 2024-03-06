#ifndef TOOLS_COLLECTIVES_H
#define TOOLS_COLLECTIVES_H

#include "tools.h"

void *Quantize(float *sendbuf, int count, QUANT algo, void *struct_ptr);

void DequantizeVector(void *struct_ptr, float *dequantized, QUANT algo,
                      int dim);

void *Receive(QUANT algo, int dim, int source, void *void_ptr);

int Send(void *struct_ptr, QUANT algo, int dim, int dest);

void *Allocate(QUANT algo, int count);

void Free(QUANT algo, void *void_ptr);

#endif
