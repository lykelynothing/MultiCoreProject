#ifndef RING_REDUCE_H
#define RING_REDUCE_H

#include <stdlib.h>

#include "tools.h"

int RingAllreduce(int my_rank, int comm_sz, float *data, size_t dim,
                  float **output_ptr, QUANT algo);
int RingAllreduce_16(int my_rank, int comm_sz, float *data, size_t dim,
                     float **output_ptr, QUANT algo);

#endif
