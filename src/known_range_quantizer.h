#ifndef KNOWN_RANGE_QUANTIZER_H
#define KNOWN_RANGE_QUANTIZER_H

#include <omp.h>
#include <stdlib.h>

#include "tools.h"

struct unif_quant *KnownRangeQuantization(float *input, size_t input_size,
                                          MPI_Comm comm, void *struct_ptr);

struct unif_quant_16 *KnownRangeQuantization_16(float *input, size_t input_size,
                                                MPI_Comm comm,
                                                void *struct_ptr);

#endif // !KNOWN_RANGE_QUANTIZER_H
