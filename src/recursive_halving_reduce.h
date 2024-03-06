#ifndef RECURSIVE_HALVING_REDUCE_H
#define RECURSIVE_HALVING_REDUCE_H

#include "tools.h"

int RecursiveHalvingSend(int my_rank, int comm_sz, int dim, QUANT algo,
                         float *my_numbers, float *recv_buf);
int RecursiveHalvingSendHomomorphic(int my_rank, int comm_sz, int count,
                                    QUANT algo, float *sendbuf,
                                    float **recvbuf);

#endif
