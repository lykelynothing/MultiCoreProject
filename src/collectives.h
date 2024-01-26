#ifndef COLLECTIVES_H
#define COLLECTIVES_H

#include <mpi.h>

int MPI_Allreduce(const void * sendbuf, void * recvbuf, 
    int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Send(void * struct_ptr, int algo, int dim, int dest);
int RecursiveHalvingSend(int my_rank, int comm_sz, int dim, int algo, float * my_numbers);
void * Receive(int algo, int dim, int source);
void * Quantize(float * sendbuf, int count, int algo);
void DequantizeVector(void * struct_ptr1, float ** dequantized_1, uint8_t * quantized, int algo, int dim);



#endif
