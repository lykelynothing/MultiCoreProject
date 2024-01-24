#ifndef TOOLS_H
#define TOOLS_H

#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>

extern int BITS;
extern int REPR_RANGE;

struct unif_quant{
	uint8_t* vec;
	float min;
	float max;
};

struct non_linear_quant{
	uint8_t* vec;
	float min;
	float max;
	int type;
};

struct lloyd_max_quant{
	uint8_t* vec;
	float min;
	float max;
	float* codebook;
};

MPI_Datatype UnifQuantType(int array_size);

MPI_Datatype NonLinearQuantType(int array_size);

MPI_Datatype LloydMaxQuantType(int array_size);

float* RandFloatGenerator(size_t lenght, float lowerbound, float upperbound);

void MinMax(float* vec, size_t lenght, float* min, float* max);

void PrintFloatVec(float* vec, size_t lenght, char* prompt);

void PrintInt8Vec(uint8_t* vec, size_t lenght, char* prompt);

float sign(float x);

float MeanSquaredError(float* v1, float* v2, size_t lenght);

float NormalizedMSE(float*v1, float* v2, size_t lenght);

void Quicksort(float* input, size_t dim);

#endif
