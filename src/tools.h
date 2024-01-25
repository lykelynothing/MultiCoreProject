#ifndef TOOLS_H
#define TOOLS_H

#include <stdint.h>
#include <stdlib.h>
#include <mpi.h>

extern int BITS;
extern int REPR_RANGE;

struct unif_quant{
	float min;
	float max;
	uint8_t* vec;
};

struct non_linear_quant{
	float min;
	float max;
	int type;
	uint8_t* vec;
};

struct lloyd_max_quant{
	float min;
	float max;
	float* codebook;
	uint8_t* vec;
};

MPI_Datatype UnifQuantType();

MPI_Datatype NonLinearQuantType();

MPI_Datatype LloydMaxQuantType();

float* RandFloatGenerator(size_t lenght, float lowerbound, float upperbound);

void MinMax(float* vec, size_t lenght, float* min, float* max);

void PrintFloatVec(float* vec, size_t lenght, char* prompt);

void PrintInt8Vec(uint8_t* vec, size_t lenght, char* prompt);

float sign(float x);

float MeanSquaredError(float* v1, float* v2, size_t lenght);

float NormalizedMSE(float*v1, float* v2, size_t lenght);

void Quicksort(float* input, size_t dim);

#endif
