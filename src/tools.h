#ifndef TOOLS_H
#define TOOLS_H

#include <stdint.h>
#include <stdlib.h>

#define BITS 8
#define REPR_RANGE (1 << BITS) 

struct compressed {
	uint64_t number : BITS;
}__attribute__((packed));

struct unif_quant{
	struct compressed* vec;
	float min;
	float max;
};

struct non_linear_quant{
	struct compressed* vec;
	float min;
	float max;
	int type;
};

struct lloyd_max_quant{
	struct compressed* vec;
	float min;
	float max;
	float* codebook;
};

struct vector_quant{
	struct compressed* vec;
	float min;
	float max;
	size_t vec_size;	//size of the sub-vectors of the vector vec
	float* vectorbook;
};

float* RandFloatGenerator(size_t lenght, float lowerbound, float upperbound);

void MinMax(float* vec, size_t lenght, float* min, float* max);

void PrintFloatVec(float* vec, size_t lenght, char* prompt);

void PrintInt8Vec(uint64_t* vec, size_t lenght, char* prompt);

void PrintVectorbook(struct vector_quant* in, char* prompt);

float sign(float x);

float MeanSquaredError(float* v1, float* v2, size_t lenght);

float NormalizedMSE(float*v1, float* v2, size_t lenght);

//float VectorMean(float* vec, int lenght);

#endif
