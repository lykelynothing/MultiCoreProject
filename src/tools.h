#ifndef TOOLS_H
#define TOOLS_H

#include <stdint.h>
#include <stdlib.h>

struct uint8_vec{
	uint8_t* vec;
	float min;
	float max;
};

struct uint8_vec_lm{
	uint8_t* vec;
	float min;
	float max;
	float* codebook;
};
	

float* RandFloatGenerator(size_t lenght, float lowerbound, float upperbound);

void MinMax(float* vec, size_t lenght, float* min, float* max);

void PrintFloatVec(float* vec, size_t lenght, char* prompt);

void PrintInt8Vec(uint8_t* vec, size_t lenght, char* prompt);

float sign(float x);

float MeanSquaredError(float* v1, float* v2, size_t lenght);

float NormalizedMSE(float*v1, float* v2, size_t lenght);

//float VectorMean(float* vec, int lenght);


#endif