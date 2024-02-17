#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>	
#include <omp.h>
#include <math.h>

#include "tools.h"

/* Quantize the data partitioning the interval in equal parts,
 * the output is a struct pointer with the quantized vector, a
 * min and a max value (of the original vector)
 * To caclulate the interval a function to lookup for min value
 * and max value is called.
 * Min and max of original vector will be stored because needed
 * for dequantization.
 * The quantization itself is done in parallel with OpenMP. */
struct unif_quant* UniformRangedQuantization(float* in, size_t input_size, void * struct_ptr){
	//define the output struct and allocate the memory for the vector itself
	struct unif_quant* out = (struct unif_quant*) struct_ptr;
	//run through the vector and saves min and max value in the struct 
	MinMax(in, input_size, &(out->min), &(out->max), 0);

	//calculate the steps of the quantization
	float range = out->max - out->min;
	float min = out->min;
  	float step = (REPR_RANGE - 1) / range;
	#pragma omp parallel for
	for(int i = 0; i < input_size; i++){
		float quant = round((in[i] - min) * step);
		out->vec[i] = (uint8_t) quant;
	}

	return out;
}


/* To be used to dequantize in couple with UniformRangedQuantization.
 * A vector of uint8_t, its lenght, a min and a max value (of data 
 * before quantization) are needed as input.
 * The out1put will be a vector of dequantized data out. */
float* UniformRangedDequantization(struct unif_quant* in, size_t input_size, float * out){
	
	//calculate dequantization steps
	float max = in->max;
	float min = in->min;
	float range = max - min;
	float step = range / (REPR_RANGE - 1);

	#pragma omp parallel for
	for(int i = 0; i < input_size; i++)
		out[i] = ((float)in->vec[i]) * step + min;

	return out;
}

//SAME FOR UINT16
struct unif_quant_16* UniformRangedQuantization_16(float* in, size_t input_size, void * struct_ptr){
	struct unif_quant_16* out = (struct unif_quant_16*) struct_ptr;
	MinMax(in, input_size, &(out->min), &(out->max), 0);

	float range = out->max - out->min;
	float min = out->min;
  float step = (REPR_RANGE - 1) / range;
	#pragma omp parallel for
	for(int i = 0; i < input_size; i++){
		float quant = round((in[i] - min) * step);
		out->vec[i] = (uint16_t) quant;
	}

	return out;
}

float* UniformRangedDequantization_16(struct unif_quant_16* in, size_t input_size, float * out){
	float max = in->max;
	float min = in->min;
	float range = max - min;
	float step = range / (REPR_RANGE - 1);

	#pragma omp parallel for
	for(int i = 0; i < input_size; i++)
		out[i] = ((float)in->vec[i]) * step + min;

	return out;
}

