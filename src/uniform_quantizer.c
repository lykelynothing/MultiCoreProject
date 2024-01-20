#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>	
#include <omp.h>
#include <math.h>

#include "tools.h"

/* Quantize the data partitioning the interval in equal parts,	*
 * the output is a struct pointer with the quantized vector, a	*
 * min and a max value (of the original vector).		*
 * To caclulate the interval a function to lookup for min value	*
 * and max value is called.					*
 * Min and max of original vector will be stored because needed	*
 * for dequantization.						*
 * The quantization itself is done in parallel with OpenMP.	*/
struct unif_quant* UniformRangedQuantization(float* in, size_t input_size){
	//define the output struct and allocate the memory for the vector itself
	struct unif_quant* out = (struct unif_quant *) malloc(sizeof(struct unif_quant));
	out->vec = (struct compressed*) malloc(input_size*sizeof(struct compressed));

	//run through the vector and saves min and max value in the struct 
	MinMax(in, input_size, &(out->min), &(out->max));

	//calculate the steps of the quantization
	float range = out->max - out->min;
	float min = out->min;			//cache optimization for parallel loop
	float step = range / REPR_RANGE;

	#pragma omp parallel for default(none) shared(out, in, input_size, min, step)
	for(int i = 0; i < input_size; i++){
		float quant = floor((in[i] - min) / step);
		out->vec[i].number = (uint64_t) quant;
	}

	return out;
}


/* To be used to dequantize in couple with			*
 * UniformRangedQuantization.					*
 * A vector of uint8_t, its lenght, a min and a max value (of	*
 * data before quantization) are needed as input.		*
 * The output will be a vector of dequantized data out.		*
 *								*/
float* UniformRangedDequantization(struct unif_quant* in, size_t input_size){
	//allocates memory for the dequantized vector
	float* out = malloc(input_size*sizeof(float));
	
	//calculate dequantization steps
	float max = in->max;
	float min = in->min;
	float range = max - min;
	float step = range / REPR_RANGE;

	#pragma omp parallel for default(none) shared(out, in, input_size, step, min)
	for(int i = 0; i < input_size; i++)
		out[i] = ((float)in->vec[i].number) * step + min;

	return out;
}

