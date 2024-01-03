#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>	
#include <omp.h>
#include <math.h>

#include "tools.h"

//		FLOAT32---->INT8				//
/* Applies the affine quantization rounding to closest number	*
 * and utilizing more thread in a parallel for.			*
 * The result is written into an int8 array called out.		*
 *								*
 * input:	output pointer,					* 
 *		input float pointer,				*
 *		array lenght,					*
 *		scale,						*
 *		offset						*/
void UniformAffineQuantization(int8_t* out, float* in, int lenght, float scale, float offset){
	#pragma omp parallel for default(none) shared(in, out, lenght, scale, offset)
	for(int i=0; i<lenght; i++){
		float quant = floor(in[i]/scale + offset + 0.5);
		out[i] = (quant < -128) ? INT8_MIN : (quant > 127) ? INT8_MAX : (int8_t) quant;
	}
}


//		INT8---->FLOAT32				//
/* Dequantize the int8 input array into an ourput float array	*
 * utilizing more thread in a parallel for.			*
 *								*
 * input:	output pointer,					*
 *		input pointer,					*
 *		input lenght,					*
 *		scale,						*
 *		offset						*/
void UniformAffineDequantization(float* out, int8_t* in, int lenght, float scale, float offset){
	#pragma omp parallel for default(none) shared(out, in, lenght, scale, offset)
	for(int i=0; i<lenght; i++)
		out[i] = (((float) in[i]) + offset)*scale;
}


//		FLOAT---->UINT8					// 
/* Quantize the data partitioning the interval in equal parts,	*
 * the output is a struct pointer with the quantized vector, a	*
 * min and a max value (of the original vector).		*
 * To caclulate the interval a function to lookup for min value	*
 * and max value is called.					*
 * Min and max of original vector will be stored because needed	*
 * for dequantization.						*
 * The quantization itself is done in parallel with OpenMP.	*/
struct uint8_vec * UniformRangedQuantization(float* in, size_t input_size){
	//define the output struct and allocate the memory for the vector itself
	struct uint8_vec * out = (struct uint8_vec *) malloc(sizeof(struct uint8_vec));
	out->vec = (uint8_t*) malloc(input_size*sizeof(uint8_t));

	//run through the vector and saves min and max value in the struct 
	MinMax(in, input_size, &(out->min), &(out->max));

	//calculate the steps of the quantization (256 because it's the range of uint8_t)
	float range = out->max - out->min;
	float min = out->min;			//cache optimization for parallel loop
	float step = range / 256.0;

	#pragma omp parallel for default(none) shared(out, in, input_size, min, step)
	for(int i = 0; i < input_size; i++){
		float quant = floor((in[i] - min) / step);
		out->vec[i] = (uint8_t) quant;
	}

	return out;
}

//		UINT8_T---->FLOAT				 
/* To be used to dequantize in couple with			*
 * UniformRangedQuantization.					*
 * A vector of uint8_t, its lenght, a min and a max value (of	*
 * data before quantization) are needed as input.		*
 * The output will be a vector of dequantized data out.		*
 *								*/
float* UniformRangedDequantization(struct uint8_vec* in, size_t input_size){
	//allocates memory for the dequantized vector
	float* out = malloc(input_size*sizeof(float));
	
	//calculate dequantization steps
	float max = in->max;
	float min = in->min;
	float range = max - min;
	float step = range / 256.0;

	#pragma omp parallel for default(none) shared(out, in, input_size, step, min)
	for(int i = 0; i < input_size; i++)
		out[i] = ((float)in->vec[i]) * step + min;

	return out;
}

