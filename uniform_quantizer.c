#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>	
#include <omp.h>
#include <math.h>

int thread_count;

void UniformAffineQuantization(int8_t* out, float* in, int lenght, float scale, float offset);

void UniformAffineDequantization(float* out, int8_t* in, int lenght, float scale, float offset);

void UniformRangedQuantization(uint8_t* out, float* in, int lenght, float* min, float* max);

void UniformRangedDequantization(float* out, uint8_t* in, int lenght, float min, float max);


int main(int argc, char** argv){
	thread_count = 8;

	/*int dim = strtol(argv[1], NULL, 10);
	float* list = malloc(dim*sizeof(float));

	float scale = 10.0;
	RandFloatGenerator(list, dim, 256.0*scale);

	struct q_val datas;
	VectorDatas(list, dim, &datas);

	int8_t* quantized_data = malloc(dim*sizeof(int8_t));
	
	AffineQuantization(quantized_data, list, dim, scale, 0);

	float* dequantized_data = malloc(dim*sizeof(float));

	AffineDequantization(dequantized_data, quantized_data, dim, scale, 0);

	char* p1 = "The float list is:";
	PrintFloatVec(list, dim,p1);

	printf("Mean: %f \tMin: %f\t Max: %f\n\n\n", datas.mean, datas.min, datas.max);
	
	char* p2 = "The quantized int8 list is:";
	PrintInt8Vec(quantized_data, dim, p2);

	char* p3 = "The dequantized float list is:";
	PrintFloatVec(dequantized_data, dim, p3);
	
	printf("The mean squared error between the two lists is: %f \n", MeanSquaredError(list, dequantized_data, dim));
	free(list);
	free(quantized_data);
	free(dequantized_data);*/

	return 0;
}

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

void UniformRangedQuantization(uint8_t* out, float* in, int lenght, float* min, float* max){
	float range, step;
	MinMax(in, lenght, max, min);
	float range = max - min;
	float step = range / 256.0;
	#pragma omp parallel for default(none) shared(out, in, lenght, min, step)
	for(int i=0; i<lenght; i++){
		float quant = floor((in[i] - min)/step);
		out[i] = (unsigned int8_t) quant;
	}
}

void UniformRangedDequantization(float* out, uint8_t* in, int lenght, float min, float max){
	float range, step;
	range = max - min;
	step = range / 256.0;
	#pragma omp parallel for default(none) shared(out, in, lenght, step, min)
	for(int i=0; i<lenght; i++)
		out[i] = ((float)in[i]) * step + min;
}


