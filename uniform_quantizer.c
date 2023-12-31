#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>		// used for the definition of int8 (int8_t in the library), not really necessary since we can use char
#include <omp.h>
#include <math.h>		// used for floor function, not really necessary since we can do bitwise operation
#include <time.h>		// used to random generate vectors

int thread_count;

void RandFloatGenerator(float* list, int lenght, float upperbound);

void AffineQuantization(int8_t* out, float* in, int lenght, float scale, float offset);

void AffineDequantization(float* out, int8_t* in, int lenght, float scale, float offset);

int main(int argc, char** argv){
	srand((unsigned int) time(NULL));
	thread_count = 8;

	int dim = strtol(argv[1], NULL, 10);
	/*float* list = malloc(dim*sizeof(float));

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


/* generates a random vector of float parallelizing the work	*
 * on more threads and using rand_r (a thread safe version of	*
 * time.h rand that requires an explicit seed) and an		*
 * upperbound. The values are then zero meaned.			*
 *								*
 * input: pointer to float array, array lenght, upperbound	*/
void RandFloatGenerator(float* list, int lenght, float upperbound){
	int i=0;
	unsigned int seed;
	float off=upperbound/2;
	#pragma omp parallel num_threads(thread_count)\
		default(none) shared(list, lenght, upperbound, off) private(i, seed)
	{
		seed=rand();
		#pragma omp for
		for(i=0; i<lenght; i++)
			list[i] = (float) rand_r(&seed) / (float) (RAND_MAX/upperbound) - off;
	}
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
void AffineQuantization(int8_t* out, float* in, int lenght, float scale, float offset){
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
void AffineDequantization(float* out, int8_t* in, int lenght, float scale, float offset){
	#pragma omp parallel for default(none) shared(out, in, lenght, scale, offset)
	for(int i=0; i<lenght; i++)
		out[i] = (((float) in[i]) + offset)*scale;
}

