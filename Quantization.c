#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>		// used for the definition of int8 (int8_t in the library), not really necessary since we can use char
#include <omp.h>
#include <math.h>		// used for floor function, not really necessary since we can do bitwise operation
#include <time.h>		// used to random generate vectors

/* TODO list:
 * 
 * 
 *
 * 
 * offset = mean of the vector???????
 * utilizing symmetric quantization discard offset, so can be better
 *
 * SCALE SHOULD BE CHOSEN WITH THE AIM OF MAXIMIZING DATA VARIANCE
 * how we pre-process it/fine tune it? So what shound [a,b] be?
 * - [min,max] when you  have something that looks like a normal distribution
 * - [min observed, max observed] when normal distribution and vector is too big
 * - entropy
 * - mean squared error
 * 
 * Can we use PCA to precompute those datas (i.e. find a range thar preserves the most variance)?
 * Is it computationally heavy? Is it useful?
 *
 * HOW TF ARE WE GONNA IMPLEMENT ALLTOALL AND ALLREDUCE?
 *
 *
 *
 * */

struct q_val{
	float mean;
	float min;
	float max;
};

int thread_count;

void RandFloatGenerator(float* list, int lenght, float upperbound);

void AffineQuantization(int8_t* out, float* in, int lenght, float scale, float offset);

void AffineDequantization(float* out, int8_t* in, int lenght, float scale, float offset);

void PrintFloatVec(float* vec, int lenght, char* prompt);

void PrintInt8Vec(int8_t* vec, int lenght, char* prompt);

float MeanSquaredError(float* v1, float* v2, int lenght);

float VectorMean(float* vec, int lenght);

void VectorDatas(float* vec, int lenght, struct q_val* out);

int main(int argc, char** argv){
	srand((unsigned int) time(NULL));
	thread_count = 8;

	int dim = strtol(argv[1], NULL, 10);
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
	free(dequantized_data);

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


//MeanSquaredError between two float arrays
float MeanSquaredError(float* v1, float* v2, int lenght){
	float out=0;
	#pragma omp parallel for default(none) shared(v1, v2, lenght) reduction(+:out)
	for(int i = 0; i<lenght; i++)
		out+= (v1[i]-v2[i])*(v1[i]-v2[i]);

	out = out/(float)lenght;
	
	return out;
}


//Mean of a float array, can be used as offset of affine reduction
float VectorMean(float* vec, int lenght){
	float out = 0;
	float len = (float) len;
	#pragma omp parallel for default(none) shared(vec,lenght, len) reduction(+:out)
	for(int i = 0; i<lenght; i++)
		out+= vec[i]/len;

	return out;
}

/* Utilizes a struct to store datas about the vector to		*
 * quantize. The desired data to store are the mean, the	*
 * highest value and the lowest value.				*
 * OpenMP is used to parallelize the work and three reduction	*
 * variables are declared to do the job.			*
 * Those data are then used to calculate the range [a,b] to use	*
 * in the affine or symmetric quantization.			*
 *								*
 * input:	input vector pointer,				*
 *		input vector lenght,				*
 *		output structure pointer			*/
void VectorDatas(float* vec, int lenght, struct q_val* out){
	float len = (float) lenght;
	float mean = vec[0]/len;
	float minimum = vec[0];
	float maximum = vec[0];
	#pragma omp parallel for default(none) shared(vec, lenght, len) \
		reduction(+: mean) reduction(min: minimum) reduction(max: maximum)
	for(int i=1; i<lenght;i++){
		mean += vec[i]/len;
		if(vec[i] < minimum)		minimum = vec[i];
		else if(vec[i] > maximum)	maximum = vec[i];
	}

	out->mean = mean;
	out->min = minimum;
	out->max = maximum;
}

//Prints a vector of float values, has the option of including a text
void PrintFloatVec(float* vec, int lenght, char* prompt){
	printf("%s \n", prompt);
	for (int i = 0; i<lenght; i++) printf("%f \t", vec[i]);
	printf("\n\n\n");
}

//Prints a vector of int values, has the option of including a text
void PrintInt8Vec(int8_t* vec, int lenght, char* prompt){
	printf("%s \n", prompt);
	for (int i = 0; i<lenght; i++) printf("%d \t", vec[i]);
	printf("\n\n\n");
}
