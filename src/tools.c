#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>
#include <math.h>

#include "tools.h"

/* Generates a random vector of float parallelizing the work	*
 * on more threads and using rand_r (a thread safe version of	*
 * time.h rand that requires an explicit seed), an upperbound	*
 * and a lowerbound.						*/
float* RandFloatGenerator(size_t lenght, float lowerbound, float upperbound){
	float* list = malloc(lenght*sizeof(float));
	
	int i=0;
	unsigned int seed;
	
	float range = ((float)RAND_MAX) / (upperbound - lowerbound);
	
	#pragma omp parallel default(none) shared(list, lenght, range, lowerbound) private(i, seed)
	{
		seed=rand();
		#pragma omp for
		for(i=0; i<lenght; i++)
			list[i] = (float) rand_r(&seed) / range + lowerbound;
	}

	return list;
}


/* Writes scan in a parallel way (through the use of reduction	*
 * clauses) through the vector to find the minimum and maximum	*
 * values.							*
 * those values are then stored.				*
 * In the code it is often used with the uint8_vec struct to	*
 * store those value for the quantization interval setup and	*
 * for the dequantization itself.				*/
void MinMax(float* vec, size_t lenght, float* min, float* max){
	
	float minimum = vec[0];
	float maximum = vec[0];

	#pragma omp parallel for default(none) shared(vec, lenght) \
		reduction(min: minimum) reduction(max: maximum)
	for(int i = 1; i < lenght; i++){
		if(vec[i] < minimum)		minimum = vec[i];
		else if(vec[i] > maximum)	maximum = vec[i];
	}
	
	*min = minimum;
	*max = maximum;
}


//Prints a vector of float values, has the option of including a text
void PrintFloatVec(float* vec, size_t lenght, char* prompt){
	printf("%s \n", prompt);
	for (int i = 0; i < lenght; i++) printf("%f \t", vec[i]);
	printf("\n\n\n");
}

//Prints a vector of int values, has the option of including a text
void PrintInt64Vec(uint64_t* vec, size_t lenght, char* prompt){
	printf("%s \n", prompt);
	for (int i = 0; i < lenght; i++) printf("%lld \t", vec[i]);
	printf("\n\n\n");
}

void PrintVectorbook(struct vector_quant* in, char* prompt){
	printf("%s", prompt);
	for (int i = 0; i < 256; i++){
		size_t vector_size = in->vec_size;
		printf("\nvector[%d]\t", i);
		for(int j = 0; j < vector_size; j++)
			printf("\t%f\t", in->vectorbook[i*vector_size + j]);
	}
	printf("\n\n\n");
}


float sign(float x){
	if (x >=0) return 1.0;
	else return -1.0;
}

//MeanSquaredError between two float arrays
float MeanSquaredError(float* v1, float* v2, size_t lenght){
	float out=0;
//	#pragma omp parallel for default(none) shared(v1, v2, lenght) reduction(+:out)
	for(int i = 0; i<lenght; i++)
		out+= (v1[i]-v2[i])*(v1[i]-v2[i]);

	out = out/(float)lenght;
	
	return out;
}

float NormalizedMSE(float*v1, float* v2, size_t lenght){
	float out = MeanSquaredError(v1,v2, lenght);
	float min, max, range;
	MinMax(v1, lenght, &min, &max);
	range = max - min;
	out = out/range;
	return out;
}

/*
//Mean of a float array, can be used as offset of affine reduction
float VectorMean(float* vec, int lenght){
	float out = 0;
	float len = (float) len;
	#pragma omp parallel for default(none) shared(vec,lenght, len) reduction(+:out)
	for(int i = 0; i<lenght; i++)
		out+= vec[i]/len;

	return out;
}*/

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

/*void VectorDatas(float* vec, int lenght, struct q_val* out){
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
}*/

