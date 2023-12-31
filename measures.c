#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>

struct q_val{
	float mean;
	float min;
	float max;
};

/* generates a random vector of float parallelizing the work	*
 * on more threads and using rand_r (a thread safe version of	*
 * time.h rand that requires an explicit seed) and an		*
 * upperbound. The values are then zero meaned.			*
 *								*
 * input:	pointer to float array, array lenght,		*
 *		lowerbound, upperbound				*/
void RandFloatGenerator(float* list, int lenght, float lowerbound, float upperbound){
	int i=0;
	unsigned int seed;
	float range = upperbound - lowerbound;
	float my_range = ((float)RAND_MAX)/range;
	
	#pragma omp parallel num_threads(thread_count)\
		default(none) shared(list, lenght, my_range, lowerbound) private(i, seed)
	{
		seed=rand();
		#pragma omp for
		for(i=0; i<lenght; i++)
			list[i] = (float) rand_r(&seed) / my_range + lowerbound;
	}
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

float NormalizedMSE(float*v1, float* v2, int lenght){
	float out = MeanSquaredError(v1,v2, lenght);
	float min, max, range;
	MinMax(v1, &min, &max);
	range = max - min;
	out = out/range;
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

void MinMax(float* vec, int lenght, float* min, float* max){
	float minimum = vec[0];
	float maximum = vec[0];
	#pragma omp parallel for default(none) shared(vec, lenght) \
		reduction(min: minimum) reduction(max: maximum)
	for(int i=1; i<lenght;i++){
		if(vec[i] < minimum)		minimum = vec[i];
		else if(vec[i] > maximum)	maximum = vec[i];
	}
	min = minimum;
	max = maximum;
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
