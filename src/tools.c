#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#include "tools.h"


MPI_Datatype UnifQuantType(int array_size){
    MPI_Datatype MPI_Unif_quant;
    MPI_Datatype types[3] = {MPI_UINT8_T, MPI_FLOAT, MPI_FLOAT};
    int block_lengths[3] = {array_size, 1, 1};  
    MPI_Aint offsets[3];
    offsets[0] = offsetof(struct unif_quant, vec);
    offsets[1] = offsetof(struct unif_quant, min);
    offsets[2] = offsetof(struct unif_quant, max);

    MPI_Type_create_struct(3, block_lengths, offsets, types, &MPI_Unif_quant);
    MPI_Type_commit(&MPI_Unif_quant);

    return MPI_Unif_quant;
}


MPI_Datatype NonLinearQuantType(int array_size){
    MPI_Datatype MPI_Non_linear_quant;
    MPI_Datatype types[4] = {MPI_UINT8_T, MPI_FLOAT, MPI_FLOAT, MPI_INT};
    int block_lengths[4] = {array_size, 1, 1, 1}; 
    MPI_Aint offsets[4];
    offsets[0] = offsetof(struct non_linear_quant, vec);
    offsets[1] = offsetof(struct non_linear_quant, min);
    offsets[2] = offsetof(struct non_linear_quant, max);
    offsets[3] = offsetof(struct non_linear_quant, type);

    MPI_Type_create_struct(4, block_lengths, offsets, types, &MPI_Non_linear_quant);
    MPI_Type_commit(&MPI_Non_linear_quant);

    return MPI_Non_linear_quant;
}


MPI_Datatype LloydMaxQuantType(int array_size){
    MPI_Datatype MPI_Lloyd_max_quant;
    MPI_Datatype types[4] = {MPI_UINT8_T, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    int block_lengths[4] = {array_size, 1, 1, REPR_RANGE};
    MPI_Aint offsets[4];
    offsets[0] = offsetof(struct lloyd_max_quant, vec);
    offsets[1] = offsetof(struct lloyd_max_quant, min);
    offsets[2] = offsetof(struct lloyd_max_quant, max);
    offsets[3] = offsetof(struct lloyd_max_quant, codebook);

    MPI_Type_create_struct(4, block_lengths, offsets, types, &MPI_Lloyd_max_quant);
    MPI_Type_commit(&MPI_Lloyd_max_quant);

    return MPI_Lloyd_max_quant;
}


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
void PrintInt8Vec(uint8_t* vec, size_t lenght, char* prompt){
	printf("%s \n", prompt);
	for (int i = 0; i < lenght; i++) printf("%llu \t", vec[i]);
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

void swap(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

int partition(float arr[], int low, int high) {
    float pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (arr[j] < pivot) {
            ++i;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

void quicksort(float arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

void Quicksort(float* input, size_t dim){
  quicksort(input, 0, (int) dim -1);
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

