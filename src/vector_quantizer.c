#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#include "tools.h"

#define ITER 100


uint64_t NearestCodevector(float* vector, float* vectorbook, size_t vector_size, int index){
	uint64_t nearest_vector = 0;
	float min_dist = INFINITY;

	for (uint64_t i = 0; i < REPR_RANGE; i++){
		float dist = 0;
		for (int j = 0; j < vector_size; j++){
			dist += ((vector[index + j] - vectorbook[i*vector_size + j]) * (vector[index + j] - vectorbook[i*vector_size + j]));
		}
		dist = sqrtf(dist);

		if (dist<min_dist){
			min_dist = dist;
			nearest_vector = i;
		}
	}
	return nearest_vector;
}


void UpdateVectorbook(float* input, struct compressed* assignments, float* vectorbook, size_t input_size, size_t vector_size){
	int counts[REPR_RANGE];

	size_t vector_number = input_size / vector_size;

	for (int i = 0; i < REPR_RANGE; i++){
		counts[i] = 0;
		for (int j = 0; j < vector_size; j++)
			vectorbook[i * vector_size + j] = 0;
	}

	for (int i = 0; i < vector_number; i++){
		int cluster = (int) assignments[i].number;
		counts[cluster]++;
		for(int j = 0; j < vector_size; j++)
			vectorbook[cluster*vector_size + j] += input[i*vector_size + j];
	}

	for (int i = 0; i < REPR_RANGE; i++){
		if(counts[i]!=0){
			for (int j = 0; j<vector_size; j++)
				vectorbook[i*vector_size + j] /= counts[i];
		}
	}
}


struct vector_quant* LBGVectorQuantizer(float* in, size_t input_size, size_t vector_size){
	//check if you can divide the original vector into subvectors of dimension vector_size
	if (input_size%vector_size!=0){
		printf("ERROR: input_size %ld not divisible by vector_size %ld", \
			input_size, vector_size);
		return NULL;
	}
	
	//allocate memory for the struct and sets its value, starting with a randomized starting vectorbook
	struct vector_quant* out = (struct vector_quant*) malloc(sizeof(struct vector_quant));
	out->vec_size = vector_size;
	MinMax(in, input_size, &(out->min), &(out->max));
	out->vectorbook = RandFloatGenerator(REPR_RANGE*vector_size, out->min, out->max);
	out->vec = (struct compressed*) malloc(REPR_RANGE*sizeof(struct compressed));
	
	//needed for the size of  innerloop of LBG algorithm
	size_t vector_number = input_size / vector_size;

	for (int i = 0; i<ITER; i++){
	//	#pragma omp parallel for default(none) shared(in, out, vector_number, vector_size)
		for(int j = 0; j < vector_number; j++)
			out->vec[j].number = NearestCodevector(in, out->vectorbook, vector_size, j);
		UpdateVectorbook(in, out->vec, out->vectorbook, input_size, out->vec_size);
	}

	return out;
}



float* LBGVectorDequantizer(struct vector_quant* in, size_t input_size){
	float* out = malloc(input_size*sizeof(float));
	
	size_t vector_size = in->vec_size;
	size_t vector_number = input_size / vector_size;

	for(int i = 0; i < vector_number; i++){
		uint64_t cluster_index = in->vec[i].number;
		for  (int j = 0; j < vector_size; j++)
			out[i*vector_size + j] = in->vectorbook[cluster_index*vector_size + j];
	}

	return out;
}

