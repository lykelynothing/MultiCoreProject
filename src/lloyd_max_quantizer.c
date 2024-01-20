#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#include "tools.h"

#define ITERATIONS 5


//				ASSIGNMENT STEP
/* The function takes as input a float value and return as output a	*
 * uint64_t value corresponding to the index of the closest (abs value	*
 * distance).								*/
uint64_t NearestCodeword(float element, float* codebook){
	uint64_t nearest_cluster = REPR_RANGE + 1;
	float min_dist = INFINITY;
	
	for(int i=0; i<REPR_RANGE; i++){
		float dist = fabsf(codebook[i]-element);
		if (dist<min_dist){
			min_dist=dist;
			nearest_cluster = i;
		}
	}

	if (nearest_cluster != REPR_RANGE + 1)
		return nearest_cluster;
	else{
		printf("ERROR! Nearest cluster = NULL\n");
		return 0;
	}
}


//				UPDATE STEP
/* The function takes as input the original input vector, the output	*
 * (assignment) vector, the codebook, the input lenght and the codebook	*
 * lenght.								*
 * It reset the codebook to 0s to then updates it with the mean of the	*
 * assigned vectors.							*/
void UpdateCodebook(float* input, struct compressed* assignments, float* codebook, size_t input_size){
	//creates a counts vector that will count how many vector there are for each assignment
	int counts [REPR_RANGE];
	
	//set the counts to 0 and reset the codebook
	for (int i = 0; i < REPR_RANGE; i++){
		codebook[i] = 0;
		counts[i] = 0;
	}
	
	//each component of the input vector is added to the corresponding codebook
	//(and the corresponding count will be updated)
	for (int i = 0; i < input_size; i++){
		int cluster = (int) assignments[i].number;
		counts[cluster]++;
		codebook[cluster]+=input[i];
	}

	//for all the codebook value (with count > 0) the mean is taken
	for (int i = 0; i < REPR_RANGE; i++){
		if (counts[i]!=0) codebook[i]/=counts[i];
	}
}

// TODO: write what this input size corresponds to (bytes/count)
struct lloyd_max_quant * LloydMaxQuantizer(float* in, size_t input_size){
	//define and allocate memory for struct and vector inside the struct
	//memory for codebook will be allocated by RandFloatGenerator
	struct lloyd_max_quant * out = (struct lloyd_max_quant*) malloc(sizeof(struct lloyd_max_quant));
	out->vec = (struct compressed*) malloc(input_size*sizeof(struct compressed));
 
	MinMax(in, input_size, &(out->min), &(out->max));
	//will allocate memory for the codebook and randomly generate it
	out->codebook = RandFloatGenerator(REPR_RANGE, out->min, out->max);

	//applies the lloyd-max method for ITERATIONS times (to be fine tuned)
	for(int i = 0; i < ITERATIONS; i++){
		//#pragma omp parallel for default(none) shared(lenght, codebook, in, out)
		for(int j = 0; j < input_size; j++)
			//Assign to each value of the input vector a corresponding codeword assignment
			out->vec[j].number = NearestCodeword(in[j], out->codebook);
		//Updates the codebook (array of codeword) by taking the mean of all the vectors assigned
		UpdateCodebook(in, out->vec, out->codebook, input_size);
	}

	return out;
}


float * LloydMaxDequantizer(struct lloyd_max_quant * in, size_t input_size){
	
	float* out = malloc(input_size*sizeof(float));

	for(int i = 0; i < input_size; i++){
		int cluster_index = (int) in->vec[i].number;

		if (cluster_index>=0 && cluster_index < REPR_RANGE) out[i] = in->codebook[cluster_index];
		else printf("ERROR: invalid cluster index %d at position %d\n", cluster_index, i);
	}

	return out;
}

