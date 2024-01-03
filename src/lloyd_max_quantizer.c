#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

#include "tools.h"

#define EPSILON 0.00001			//1e-5
#define ITERATIONS 1000


//				ASSIGNMENT STEP
/* The function takes as input a float value and return as output a	*
 * uint8_t value corresponding to the index of the closest (abs value	*
 * distance). The codebook_len parameter should be equal to the range	*
 * of values of the quantized result.					*/
uint8_t NearestCodeword(float element, float* codebook, int codebook_len){
	uint8_t nearest_cluster = 0;
	float min_dist = INFINITY;
	
	for(int i=0; i<codebook_len; i++){
		float dist = fabsf(codebook[i]-element);
		if (dist<min_dist){
			min_dist=dist;
			nearest_cluster = i;
		}
	}
	return nearest_cluster;
}


//				UPDATE STEP
/* The function takes as input the original input vector, the output	*
 * (assignment) vector, the codebook, the input lenght and the codebook	*
 * lenght.								*
 * It reset the codebook to 0s to then updates it with the mean of the	*
 * assigned vectors.							*/
void UpdateCodebook(float* input, uint8_t* assignments, float* codebook, size_t input_size, size_t codebook_size){
	//creates a counts vector that will count how many vector there are for each assignment
	int counts [codebook_size];
	
	//set the counts to 0 and reset the codebook
	for (int i = 0; i< codebook_size; i++){
		codebook[i] = 0;
		counts[i] = 0;
	}
	
	//each component of the input vector is added to the corresponding codebook
	//(and the corresponding count will be updated)
	for (int i = 0; i<input_size; i++){
		int cluster = (int) assignments[i];
		counts[cluster]++;
		codebook[cluster]+=input[i];
	}

	//for all the codebook value (with count > 0) the mean is taken
	for (int i = 0; i<codebook_size; i++){
		if (counts[i]!=0) codebook[i]/=counts[i];
	}
}


struct uint8_vec_lm * LloydMaxQuantizer(float* in, size_t input_size){
	//define and allocate memory for struct and vector inside the struct
	//memory for codebook will be allocated by RandFloatGenerator
	struct uint8_vec_lm * out = (struct uint8_vec_lm*) malloc(sizeof(struct uint8_vec_lm));
	out->vec = (uint8_t*) malloc(input_size*sizeof(uint8_t));
 
	MinMax(in, input_size, &(out->min), &(out->max));
	//will allocate memory for the codebook and randomly generate it
	//size of 256 is chosen due to uint8_t range (you can't codify more codeword)
	out->codebook = RandFloatGenerator(256, out->min, out->max);

	//applies the lloyd-max method for ITERATIONS times (to be fine tuned)
	for(int i = 0; i < ITERATIONS; i++){
		//#pragma omp parallel for default(none) shared(lenght, codebook, in, out)
		for(int j = 0; j < input_size; j++)
			//Assign to each value of the input vector a corresponding codeword assignment
			out->vec[j] = NearestCodeword(in[j], out->codebook, 256);
		//Updates the codebook (array of codeword) by taking the mean of all the vectors assigned
		UpdateCodebook(in, out->vec, out->codebook, input_size, 256);
	}

	return out;

}


float * LloydMaxDequantizer(struct uint8_vec_lm * in, size_t input_size, size_t codebook_size){
	
	float* out = malloc(input_size*sizeof(float));

	for(int i = 0; i < input_size; i++){
		int cluster_index = (int) in->vec[i];

		if (cluster_index>=0 && cluster_index < codebook_size) out[i] = in->codebook[cluster_index];
		else printf("ERROR: invalid cluster index %d at position %d\n", cluster_index, i);
	}

	return out;
}

