#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define EPSILON 0.00001			//1e-5
#define ITERATIONS 100

void LloydMaxQuantizer(uint8_t* out, float* codebook, float* in, int lenght, float* min, float* max){
	
	MinMax(in, min, max);
	RandFloatGenerator(codebook, 256, min, max);

	for(int i = 0; i<ITERATIONS; i++){
		#pragma omp parallel for default(none) shared(lenght, codebook, in, out)
		for(int j=0; j<lenght; j++){
			out[j] = NearestCodeword(in[j], codebook, lenght, 256);
		}

		UpdateCodebook(in, out, codebook, 256);
	}

}


uint8_t NearestCodeword(float element, float* codebook, int codebook_len){
	uint8_t nearest_cluster = 0;
	double min_dist = INFINITY;
	
	for(int i=0; i<codebook_len; i++){
		double dist = fabsf(codebook[i]-element);
		if (dist<min_dist){
			min_dist=dist;
			nearest_cluster = i;
		}
	}
	return nearest_cluster;
}


void UpdateCodebook(float* input, uint8_t* assignments, float* codebook, int input_lenght, int codebook_len){
	int counts [codebook_lenght];

	for (int i = 0; i< codebook_lenght; i++){
		codebook[i] = 0;
		counts[i] = 0;
	}

	for (int i = 0; i<input_lenght; i++){
		int cluster = (int) assignments[i];
		counts[cluster]++;
		codebook[cluster]+=input[i];
	}

	for (int i = 0; i<input_lenght; i++){
		if (counts[i]!=0) codebook[i]/=counts[i];
	}
}

