#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define ITER 100

void VectorQuantizer(uint8_t* out, float* vectorbook, float* in, int lenght, int vector_lenght, float* min, float* max){
	if (lenght%vector_lenght!=0) return;
	
	MinMax(in, min, max);
	RandFloatGenerator(vectorbook, 256*vector_lenght, min, max);

	int vector_quantity = lenght/vector_lenght);

	for (int i = 0; i<ITER; i++){
		for(int j = 0; j < vector_quantity; j++)
			out[j] = NearestCodevector(in, vectorbook, vector_lenght, 256);

		UpdateVectorbook(in, out, vectorbook, );

	}

}


uint8_t NearestCodevector(float* vector, float* vectorbook, int vector_lenght, int vectorbook_lenght){
	uint8_t nearest_vector = 0;
	float min_dist = INFINITY;

	for (int i = 0; i<vectorbook_lenght; i++){
		float dist = 0;
		for (int j = 0; j<vector_lenght; j++)
			dist += vectorbook[i*vector_lenght + j] * vectorbook[i*vector_lenght + j];
		dist = sqrtf(dist);
		if (dist<min_dist){
			min_dist = dist;
			nearest_vector = i;
		}
	}

	return nearest_vector;
}


void UpdateVectorbook(float* input, uint8_t* assignments, float* vectorbook, int input_lenght, int vectorbook_lenght, int vector_lenght){
	int counts[vectorbook_lenght];

	int vector_quantity = input_lenght/vector_lenght;

	for (int i = 0; i<vectorbook_lenght; i++){
		counts[i] = 0;
		for (int j = 0; j<vector_lenght; j++)
			vectorbook[i*vector_lenght + j] = 0;
	}

	for (int i = 0; i<vector_quantity; i++){
		int cluster = (int) assignment[i];
		counts[cluster]++;
		for(int j = 0; j<vector_lenght; j++)
			vectorbook[cluster*vector_lenght + j] += input[i*vector_lenght + j];
	}

	for (int i = 0; i<vectorbook_lenght; i++){
		if(counts[i]!=0){
			for (int j = 0; j<vector_lenght; j++)
				vectorbook[i*vector_lenght + j] /= counts[i];
		}
	}
}


void VectorDequantizer(float* output, float* vectorbook, uint8_t* assignments, float output_lenght, int codebook_lenght, int vector_lenght){
	for(int i = 0; i < output_lenght; i++){
		int cluster_index = (int) assignments[i];

		if (cluster_index>=0 && cluster_index < codebook_lenght) {
			for  (int j = 0; j < vector_lenght; j++)
				output[i*vector_lenght + j] = vectorbook[cluster_index*vector_lenght + j];
		}
		else printf("ERROR: invalid cluster index %d at position %d\n", cluster_index, i);
	}
}

