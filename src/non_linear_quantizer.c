#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>

#include "tools.h"
#include "uniform_quantizer.h"

#define MU_LAW_COMPANDER(mu, x) (sign(x) * (log(1 + mu * fabsf(x)) / log(1 + mu)))
#define A_LAW_COMPANDER(alpha, x) (sign(x) * (log(1 + alpha * fabsf(x)) / log(1 + alpha)))
#define MU_LAW_EXPANDER(mu, y) (sign(y) * (1.0 / mu) * ((pow(1.0 + mu, fabsf(y)) - 1.0) / mu))
#define A_LAW_EXPANDER(alpha, y) (sign(y) * (1.0 / alpha) * ((pow(1.0 + alpha, fabsf(y)) - 1.0) / alpha))



//for both mu law and a law you need values in [-1,1], so we need to translate the vector into  the [-1,1] interval
//A law needs to be expanded bc it's wrong: it assumes a different formula for values less than 1/alpha
struct uint8_vec * NonLinearQuantization(float* in, size_t input_size, int type){
	float* temp = malloc(input_size*sizeof(float));

	switch(type){
		case 1:
			#pragma omp parallel for
			for(int i = 0; i < input_size; i++) temp[i]=MU_LAW_COMPANDER(255.0, in[i]);
			break;
		case 2:
			#pragma omp parallel for 
			for(int i = 0; i < input_size; i++) temp[i]=A_LAW_COMPANDER(87.6, in[i]);
			break;
		default:
			printf("\tERROR\t companding type not valid\n");
			return NULL;
	}

	struct uint8_vec* out = UniformRangedQuantization(temp, input_size);

	free(temp);

	return out;
}



float* NonLinearDequantization(struct uint8_vec* in, size_t input_size, int type){	
	float* out = UniformRangedDequantization(in, input_size);
	
	switch(type){
		case 1:
			#pragma omp parallel for
			for(int i = 0; i < input_size; i++) out[i]=MU_LAW_EXPANDER(255.0, out[i]);
			break;
		case 2:
			#pragma omp parallel for 
			for(int i = 0; i < input_size; i++) out[i]=A_LAW_EXPANDER(87.6, out[i]);
			break;
		default:
			printf("\tERROR\t companding type not valid\n");
			return NULL;
	}
	
	return out;
}

