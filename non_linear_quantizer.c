#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>

#define MU_LAW_COMPANDER(mu, x) (sign(x) * (log(1 + mu * fabs(x)) / log(1 + mu)))
#define A_LAW_COMPANDER(alpha, x) (sign(x) * (log(1 + alpha * fabs(x)) / log(1 + alpha)))
#define MU_LAW_EXPANDER(mu, y) (sign(y) * (1.0 / mu) * ((pow(1.0 + mu, fabs(y)) - 1.0) / mu))
#define A_LAW_EXPANDER(alpha, y) (sign(y) * (1.0 / alpha) * ((pow(1.0 + alpha, fabs(y)) - 1.0) / alpha))

void NonLinearQuantization(uint8_t* out, float* in, int lenght, float* min, float* max, int type){
	switch(type){
		case 1:
			#pragma omp parallel for
			for(int i=0; i<lenght; i++) in[i]=MU_LAW_COMPANDER(255.0, in[i]);
			break;
		case 2:
			#pragma omp parallel for 
			for(int i=0; i<lenght; i++) in[i]=A_LAW_COMPANDER(87.6, in[i]);
			break;
		default:
			printf("\tERROR\t companding type not valid\n");
			return;
	}

	UniformRangedQuantization(out, in, lenght, min, max);
}

void NonLinearDequantization(float* out, uint8_t* in, int lenght, float min, float max, int type){	
	UniformRangedDequantization(out, in, lenght, min, max);
	
	switch(type){
		case 1:
			#pragma omp parallel for
			for(int i=0; i<lenght; i++) out[i]=MU_LAW_EXPANDER(255.0, in[i]);
			break;
		case 2:
			#pragma omp parallel for 
			for(int i=0; i<lenght; i++) out[i]=A_LAW_EXPANDER(87.6, in[i]);
			break;
		default:
			printf("\tERROR\t companding type not valid\n");
			return;
	}

}
