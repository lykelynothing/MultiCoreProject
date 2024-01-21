#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>

#include "tools.h"

#define MU		255
#define ALPHA		87.6


void MuLawCompander(float* in, size_t input_size){
	//some useful values to have stored (to not compute them every time)
	float ln_one_plus_mu = logf(1.0 + MU);
	
	for(int i = 0; i < input_size; i++)
		in[i] = sign(in[i]) * (logf(1 + MU * fabsf(in[i])) / ln_one_plus_mu);
}

void MuLawExpander(float* in, size_t input_size){
	//some useful values to have stored (to not compute them every time)
	float one_over_mu = 1.0 / MU;
	float one_plus_mu = MU + 1.0;
	
	for(int i = 0; i < input_size; i++)
		in[i] = sign(in[i]) * one_over_mu * (powf(one_plus_mu, fabsf(in[i])) - 1.0);
}

void ALawCompander(float* in, size_t input_size){
	//some useful values to have stored (to not compute them every time)
	float condition = 1.0 / ALPHA;
	float one_plus_ln_a = 1.0 + logf(ALPHA);
	float one_over_one_plus_ln_a = 1.0 / one_plus_ln_a;
	float alpha_over_one_plus_ln_a = ALPHA * one_over_one_plus_ln_a;

	for(int i = 0; i < input_size; i++){
		float abs = fabsf(in[i]);
		in[i] = (abs < condition) ? \
			sign(in[i]) * abs * alpha_over_one_plus_ln_a : \
			sign(in[i]) * ((1 + logf(ALPHA * abs)) / one_plus_ln_a);

	}
}

void ALawExpander (float* in, size_t input_size){
	//some useful values to have stored (to not compute them every time)
	float one_over_alpha = 1.0 / ALPHA;
	float one_plus_ln_a = 1.0 + logf(ALPHA);
	float ln_one_plus_alpha = logf(1.0 + ALPHA);
	float condition = 1.0 / one_plus_ln_a;
	
	for(int i = 0; i < input_size; i++){
		float abs = fabsf(in[i]);
		in[i] = (abs < condition) ? \
			sign(in[i]) * abs * one_over_alpha * one_plus_ln_a : \
			sign(in[i]) * one_over_alpha * powf(M_E, -1.0 + abs * one_plus_ln_a);
	}
}



float* RangeReducer(float* in, size_t input_size, float* min, float* max){
	float* out = malloc(input_size*sizeof(float));
	MinMax(in, input_size, min, max);
	float range = *max - *min;

	#pragma omp parallel for default(none) shared(out, in, min, range, input_size)
	for(int i = 0; i < input_size; i++)
		out[i] = ((in[i] - *min) * 2) / range - 1;
	
	return out;
}


void RangeRestorer(float* in, size_t input_size, float min, float max){
	float range = max - min;
	
	#pragma omp parallel for default(none) shared(in, input_size, min, range)
	for(int i = 0; i < input_size; i++)
		in[i] = ((in[i] + 1) * range) / 2 + min;
}


struct compressed* NormalizedSymmetricQuantization(float* in, size_t input_size){
	struct compressed* out = (struct compressed*) malloc(input_size*sizeof(struct compressed));
	
	float this_range = (REPR_RANGE - 1) / 2;
	for(int i = 0; i < input_size; i++)
		out[i].number = (uint64_t) ((in[i] + 1) * this_range);
	
	return out;
}

float* NormalizedSymmetricDequantization(struct compressed* in, size_t input_size){
	float* out = malloc(input_size*sizeof(float));
	
	float this_range = (REPR_RANGE - 1) / 2;
	for(int i = 0; i < input_size; i++)
		out[i] = ((float) in[i].number) / this_range - 1.0;

	return out;
}



//A law needs to be expanded bc it's wrong: it assumes a different formula for values less than 1/alpha
struct non_linear_quant* NonLinearQuantization(float* in, size_t input_size, int type){
	float min, max;
	float* temp = RangeReducer(in, input_size, &min, &max);
	
/*	char * p2 = "Range reduced vec:";
	printf("min: %f\t max: %f\n", min, max);
	PrintFloatVec(temp, input_size, p2);
*/
	switch(type){
		case 1:
			MuLawCompander(temp, input_size);
			break;
		case 2:
			ALawCompander(temp, input_size);
			break;
		default:
			printf("\tERROR\t companding type not valid\n");
			free(temp);
			return NULL;
	}
	
/*	char *p3 = "Companded vec:";
	PrintFloatVec(temp, input_size, p3);
*/
	struct non_linear_quant* out = (struct non_linear_quant*) malloc(sizeof(struct non_linear_quant));
	out->vec = NormalizedSymmetricQuantization(temp, input_size);
	out->min = min;
	out->max = max;
	out->type = type;

	free(temp);
	
	return out;
}



float* NonLinearDequantization(struct non_linear_quant* in, size_t input_size){	
	float* out = NormalizedSymmetricDequantization(in->vec, input_size);
/*	char* p1 = "dequantized (not expanded)";
	PrintFloatVec(out, input_size, p1);
*/
	switch(in->type){
		case 1:
			MuLawExpander(out, input_size);
			break;
		case 2:
			ALawExpander(out, input_size);
			break;
		default:
			printf("\tERROR\t companding type not valid\n");
			return NULL;
	}
/*	char* p2 = "a expanded";
	PrintFloatVec(out, input_size, p2);
*/
	RangeRestorer(out, input_size, in->min, in->max);

	return out;
}

