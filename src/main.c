#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "lloyd_max_quantizer.h"
#include "non_linear_quantizer.h"
#include "uniform_quantizer.h"
#include "tools.h"

int main(int argc, char** argv){
	srand(time(NULL));
	size_t dim;

	switch(argc){
		case 2:
			dim = (size_t) strtol(argv[1], NULL, 10);
			break;
		default:
			dim= 100;
	}

	float* vector = RandFloatGenerator(dim, -1000.0, 1000.0);
	
	struct uint8_vec_lm * quantized = LloydMaxQuantizer(vector, dim);

	float* dequantized = LloydMaxDequantizer(quantized, dim, 256);

	PrintFloatVec(vector, dim, "");
	
	char* p1 = "the quantized vector is";
	PrintInt8Vec(quantized->vec, dim, p1);

	char* p2 = "the dequantized vector is";
	PrintFloatVec(dequantized, dim, p2);

	printf("the MSE is: \t %f \n", MeanSquaredError(vector, dequantized, dim));

	free(vector);
	free(quantized->vec);
	free(quantized->codebook);
	free(quantized);
	free(dequantized);

	return 0;
	
/*	float* list = malloc(dim*sizeof(float));

	float scale = 10.0;

	RandFloatGenerator(list, dim, 256.0*scale, -256.0*scale);

 	struct q_val datas;
	VectorDatas(list, dim, &datas); 

	int8_t* quantized_data = malloc(dim*sizeof(int8_t));
	
	UniformAffineQuantization(quantized_data, list, dim, scale, 0);

	float* dequantized_data = malloc(dim*sizeof(float));

	UniformAffineDequantization(dequantized_data, quantized_data, dim, scale, 0);

	char* p1 = "The float list is:";
	PrintFloatVec(list, dim,p1);

	printf("Mean: %f \tMin: %f\t Max: %f\n\n\n", datas.mean, datas.min, datas.max);
	
	char* p2 = "The quantized int8 list is:";
	PrintInt8Vec(quantized_data, dim, p2);

	char* p3 = "The dequantized float list is:";
	PrintFloatVec(dequantized_data, dim, p3);
	
	printf("The mean squared error between the two lists is: %f \n", MeanSquaredError(list, dequantized_data, dim));
	free(list);
	free(quantized_data);
	free(dequantized_data);
	return 0*/ 

}

