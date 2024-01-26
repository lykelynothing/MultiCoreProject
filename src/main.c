#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "lloyd_max_quantizer.h"
#include "non_linear_quantizer.h"
#include "uniform_quantizer.h"
#include "tools.h"
#include "collectives.h"

int BITS = 8;
int REPR_RANGE = 1 << 8;

int main(int argc, char** argv){
	srand(time(NULL));
	size_t dim;
	int my_rank, comm_sz;
    char * bits_env_var;

	bits_env_var = getenv("BITS_VAR");
	if (bits_env_var != NULL){
		int bits_env_int = atoi(bits_env_var);
		BITS = bits_env_int;
		REPR_RANGE = (1 << BITS);
	} else {
		printf("\n Error : No environmental variable BITS_VAR found\n");
		return 0;
	}


	switch(argc){
		case 2:
			dim = (size_t) strtol(argv[1], NULL, 10);
			break;
		default:
			dim = 1000;
	}

	MPI_Init(NULL, NULL);
	MPI_Pcontrol(2);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	//float * in = RandFloatGenerator(dim, -1000.0, 1000.0);
	dim = 100;
	float * in = malloc(sizeof(float) * dim);
	for (int i = 0; i < dim; i ++)
		in[i] = 0;
	uint8_t * out = malloc(sizeof(uint8_t) * dim);
	// TODO dequantize out and check if it's right
	MPI_Allreduce((void *) in, (void *) out, dim, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
	
	free(in);
	free(out);

	MPI_Finalize();
	 
/*
	srand(time(NULL));

	size_t dim = 1000000;

	float* original = RandFloatGenerator(dim, -10000, 10000);

	struct lloyd_max_quant* quantized = LloydMaxQuantizer(original, dim );

	float* dequantized = LloydMaxDequantizer(quantized, dim);

	printf("Here's the results: \n INDEX \t ORIGINAL \t QUANT \t DEQUANT \n");
	for(int i = 0; i < dim; i++)
		printf("%d:\t %f \t %d \t %f \n", i, original[i], quantized->vec[i].number, dequantized[i]);
	

	printf("MSE is: %f\n", MeanSquaredError(original, dequantized, dim));

	free(original);
	free(quantized);
	free(dequantized);
*/
	return 0;
}

