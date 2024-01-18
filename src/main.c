#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "vector_quantizer.h"
#include "lloyd_max_quantizer.h"
#include "non_linear_quantizer.h"
#include "uniform_quantizer.h"
#include "tools.h"
#include "collectives.h"

int main(int argc, char** argv){
/*	srand(time(NULL));
	size_t dim;
	int my_rank, comm_sz;

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

	float * in = RandFloatGenerator(dim, -1000.0, 1000.0);

	// god please make this work
	uint8_t * out = malloc(sizeof(struct compressed) * dim);
	// inshallah

	MPI_Allreduce((void *) in, (void *) out, dim, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

/*	if (my_rank == 0){
		for (int i = 0; i < dim; i++){
			printf("out[%d] = %d \t in[%d] = %f \n", i, out[i], i, in[i]);
		}
	} */

	free(in);
	free(out);

	MPI_Finalize();
	*/ 

	srand(time(NULL));

	size_t dim = 1000;

	float* original = RandFloatGenerator(dim, -10000, 10000);

	struct unif_quant* quantized = UniformRangedQuantization(original, dim);

	float* dequantized = UniformRangedDequantization(quantized, dim);

	printf("Here's the results: \n INDEX \t ORIGINAL \t QUANT \t DEQUANT \n");
	for(int i = 0; i < dim; i++)
		printf("%d:\t %f \t %d \t %f \n", i, original[i], quantized->vec[i].number, dequantized[i]);
	
	printf("MSE is: %f\n", MeanSquaredError(original, dequantized, dim));

	free(original);
	free(quantized);
	free(dequantized);

	return 0;
}

