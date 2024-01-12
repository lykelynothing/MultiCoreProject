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
	srand(time(NULL));
	size_t dim;
	int my_rank, comm_sz;

	switch(argc){
		case 2:
			dim = (size_t) strtol(argv[1], NULL, 10);
			break;
		default:
			dim = 1000;
	}

	dim++;
	MPI_Init(NULL, NULL);
	MPI_Pcontrol(2);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	float * in = RandFloatGenerator(dim, -1000.0, 1000.0);

	// god please make this work
	uint8_t * out = malloc(sizeof(struct compressed) * dim);
	// inshallah

	MPI_Allreduce((void *) in, (void *) out, dim, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

	if (my_rank == 0){
		for (int i = 0; i < dim; i++){
			printf("out[%d] = %d \t in[%d] = %f \n", i, out[i], i, in[i]);
		}
	}

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

	free(in);
	free(out);

	MPI_Finalize();

	return 0;
}

