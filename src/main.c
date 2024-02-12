#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "homomorphic_quantizer.h"
#include "lloyd_max_quantizer.h"
#include "non_linear_quantizer.h"
#include "uniform_quantizer.h"
#include "tools.h"
#include "collectives.h"

int main(int argc, char** argv){
  size_t dim;
	switch(argc){
		case 2:
			dim = (size_t) strtol(argv[1], NULL, 10);
			break;
		default:
			dim = 6;
	}
  int var[2];
  GetEnvVariables(var);
  int my_rank, comm_sz;
	
  MPI_Init(NULL, NULL);
	MPI_Pcontrol(2);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  srand(time(NULL) + my_rank);
	
  float* in = RandFloatGenerator(dim, -500.0, 500.0);
  if(my_rank==0) printf("\nORIGINAL VECTORS\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(in, dim, my_rank, comm_sz, MPI_COMM_WORLD, FLOAT);
  
  struct unif_quant* q = HomomorphicQuantization(in, dim, MPI_COMM_WORLD); 
  if(my_rank==0) printf("\nQUANTIZED VECTORS\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(q->vec, dim, my_rank, comm_sz, MPI_COMM_WORLD, UINT8);
  
  float* d = HomomorphicDequantization(q->vec, q->min, q->max, comm_sz, dim, 0);
  if(my_rank==0) printf("\nDEQUANTIZED VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(d, dim, my_rank, comm_sz, MPI_COMM_WORLD, FLOAT);

  float* out = malloc(dim * sizeof(float));
  MPI_Allreduce(in, out, dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  if(my_rank==0) printf("\nALLRED VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(out, dim, my_rank, comm_sz, MPI_COMM_WORLD, FLOAT);
  
  float* control = malloc(dim * sizeof(float));
	PMPI_Allreduce(in, control, dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  if(my_rank==0) printf("\nCONTROL VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(control, dim, my_rank, comm_sz, MPI_COMM_WORLD, FLOAT);

  free(in);
  free(q->vec);
  free(q);
  free(d);
  free(out);
  free(control);

	MPI_Finalize();

  return 0;
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

  return 0;
*/
}

