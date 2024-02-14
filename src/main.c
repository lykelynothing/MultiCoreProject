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
			dim = 8;
	}

  int my_rank, comm_sz;
  MPI_Init(NULL, NULL);
	MPI_Pcontrol(2);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  if (dim < comm_sz){
    printf("ERROR!! The dimension of the array must be bigger than comm_sz\n");
    MPI_Finalize();
    return 0;
  }

  srand(time(NULL) + my_rank);
	
  float* in = RandFloatGenerator(dim, -500.0, 500.0);
  if(my_rank==0) printf("\nORIGINAL VECTORS\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(in, dim, my_rank, comm_sz, MPI_COMM_WORLD, FLOAT);
  
  /*struct unif_quant* q = HomomorphicQuantization(in, dim, MPI_COMM_WORLD); 
  if(my_rank==0) printf("\nQUANTIZED VECTORS\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(q->vec, dim, my_rank, comm_sz, MPI_COMM_WORLD, UINT8);
  */
  float* ring_allred = malloc(dim * sizeof(float));
	MPI_Allreduce(in, ring_allred, dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  if(my_rank==0) printf("\nRING_ALLRED VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(ring_allred, dim, my_rank, comm_sz, MPI_COMM_WORLD, FLOAT);

  /*float* dequantized = HomomorphicDequantization(q->vec, q->min, q->max, comm_sz, dim, 0);
  if(my_rank==0) printf("\nDEQUANTIZED AFTER RING ALLRED VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(dequantized, dim, my_rank, comm_sz, MPI_COMM_WORLD, FLOAT);
  
  float* reduced = malloc(dim*sizeof(float));
  MPI_Allreduce(in, reduced, dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  if(my_rank==0) printf("\nREDUCED VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(reduced, dim, my_rank, comm_sz, MPI_COMM_WORLD, FLOAT);
  */
  float* control = malloc(dim * sizeof(float));
	PMPI_Allreduce(in, control, dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  if(my_rank==0) printf("\nCONTROL VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(control, dim, my_rank, comm_sz, MPI_COMM_WORLD, FLOAT);

  free(in);
  //free(q->vec);
  //free(q);
  free(ring_allred);
  //free(dequantized);
  //free(reduced);
  free(control);

	MPI_Finalize();

  return 0;
}

