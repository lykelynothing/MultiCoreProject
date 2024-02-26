#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "tools.h"
#include "collectives.h"



int main(int argc, char** argv){
  size_t dim;
	switch(argc){
		case 2:
			dim = (size_t) strtol(argv[1], NULL, 10);
			break;
		default:
			dim = 100;
      break;
	}

  float NMSE;  

  double start, end;
  double loc_elapsed;
  double cpu_time;

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
	
  float* in = RandFloatGenerator(dim, -500.0, 500); 
  if(my_rank==0) printf("\nORIGINAL VECTORS\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(in, dim, my_rank, comm_sz, FLOAT);
  

  float* control = malloc(dim * sizeof(float));
  PMPI_Allreduce(in, control, dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  if(my_rank==0) printf("\nCONTROL VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(control, dim, my_rank, comm_sz, FLOAT);
  
  float* out = malloc(dim * sizeof(float));
  
  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();
	MPI_Allreduce(in, out, dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  end = MPI_Wtime();
  loc_elapsed = end - start;
  PMPI_Reduce(&loc_elapsed, &cpu_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  
  if(my_rank==0) printf("\nALLRED VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(out, dim, my_rank, comm_sz, FLOAT);
  /*

  float* dequantized = HomomorphicDequantization(q->vec, q->min, q->max, comm_sz, dim, 0);
  if(my_rank==0) printf("\nDEQUANTIZED AFTER RING ALLRED VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(dequantized, dim, my_rank, comm_sz, MPI_COMM_WORLD, FLOAT);
  
  float* reduced = malloc(dim*sizeof(float));
  MPI_Allreduce(in, reduced, dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  if(my_rank==0) printf("\nREDUCED VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(reduced, dim, my_rank, comm_sz, MPI_COMM_WORLD, FLOAT);
  */


/*
  if (my_rank == 0) 
    printf("\nTime Elapsed : \n");

  MPI_Barrier(MPI_COMM_WORLD);
  printf("\n\nRank %d : %lf \n", my_rank, cpu_time);

*/
  if (my_rank == 0){
    NMSE = NormalizedMSE(out, control, dim);
    printf("%f\n", NMSE);
    printf("%lf\n", cpu_time);
    printf("%ld\n", dim);
    } 



  free(in);
  //free(q->vec);
  //free(q);
  free(out);
  //free(dequantized);
  //free(reduced);
  free(control);

	MPI_Finalize();

  return 0;
}

