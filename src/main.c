#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#include "tools.h"
#include "collectives.h"


int main(int argc, char** argv){
  size_t dim;
	switch(argc){
		case 2:{
			dim = (size_t) strtol(argv[1], NULL, 10);
			break;
    }
		default:{
			dim = 100;
      break;
    }
	}

  //TIMING VANILLA NOTQUANTIZED ALLREDUCE
  double start_v, end_v;
  double loc_elapsed_v;
  double cpu_time_v;
  //TIMING CUSTOM QUANTIZED ALLREDUCE
  double start_q, end_q;
  double loc_elapsed_q;
  double cpu_time_q;

  int my_rank, comm_sz;
  MPI_Init(NULL, NULL);
	MPI_Pcontrol(2);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  //ADJUST CORE COUNT TO THE NUMBER OF CORES OF YOUR CPU
  //int core_count = 8;
  //int n_threads = core_count / comm_sz;
  //omp_set_num_threads(n_threads);

  if (dim < comm_sz){
    printf("ERROR!! The dimension of the array must be bigger than comm_sz\n");
    MPI_Finalize();
    return 0;
  }

  srand(time(NULL) + my_rank);
	
  float* in = RandFloatGenerator(dim, -500.0, 500); 
  /*if(my_rank==0) printf("\nORIGINAL VECTORS\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(in, dim, my_rank, comm_sz, FLOAT);*/
  

  float* control = malloc(dim * sizeof(float));
  start_v = MPI_Wtime();
  PMPI_Allreduce(in, control, dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  end_v = MPI_Wtime();
  loc_elapsed_v = end_v - start_v;
  PMPI_Reduce(&loc_elapsed_v, &cpu_time_v, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if(my_rank==0) printf("\nCONTROL VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(control, dim, my_rank, comm_sz, FLOAT);
  
  float* out = malloc(dim * sizeof(float));
  MPI_Barrier(MPI_COMM_WORLD);
  start_q = MPI_Wtime();
	MPI_Allreduce(in, out, dim, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  end_q = MPI_Wtime();
  loc_elapsed_q = end_q - start_q;
  PMPI_Reduce(&loc_elapsed_q, &cpu_time_q, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if(my_rank==0) printf("\nALLRED VECTOR\n");
  MPI_Barrier(MPI_COMM_WORLD);
  ProcessPrinter(out, dim, my_rank, comm_sz, FLOAT);
  
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 0){
    float error = NormalizedMSE(out, control, dim);
    printf("error: %f\n", error);
    printf("quantized allreduce: %lf\n", cpu_time_q);
    printf("vanilla allreduce: %lf\n", cpu_time_v);
    printf("dim: %ld\n", dim);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  free(in);
  free(out);
  free(control);

	MPI_Finalize();

  return 0;
}

