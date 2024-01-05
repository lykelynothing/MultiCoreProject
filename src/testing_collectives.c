#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include "uniform_quantizer.c"
#include "tools.h"

int main(int argc, char ** argv){
	
	int my_rank, comm_sz;

	MPI_Init(NULL, NULL);
	
	MPI_Pcontrol(2);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	float * my_vec = RandFloatGenerator(2, 0.0, 100.0);
	float *res = malloc(2 * sizeof(float));

	if (my_rank != 0){
		MPI_Allreduce(my_vec, NULL, 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	} else {
		MPI_Allreduce(my_vec, res, 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	}
	
	MPI_Finalize();
}

void MPI_Allreduce(const void * sendbuf, void * recvbuf, int count,
		MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
	
	struct uint8_vec * out = malloc(count * sizeof(datatype));
	clock_t start, end;
	double time_elapsed;

	out = UniformRangedQuantization(sendbuf, count);

	start = clock();

	PMPI_Allreduce(out, recvbuf, count, datatype, op, comm);

	end = clock();

	time_elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("Total time elapsed was %lf \n", time_elapsed);
}
