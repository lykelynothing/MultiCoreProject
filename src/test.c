#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

struct unif_quant{
	float min;
	float max;
  uint8_t* vec;
};

struct lloyd_max_quant{
	float min;
	float max;
	float* codebook;
	uint8_t* vec;
};

MPI_Datatype LloydMaxQuantType(){
    MPI_Datatype MPI_Lloyd_max_quant;
    MPI_Datatype types[4] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_UINT8_T};
    int block_lengths[4] = {1, 1, 1, 8};
    MPI_Aint offsets[4];

	MPI_Get_address(&(((struct lloyd_max_quant *)0)-> min), &offsets[0]);
	MPI_Get_address(&(((struct lloyd_max_quant *)0)-> max), &offsets[1]);
	MPI_Get_address(&(((struct lloyd_max_quant *)0)-> codebook), &offsets[2]);
	MPI_Get_address(&(((struct lloyd_max_quant *)0)-> vec), &offsets[3]);

	for (int i = 0; i < 4; i++)
		offsets[i] -= offsets[0];

    MPI_Type_create_struct(4, block_lengths, offsets, types, &MPI_Lloyd_max_quant);
    MPI_Type_commit(&MPI_Lloyd_max_quant);

    return MPI_Lloyd_max_quant;
}

MPI_Datatype UnifQuantType(){
  MPI_Datatype MPI_Unif_quant;
  MPI_Datatype types[3] = {MPI_FLOAT, MPI_FLOAT, MPI_UINT8_T};
  int block_lengths[3] = {1, 1, 1};  
  MPI_Aint displacements[3];
  
  MPI_Get_address(&(((struct unif_quant*)0)->min), &displacements[0]);
  MPI_Get_address(&(((struct unif_quant*)0)->max), &displacements[1]);
  MPI_Get_address(&(((struct unif_quant*)0)->vec), &displacements[2]);

  for(int i = 0; i < 3; i++)
    displacements[i] -= displacements[0];

  MPI_Type_create_struct(3, block_lengths, displacements, types, &MPI_Unif_quant);
  MPI_Type_commit(&MPI_Unif_quant);

  return MPI_Unif_quant;
}


int main(){
  int dim = 11;
  
  int my_rank, comm_sz;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  

  if(my_rank==0){
    struct lloyd_max_quant * data = (struct lloyd_max_quant*) malloc(sizeof(struct lloyd_max_quant));
    data -> vec = (uint8_t*) malloc(dim*sizeof(uint8_t));
    data -> min = 0;
    data -> max = (float)(dim-1)*(dim-1);
    // for testing assume codebook only contains 1 float
    data -> codebook = malloc(sizeof(float));
    data -> codebook[0] = 1.0f;

    for(int i=0; i < dim; i++){
      data->vec[i]= i*i;
    }

    data->vec[0] = 7;

    //send struct (not sending array)
    MPI_Datatype MPI_Unif = LloydMaxQuantType();
    MPI_Send(data, 1, MPI_Unif, 1, 0, MPI_COMM_WORLD);
    MPI_Type_free(&MPI_Unif);
    
    //sending array
    MPI_Send(data->vec, dim, MPI_UINT8_T, 1, 0, MPI_COMM_WORLD);
    // send codebook
    MPI_Send(data->codebook, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

    free(data->vec);
    free(data);
  }
  else{
    struct lloyd_max_quant * recv = malloc(sizeof(struct lloyd_max_quant));
    
    MPI_Datatype MPI_Unif = LloydMaxQuantType();
    MPI_Recv(recv, 1, MPI_Unif, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    recv->vec = (uint8_t*) malloc(dim * sizeof(uint8_t));
    recv->codebook = malloc(sizeof(float));
    MPI_Recv(recv->vec, dim, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(recv->codebook, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    printf("recv->min = %f \t recv->max = %f\n", recv->min, recv->max);
    for(int i=0; i<dim; i++)
      printf("recv->vec[%d] = %d \n", i, recv->vec[i]);
    
    printf("what about the codebook? \n");
    printf("First elem of codebook: %lf \n", recv->codebook[0]);

    MPI_Type_free(&MPI_Unif);
    free(recv->vec);
    free(recv->codebook);
    free(recv);
  }

  MPI_Finalize();

  return 0;
}

