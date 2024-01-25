#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

struct unif_quant{
	float min;
	float max;
  uint8_t* vec;
};

MPI_Datatype UnifQuantType(size_t dim){
  MPI_Datatype MPI_Unif_quant;
  MPI_Datatype types[3] = {MPI_UINT8_T, MPI_FLOAT, MPI_FLOAT};
  int block_lengths[3] = {dim, 1, 1};  
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
  int dim = 10;
  
  int my_rank, comm_sz;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  

  if(my_rank==0){
    struct unif_quant* data = (struct unif_quant*) malloc(sizeof(struct unif_quant));
    data -> vec = (uint8_t*) malloc(dim*sizeof(uint8_t));
    data -> min = 0;
    data -> max = (float)(dim-1)*(dim-1);

    for(int i=0; i < dim; i++){
      data->vec[i]= i*i;
    }
    
    //send struct (not sending array)
    MPI_Datatype MPI_Unif = UnifQuantType(dim);
    MPI_Send(data, 1, MPI_Unif, 1, 0, MPI_COMM_WORLD);
    MPI_Type_free(&MPI_Unif);
    
    //sending array
    MPI_Send(data->vec, dim, MPI_UINT8_T, 1, 0, MPI_COMM_WORLD);

    free(data->vec);
    free(data);
  }
  else{
    struct unif_quant* recv = (struct unif_quant*) malloc(sizeof(struct unif_quant));
    
    MPI_Datatype MPI_Unif = UnifQuantType(dim);
    MPI_Recv(recv, 1, MPI_Unif, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    recv->vec = (uint8_t*) malloc(dim * sizeof(uint8_t));
    MPI_Recv(recv->vec, dim, MPI_UINT8_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    printf("recv->min = %f \t recv->max = %f\n", recv->min, recv->max);
    for(int i=0; i<dim; i++)
      printf("recv->vec[%d] = %d \n", i, recv->vec[i]);

    MPI_Type_free(&MPI_Unif);
    free(recv->vec);
    free(recv);
  }

  MPI_Finalize();

  return 0;
}

