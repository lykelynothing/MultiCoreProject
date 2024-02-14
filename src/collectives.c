#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <time.h>

#include "collectives.h"
#include "tools.h"
#include "lloyd_max_quantizer.h"
#include "non_linear_quantizer.h"
#include "uniform_quantizer.h"
#include "homomorphic_quantizer.h"

int RingAllreduce();
int Send();
void* Receive();
void* Quantize();
void DequantizeVector();
int RecursiveHalvingSend();
int RecursiveHalvingSendHomomorphic();
int RingAllreduce();

// TODO: Write correct precise returns 
// TODO: Create custom datatypes only once

/* Custom MPI_Allreduce that will intercept any calls to it.
 * Will look for the environment variable "QUANT_ALGO" and choose
 * the quantization algorithm accordingly (if QUANT_ALGO = NON_LINEAR 
 * then also environment variable NON_LINEAR_TYPE will be checked).
 * Once the sendbuf is quantized, it executes a normal Allreduce collective 
 * through PMPI_Allreduce.
 * 
 * Algo encoding:
 * LLOYD = 0
 * NON_LINEAR = 1
 * UNIFORM = 2 
 * HOMOMORPHIC = 3
 *
 * Send algo encoding:
 * REC_HALVING = 0 
 * RING = 1 
 * default: NO_QUANTIZATION*/
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
  int env_var[2];
  GetEnvVariables(env_var);

  int my_rank, comm_sz;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_sz);

  //TODO: need to add rcv buff that will store final sum in RecursiveHalvingSend
  switch (env_var[1]){
    case 0:
      if (env_var[0] != 3)
        RecursiveHalvingSend(my_rank, comm_sz, count, env_var[0], (float *) sendbuf);
      else 
        RecursiveHalvingSendHomomorphic(my_rank, comm_sz, count, (float *) sendbuf, (float *) recvbuf);
      break;
    case 1:
      RingAllreduce(my_rank, comm_sz, (float*) sendbuf, count, (float*) recvbuf);
      break;
    default:
      PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
      break;
  }
  
  return MPI_SUCCESS;
}


/* Each active node will receive the entire vector of 
 * quantized data, dequantize it, sum it with its own vector, 
 * requantize the partial sum vector and send it to the next process.
 * For now assumes comm_sz is always divisible by 2. */
// TODO: Free stuff
// TODO: Scatter final results
int RecursiveHalvingSend(int my_rank, int comm_sz, int dim, int algo, float * my_numbers) { 
  int remaining = comm_sz;
  int half;
  // used to store quantized received bits
  void * struct_ptr;

  float * dequantized;
  float * partial_sum = my_numbers;
  // need to send struct as well
  while (remaining != 1) {
    half = remaining / 2;
    if (my_rank < half) {
      int source = half + my_rank;
      // receive struct and quantized array
      struct_ptr = Receive(algo, dim, source);
      DequantizeVector(struct_ptr, &dequantized, algo, dim);
      // probably need to free the vector allocated by the dequantizer
      // sum my array with received dequantized one
      for (int i = 0; i < dim; i++)
        partial_sum[i] = partial_sum[i] + dequantized[i];
    } else {
      int dest = my_rank % half;
      // first quantize
      struct_ptr = Quantize(partial_sum, dim , algo);
      Send(struct_ptr, algo, dim, dest);
      printf("???\n");
    }
    remaining = remaining / 2;
  }

  if (my_rank == 0){
    for (int i = 0; i < dim; i++)
      printf("out[%d]=%lf\n", i, partial_sum[i]);
  }
  
  return MPI_SUCCESS;
}

/* Like Normal Recursive halving but this time the quantization and
 * dequantization only happen once i.e. before and after all reductions 
 * respectively have been executed */
int RecursiveHalvingSendHomomorphic(int my_rank, int comm_sz, int count, float * sendbuf, float * recvbuf){
  int remaining = comm_sz;
  int half;
  struct unif_quant * struct_ptr = HomomorphicQuantization(sendbuf, count, MPI_COMM_WORLD);
  struct unif_quant * rcv_bf;
  float * tmp;

  while (remaining != 1) {
    half = remaining / 2;
    if (my_rank < half) {
      int source = half + my_rank;
      // receive struct 
      rcv_bf = (struct unif_quant *) Receive(2, count, source);
      for (int i = 0; i < count; i++)
        // TODO: optimize this
        struct_ptr->vec[i] = struct_ptr->vec[i] + rcv_bf->vec[i];
    } else {
      int dest = my_rank % half;
      // send struct
      Send(struct_ptr, 2, count, dest);
    }
    remaining = remaining / 2;
  }
  // Now rank 0 dequantizes and scatters
  // TODO: maybe scatter and then each one dequantizes
  
  if (my_rank == 0){
    // this doesn't affect recvbuff
    tmp = HomomorphicDequantization(struct_ptr->vec, struct_ptr->min, struct_ptr->max, comm_sz, count, 1);
    for (int i = 0; i < count; i++){
      recvbuf[i] = tmp[i];
    }
    free(tmp);
  }
  MPI_Bcast(recvbuf, count, MPI_FLOAT, 0, MPI_COMM_WORLD);

  free(struct_ptr -> vec);
  free(struct_ptr);

  return MPI_SUCCESS;
}


/* Imprementation of the ring allreduce algorithm
 * The algorithm is divided into two parts and each send and recieve is done in a fashon: 
 * -  in the first part, the vector of each process is divided into comm_sz parts and
 *    each one of tose parts is sent to a different process (like an MPI_SCATTER), but
 *    a reduction is done by each process while is recieving those parts of the vector 
 * -  in the second part the reduced part of the vectors are then gathered by all processes
 *    in what is effectively a ring MPI_Allgather
 * TODO: This implementation of the ring allreduce uses only an Homomorphic Quantization scheme, other quant algo needs to be added.*/
int RingAllreduce(int my_rank, int comm_sz, float* data, size_t dim, float* output_ptr) {
  struct unif_quant* quantized_data = HomomorphicQuantization(data, dim, MPI_COMM_WORLD);

  //The array will be divided into  N equal-sized chunks
  size_t size = dim / comm_sz;
  size_t remainder = dim % size;
  
  //Segment sizes and segment ends arrays are used to work as indexes for 
  //scattering and gathering the array.
  size_t* segment_sizes = malloc(comm_sz * sizeof(size_t));
  for (int i = 0; i < comm_sz; i++)
    segment_sizes[i] = (i < remainder) ? size + 1 : size;
  size_t* segment_ends = malloc(comm_sz * sizeof(size_t));
  segment_ends[0] = segment_sizes[0];
  for (int i = 1; i < comm_sz; i++)
    segment_ends[i] = segment_sizes[i] + segment_ends[i-1];

  //This will store the total output and will be used when sending parts of the array 
  //The quantized array gets first copied.
  uint8_t* output = malloc(dim * sizeof(uint8_t));
  for (int i = 0; i < dim; i++)
    output[i] = quantized_data->vec[i];

  // Temporary buffer for incoming data
  uint8_t* buffer = malloc(segment_sizes[0] * sizeof(uint8_t));

  const size_t recv_from = (my_rank - 1 + comm_sz) % comm_sz;
  const size_t send_to = (my_rank + 1) % comm_sz;
  
  //Status and request are needed due to the use of MPI_Irecv 
  MPI_Status recv_status;
  MPI_Request recv_req;

  MPI_Datatype datatype;
  if (BITS == 8)
    datatype = MPI_UINT8_T;
  else if (BITS == 16)
    datatype = MPI_UINT16_T;
  
  //Scattering of the array. In each iteration each process recieves from its "recv_from"
  //process a chunk of the scattered array. 
  for (int i = 0; i < comm_sz - 1; i++) {
    int recv_chunk = (my_rank - i - 1 + comm_sz) % comm_sz;
    int send_chunk = (my_rank - i + comm_sz) % comm_sz;
    uint8_t* segment_send = &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);

    //The Irecv is used because each process will encounter the recv part first and we don't want it to stop
    //here without sending.
    MPI_Irecv(buffer, segment_sizes[recv_chunk], datatype, recv_from, 0, MPI_COMM_WORLD, &recv_req);
    MPI_Send(segment_send, segment_sizes[send_chunk], datatype, send_to, 0, MPI_COMM_WORLD);

    uint8_t* segment_update = &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);

    // Wait for recv to complete before reduction
    MPI_Wait(&recv_req, &recv_status);

    for(size_t i = 0; i < segment_sizes[recv_chunk]; i++)
      segment_update[i] += buffer[i];
  }

  //Gathering of the reduced array. The structure of this gathering is exactly the same as the 
  //scattering just done but without immediate communications.
  for (size_t i = 0; i < comm_sz - 1; ++i) {
    int send_chunk = (my_rank - i + 1 + comm_sz) % comm_sz;
    int recv_chunk = (my_rank - i + comm_sz) % comm_sz;

    uint8_t* segment_send = &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);
    uint8_t* segment_recv = &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);
    
    MPI_Sendrecv(segment_send, segment_sizes[send_chunk], datatype, send_to, 0, segment_recv, segment_sizes[recv_chunk], datatype, recv_from, 0, MPI_COMM_WORLD, &recv_status);
  }
  
//TODO: qua sarebbe da cambiare il pezzo
  float* temp =  HomomorphicDequantization(output, quantized_data->min, quantized_data->max, comm_sz, dim, 1);
  for(int i=0; i<dim; i++){
    output_ptr[i] = temp[i];
  }
  //Free of temporary data
  free(buffer);
  free(segment_sizes);
  free(segment_ends);

  return MPI_SUCCESS;
}

/* Function takes float vector and quantizes it according to
 * the kind of algorithm specified by int algo. Returns the 
 * struct containing the quantized vector */
void * Quantize(float * sendbuf, int count, int algo){ 
  void * struct_ptr;
  
  switch(algo){
    case 0:
      struct_ptr = LloydMaxQuantizer(sendbuf, count);
      break;
    case 1:
      char *string_type_env = getenv("NON_LINEAR_TYPE");
      int type;
      if (string_type_env != NULL)
        type = atoi(string_type_env);
      else {
        printf("\nERROR : Couldn't find a type env_var, aborting...\n\n");
        return NULL;
      }
      struct_ptr = NonLinearQuantization(sendbuf, count, type);
      break;
    case 2:
      struct_ptr = UniformRangedQuantization(sendbuf, count);
      break;
    case 3:
      struct_ptr = HomomorphicQuantization(sendbuf, count, MPI_COMM_WORLD);
      break;
    default:
      printf("ERROR!! Quant algo not valid (quantize call)\n");
      return NULL;
  }
  
  return struct_ptr;
}


/* Writes into dequantized_1 the quantized array present into 
 * struct_ptr1.vec */
void DequantizeVector(void * struct_ptr, float ** dequantized, int algo, int dim){
  switch (algo){
    case 0:
      struct lloyd_max_quant *str0 = (struct lloyd_max_quant *)struct_ptr;
      *dequantized = LloydMaxDequantizer(str0, dim);
      break;
    case 1:
      struct non_linear_quant *str1 = (struct non_linear_quant *)struct_ptr;
      *dequantized = NonLinearDequantization(str1, dim);
      break;
    case 2:
      struct unif_quant *str2 = (struct unif_quant *)struct_ptr;
      *dequantized = UniformRangedDequantization(str2, dim);
      break;
    default:
      printf("ERROR!! Quant algo not valid (dequantize call)\n");
      break;
  }
}


/* Calls mpi to receive struct and quantized vector. Handles
 * all kinds of struct and attaches to vec field the received
 * quantized vector (uint8). Returns pointer to received struct */
// TODO: THE FUNCTION DOESN'T FREE THE MEMORY IT ALLOCATES,
// REMEMBER TO DO SO OUTSIDE OF IT
void * Receive(int algo, int dim, int source){
  void * void_ptr;
  MPI_Datatype * type_ptr;

  switch(algo){
  // lloyd also contains codebook
  case 0:
    struct lloyd_max_quant * str_ptr1 = malloc(sizeof(struct lloyd_max_quant));
    str_ptr1->vec = malloc(sizeof(uint8_t) * dim);
    str_ptr1->codebook = malloc(sizeof(float) * REPR_RANGE);
    void_ptr = str_ptr1;
    MPI_Datatype MPI_Lloyd = LloydMaxQuantType();
    type_ptr = &MPI_Lloyd;
    MPI_Recv(str_ptr1, 1, MPI_Lloyd, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(str_ptr1->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Type_free(type_ptr);
    break;
  case 1:
    // first create struct
    struct non_linear_quant * str_ptr2 = malloc(sizeof(struct non_linear_quant));
    str_ptr2->vec = malloc(sizeof(uint8_t) * dim);
    void_ptr = str_ptr2;
    // then create custom mpi datatype
    MPI_Datatype MPI_NonLin = NonLinearQuantType();
    type_ptr = &MPI_NonLin;
    MPI_Recv(str_ptr2, 1, MPI_NonLin, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(str_ptr2 -> vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Type_free(type_ptr);
    break; 
  case 2:
    struct unif_quant * str_ptr3 = malloc(sizeof(struct unif_quant));
    str_ptr3->vec = malloc(sizeof(uint8_t) * dim);
    void_ptr = str_ptr3;
    MPI_Datatype MPI_Unif = UnifQuantType();
    type_ptr = &MPI_Unif;
    MPI_Recv(str_ptr3, 1, MPI_Unif, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(str_ptr3->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Type_free(type_ptr);
    break;
  default:
    printf("ERROR!! Quant algo not valid (recieve call)\n");
    break;
  }

  return void_ptr;
}


/* Sends the struct and its vec field (and codebook with LLOYD) to dest.
 * Remeber to deallocate space outside of function. */
int Send(void * struct_ptr, int algo, int dim, int dest){
  MPI_Datatype * type_ptr;

  switch(algo){
    // lloyd also contains codebook
    case 0:
      struct lloyd_max_quant * str_ptr1 = (struct lloyd_max_quant *) struct_ptr;
      MPI_Datatype MPI_Lloyd = LloydMaxQuantType();
      type_ptr = &MPI_Lloyd;
      MPI_Send(str_ptr1, 1, MPI_Lloyd, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr1->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
      MPI_Type_free(type_ptr);
      break;
    case 1:
      // first create struct
      struct non_linear_quant * str_ptr2 = malloc(sizeof(struct non_linear_quant));
      // then create custom mpi datatype
      MPI_Datatype MPI_NonLin = NonLinearQuantType();
      type_ptr = &MPI_NonLin;
      MPI_Send(str_ptr2, 1, MPI_NonLin, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr2 -> vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      MPI_Type_free(type_ptr);
      break;
    case 2:
      struct unif_quant * str_ptr3 = (struct unif_quant *) struct_ptr;
      MPI_Datatype MPI_Unif = UnifQuantType();
      type_ptr = &MPI_Unif;
      MPI_Send(str_ptr3, 1, MPI_Unif, dest, 0, MPI_COMM_WORLD);
      MPI_Send(str_ptr3->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
      MPI_Type_free(type_ptr);
      break;
    default:
      printf("ERROR!! Quant algo not valid (send_call)\n");
      break;
  }
  
  return MPI_SUCCESS;
}
