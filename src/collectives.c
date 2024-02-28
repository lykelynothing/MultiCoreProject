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
int RingAllreduce_16();
void * Allocate();
void Free();

// TODO: Write correct precise returns 
// TODO: Create custom datatypes only once
// TODO: Segmentation problems with homomorphic halving
// TODO: Infinite loop with normal halving

/* Custom MPI_Allreduce that will intercept any calls to it.
 * Will look for the environment variable "QUANT_ALGO" and choose
 * the quantization algorithm accordingly (if QUANT_ALGO = NON_LINEAR 
 * then also environment variable NON_LINEAR_TYPE will be checked).
 * Once the sendbuf is quantized, it executes a normal Allreduce collective 
 * through PMPI_Allreduce.*/ 
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
  SEND send_algo;
  QUANT quant_algo;
  GetEnvVariables(&send_algo, &quant_algo);

  int my_rank, comm_sz;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_sz);

  switch (send_algo){
    case REC_HALVING:{
      if (quant_algo != HOMOMORPHIC)
        RecursiveHalvingSend(my_rank, comm_sz, count, quant_algo, (float *) sendbuf, (float *) recvbuf);
      else 
        RecursiveHalvingSendHomomorphic(my_rank, comm_sz, count, (float *) sendbuf, (float *) recvbuf);
      break;
    }
    case RING:{
      if (BITS==16)
        RingAllreduce_16(my_rank, comm_sz, (float*) sendbuf, count, (float*) recvbuf);
      else 
        RingAllreduce(my_rank, comm_sz, (float*) sendbuf, count, (float**)&(recvbuf));
      break;
    }
    default:{
      PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
      break;
    }
  }
  
  return MPI_SUCCESS;
}


/* Each active node will receive the entire vector of 
 * quantized data, dequantize it, sum it with its own vector, 
 * requantize the partial sum vector and send it to the next process.
 * For now assumes comm_sz is always divisible by 2. */
int RecursiveHalvingSend(int my_rank, int comm_sz, int dim, QUANT algo, float * my_numbers, float * recv_buf) { 
  int remaining = comm_sz;
  int half;
  // used to store quantized received bits
  void * struct_ptr = Allocate(algo, dim);
 
  float * dequantized = malloc(sizeof(float) * dim);
  float * partial_sum = my_numbers;
  int sent = 0;

  // need to send struct as well
  while (remaining != 1) {
    half = remaining / 2;
    if (my_rank < half) {
      int source = half + my_rank;
      // receive struct and quantized array

      Receive(algo, dim, source, struct_ptr);

      // all will now be contained inside struct_ptr
      DequantizeVector(struct_ptr, dequantized, algo, dim);
      // probably need to free the vector allocated by the dequantizer
      // sum my array with received dequantized one
      for (int i = 0; i < dim; i++)
        partial_sum[i] = partial_sum[i] + dequantized[i];
    } else if (sent == 0) {
      int dest = my_rank % half;
      // first quantize
      Quantize(partial_sum, dim , algo, struct_ptr);
      // now quantize uses same struct_ptr
      Send(struct_ptr, algo, dim, dest);
      sent = 1;
    }
    remaining = remaining / 2;
  }
  if (my_rank == 0){
    for (int i = 0; i < dim; i++)
      recv_buf[i] = my_numbers[i];
  }
  PMPI_Bcast(recv_buf, dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

  Free(algo, struct_ptr);
  free(dequantized);
  return MPI_SUCCESS;
}

/* Like Normal Recursive halving but this time the quantization and
 * dequantization only happen once i.e. before and after all reductions 
 * respectively have been executed */
int RecursiveHalvingSendHomomorphic(int my_rank, int comm_sz, int count, float * sendbuf, float * recvbuf){
  int remaining = comm_sz;
  int half;
  void * void_ptr = Allocate(HOMOMORPHIC, count);
  HomomorphicQuantization(sendbuf, count, MPI_COMM_WORLD, void_ptr);
  struct unif_quant * struct_ptr = (struct unif_quant *) void_ptr;
  struct unif_quant * rcv_bf;
  float * tmp;

  while (remaining != 1) {
    half = remaining / 2;
    if (my_rank < half) {
      int source = half + my_rank;
      // receive struct 
      rcv_bf = (struct unif_quant *) Receive(2, count, source, struct_ptr);
      for (int i = 0; i < count; i++){
        struct_ptr->vec[i] = struct_ptr->vec[i] + rcv_bf->vec[i];
      }
    } else {
      int dest = my_rank % half;
      // send struct
      Send(struct_ptr, 2, count, dest);
    }
    remaining = remaining / 2;
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  MPI_Bcast(struct_ptr->vec, count, MPI_UINT8_T, 0, MPI_COMM_WORLD);

  tmp = malloc(sizeof(float) * count);
  tmp = HomomorphicDequantization(struct_ptr->vec, struct_ptr->min, struct_ptr->max, comm_sz, count, 1, tmp);
  for (int i = 0; i < count; i++)
    recvbuf[i] = tmp[i];

  free(tmp);
  //free(struct_ptr -> vec);
  //free(struct_ptr);

  return MPI_SUCCESS;
}


/* Imprementation of the ring allreduce algorithm
 * The algorithm is divided into two parts and each send and recieve is done in a fashon: 
 * -  in the first part, the vector of each process is divided into comm_sz parts and
 *    each one of tose parts is sent to a different process (like an MPI_SCATTER), but
 *    a reduction is done by each process while is recieving those parts of the vector 
 * -  in the second part the reduced part of the vectors are then gathered by all processes
 *    in what is effectively a ring MPI_Allgather*/
int RingAllreduce(int my_rank, int comm_sz, float* data, size_t dim, float** output_ptr) {
  void * void_ptr = Allocate(HOMOMORPHIC, dim);

  HomomorphicQuantization(data, dim, MPI_COMM_WORLD, void_ptr);
  struct unif_quant* quantized_data = (struct unif_quant *) void_ptr;
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

  MPI_Datatype datatype = MPI_UINT8_T;
  
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
  
  *output_ptr = HomomorphicDequantization(output, quantized_data->min, quantized_data->max, comm_sz, dim, 1, *output_ptr);
  
  //Free of temporary data
  free(buffer);
  free(segment_sizes);
  free(segment_ends);
  free(quantized_data->vec);
  free(quantized_data);

  return MPI_SUCCESS;
}

int RingAllreduce_16(int my_rank, int comm_sz, float* data, size_t dim, float* output_ptr) {
  void * void_ptr = Allocate(HOMOMORPHIC, dim);

  HomomorphicQuantization_16(data, dim, MPI_COMM_WORLD, void_ptr);
  struct unif_quant_16* quantized_data = (struct unif_quant_16 *) void_ptr;
  size_t size = dim / comm_sz;
  size_t remainder = dim % size;
  size_t* segment_sizes = malloc(comm_sz * sizeof(size_t));
  for (int i = 0; i < comm_sz; i++)
    segment_sizes[i] = (i < remainder) ? size + 1 : size;
  size_t* segment_ends = malloc(comm_sz * sizeof(size_t));
  segment_ends[0] = segment_sizes[0];
  for (int i = 1; i < comm_sz; i++)
    segment_ends[i] = segment_sizes[i] + segment_ends[i-1];
  uint16_t* output = malloc(dim * sizeof(uint16_t));
  for (int i = 0; i < dim; i++)
    output[i] = quantized_data->vec[i];
  uint16_t* buffer = malloc(segment_sizes[0] * sizeof(uint16_t));
  const size_t recv_from = (my_rank - 1 + comm_sz) % comm_sz;
  const size_t send_to = (my_rank + 1) % comm_sz;
  MPI_Status recv_status;
  MPI_Request recv_req;
  MPI_Datatype datatype = MPI_UINT16_T;
  for (int i = 0; i < comm_sz - 1; i++) {
    int recv_chunk = (my_rank - i - 1 + comm_sz) % comm_sz;
    int send_chunk = (my_rank - i + comm_sz) % comm_sz;
    uint16_t* segment_send = &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);
    MPI_Irecv(buffer, segment_sizes[recv_chunk], datatype, recv_from, 0, MPI_COMM_WORLD, &recv_req);
    MPI_Send(segment_send, segment_sizes[send_chunk], datatype, send_to, 0, MPI_COMM_WORLD);
    uint16_t* segment_update = &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);
    MPI_Wait(&recv_req, &recv_status);
    for(size_t i = 0; i < segment_sizes[recv_chunk]; i++)
      segment_update[i] += buffer[i];
  }
  for (size_t i = 0; i < comm_sz - 1; ++i) {
    int send_chunk = (my_rank - i + 1 + comm_sz) % comm_sz;
    int recv_chunk = (my_rank - i + comm_sz) % comm_sz;
    uint16_t* segment_send = &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);
    uint16_t* segment_recv = &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);
    MPI_Sendrecv(segment_send, segment_sizes[send_chunk], datatype, send_to, 0, segment_recv, segment_sizes[recv_chunk], datatype, recv_from, 0, MPI_COMM_WORLD, &recv_status);
  }
  float * temp = malloc(sizeof(float) * dim);
  temp = HomomorphicDequantization_16(output, quantized_data->min, quantized_data->max, comm_sz, dim, 1, temp);
  for(int i=0; i<dim; i++){
    output_ptr[i] = temp[i];
  }
  free(temp);
  free(buffer);
  free(segment_sizes);
  free(segment_ends);
  free(quantized_data->vec);
  free(quantized_data);
  return MPI_SUCCESS;
}


/* Function takes float vector and quantizes it according to
 * the kind of algorithm specified by int algo. Struct in arguments
 * will contain the quantized array */
void * Quantize(float * sendbuf, int count, QUANT algo, void * struct_ptr){ 

  if (BITS==8){
    switch(algo){
      case LLOYD:{
        LloydMaxQuantizer(sendbuf, count, struct_ptr);
        break;
        }
      case NON_LINEAR:{
        char *string_type_env = getenv("NON_LINEAR_TYPE");
        int type;
        if (string_type_env != NULL)
          type = atoi(string_type_env);
        else {
          printf("\nERROR : Couldn't find a type env_var, aborting...\n\n");
          return NULL;
        }
        NonLinearQuantization(sendbuf, count, type, struct_ptr);
        break;
        }
      case UNIFORM:{
        UniformRangedQuantization(sendbuf, count, struct_ptr);
        break;
        }
      case HOMOMORPHIC:{
        HomomorphicQuantization(sendbuf, count, MPI_COMM_WORLD, struct_ptr);
        break;
        }
      default:{
        printf("ERROR!! Quant algo not valid (quantize call)\n");
        return NULL;
      }
    }
  } else if (BITS==16) {
    switch(algo){
      case LLOYD:{
        LloydMaxQuantizer_16(sendbuf, count, struct_ptr);
        break;
      }
      case NON_LINEAR:{
        char *string_type_env = getenv("NON_LINEAR_TYPE");
        int type;
        if (string_type_env != NULL)
          type = atoi(string_type_env);
        else {
          printf("\nERROR : Couldn't find a type env_var, aborting...\n\n");
          return NULL;
        }
        NonLinearQuantization_16(sendbuf, count, type, struct_ptr);
        break;
      }
      case UNIFORM:{
        UniformRangedQuantization_16(sendbuf, count, struct_ptr);
        break;
      }
      case HOMOMORPHIC:{
        HomomorphicQuantization_16(sendbuf, count, MPI_COMM_WORLD, struct_ptr);
        break;
      }
      default:{
        printf("ERROR!! Quant algo not valid (quantize call)\n");
        return NULL;
      }
    }
  }
  
  return struct_ptr;
}


/* Writes into dequantized_1 the quantized array present into 
 * struct_ptr1.vec */
void DequantizeVector(void * struct_ptr, float * dequantized, QUANT algo, int dim){
  if (BITS==8){
    switch (algo){
      case LLOYD:{
        struct lloyd_max_quant *str0 = (struct lloyd_max_quant *)struct_ptr;
        LloydMaxDequantizer(str0, dim, dequantized);
        break;
      }
      case NON_LINEAR:{
        struct non_linear_quant *str1 = (struct non_linear_quant *)struct_ptr;
        NonLinearDequantization(str1, dim, dequantized);
        break;
      }
      case UNIFORM:{
        struct unif_quant *str2 = (struct unif_quant *)struct_ptr;
        UniformRangedDequantization(str2, dim, dequantized);
        break;
      }
      default:{
        printf("ERROR!! Quant algo not valid (dequantize call)\n");
        break;
      }
    }
  }else if(BITS==16){
    switch (algo){
      case LLOYD:{
        struct lloyd_max_quant_16 *str0 = (struct lloyd_max_quant_16 *)struct_ptr;
        LloydMaxDequantizer_16(str0, dim, dequantized);
        break;
      }
      case NON_LINEAR:{
        struct non_linear_quant_16 *str1 = (struct non_linear_quant_16 *)struct_ptr;
        NonLinearDequantization_16(str1, dim, dequantized);
        break;
      }
      case UNIFORM:{
        struct unif_quant_16 *str2 = (struct unif_quant_16 *)struct_ptr;
        UniformRangedDequantization_16(str2, dim, dequantized);
        break;
      }
      default:{
        printf("ERROR!! Quant algo not valid (dequantize call)\n");
        break;
      }
    }
  }
}


/* Calls mpi to receive struct and quantized vector. Handles
 * all kinds of struct and attaches to vec field the received
 * quantized vector (uint8). Returns pointer to received struct */
// Receives an already allocated struct
void* Receive(QUANT algo, int dim, int source, void * void_ptr){
  if (BITS==8){
    switch(algo){
    // lloyd also contains codebook
      case LLOYD:{
        struct lloyd_max_quant * str_ptr1 = (struct lloyd_max_quant *) void_ptr;
        MPI_Recv(&str_ptr1->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&str_ptr1->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(str_ptr1->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      }
      case NON_LINEAR:{
        struct non_linear_quant * str_ptr2 = (struct non_linear_quant*) void_ptr;
        MPI_Recv(&str_ptr2 -> min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&str_ptr2 -> max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(str_ptr2 -> vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      }
      case UNIFORM:{
        struct unif_quant * str_ptr3 = (struct unif_quant *) void_ptr;
        MPI_Recv(&str_ptr3->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&str_ptr3->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(str_ptr3->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      }
      case HOMOMORPHIC:{
        struct unif_quant * str_ptr4 = (struct unif_quant *) void_ptr;
        MPI_Recv(&str_ptr4->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&str_ptr4->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(str_ptr4->vec, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      }
      default:{
        printf("ERROR!! Quant algo not valid\n");
        break;
      }
    }
  }else if (BITS==16){
    switch(algo){
      case LLOYD:{
        struct lloyd_max_quant_16 * str_ptr1 = (struct lloyd_max_quant_16 *) void_ptr;
        MPI_Recv(&str_ptr1->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&str_ptr1->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(str_ptr1->vec, dim, MPI_UINT16_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      }
      case NON_LINEAR:{
        struct non_linear_quant_16 * str_ptr2 = (struct non_linear_quant_16*) void_ptr;
        MPI_Recv(&str_ptr2 -> min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&str_ptr2 -> max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(str_ptr2 -> vec, dim, MPI_UINT16_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      }
      case UNIFORM:{
        struct unif_quant_16 * str_ptr3 = (struct unif_quant_16 *) void_ptr;
        MPI_Recv(&str_ptr3->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&str_ptr3->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(str_ptr3->vec, dim, MPI_UINT16_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      }
      case HOMOMORPHIC:{
        struct unif_quant_16 * str_ptr4 = (struct unif_quant_16 *) void_ptr;
        MPI_Recv(&str_ptr4->min, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&str_ptr4->max, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(str_ptr4->vec, dim, MPI_UINT16_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        break;
      }
      default:{
        printf("ERROR!! Quant algo not valid\n");
        break;
      }
    }
  }

  return void_ptr;
}


/* Sends the struct and its vec field (and codebook with LLOYD) to dest.
 * Remeber to deallocate space outside of function. */
int Send(void * struct_ptr, QUANT algo, int dim, int dest){
  if (BITS==8){
    switch(algo){
      case LLOYD:{
        struct lloyd_max_quant * str_ptr1 = (struct lloyd_max_quant *) struct_ptr;
        MPI_Send(&str_ptr1->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&str_ptr1->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(str_ptr1->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
        MPI_Send(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        break;
      }
      case NON_LINEAR:{
        struct non_linear_quant * str_ptr2 = (struct non_linear_quant*) struct_ptr;
        MPI_Send(&str_ptr2 -> min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&str_ptr2 -> max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(str_ptr2 -> vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&str_ptr2 -> type, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        break;
      }
      case UNIFORM:{
        struct unif_quant * str_ptr3 = (struct unif_quant *) struct_ptr;
        MPI_Send(&str_ptr3->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&str_ptr3->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(str_ptr3->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
        break;
      }
      case HOMOMORPHIC:{
        struct unif_quant * str_ptr4 = (struct unif_quant *) struct_ptr;
        MPI_Send(&str_ptr4->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&str_ptr4->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(str_ptr4->vec, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD);
        break;
      }
      default:{
        printf("ERROR!! Quant algo not valid (send_call)\n");
        break;
      }
    }
  } else if (BITS==16){
    switch(algo){
      case LLOYD:{
        struct lloyd_max_quant_16 * str_ptr1 = (struct lloyd_max_quant_16 *) struct_ptr;
        MPI_Send(&str_ptr1->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&str_ptr1->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(str_ptr1->vec, dim, MPI_UINT16_T, dest, 0, MPI_COMM_WORLD);
        MPI_Send(str_ptr1->codebook, REPR_RANGE, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        break;
      }
      case NON_LINEAR:{
        struct non_linear_quant_16 * str_ptr2 = (struct non_linear_quant_16 *) struct_ptr;
        MPI_Send(&str_ptr2 -> min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&str_ptr2 -> max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(str_ptr2 -> vec, dim, MPI_UINT16_T, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&str_ptr2 -> type, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        break;
      }
      case UNIFORM:{
        struct unif_quant_16 * str_ptr3 = (struct unif_quant_16 *) struct_ptr;
        MPI_Send(&str_ptr3->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&str_ptr3->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(str_ptr3->vec, dim, MPI_UINT16_T, dest, 0, MPI_COMM_WORLD);
        break;
      }
      case HOMOMORPHIC:{
        struct unif_quant_16 * str_ptr4 = (struct unif_quant_16 *) struct_ptr;
        MPI_Send(&str_ptr4->min, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(&str_ptr4->max, 1, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Send(str_ptr4->vec, dim, MPI_UINT16_T, dest, 0, MPI_COMM_WORLD);
        break;
      }
      default:{
        printf("ERROR!! Quant algo not valid (send_call)\n");
        break;
      }
    }
  }
  return MPI_SUCCESS;
}


void * Allocate(QUANT algo, int count){
  void * void_ptr;
  if (BITS==8){
    switch (algo){
      case LLOYD:{
        void_ptr = malloc(sizeof(struct lloyd_max_quant));
        struct lloyd_max_quant * tmp_ptr1 = (struct lloyd_max_quant *) void_ptr;
        tmp_ptr1 -> vec = malloc(sizeof(uint8_t) * count);
        break;
      }
      case NON_LINEAR:{
        void_ptr = malloc(sizeof(struct non_linear_quant));
        struct non_linear_quant * tmp_ptr2 = (struct non_linear_quant *) void_ptr;
        tmp_ptr2 -> vec = malloc(sizeof(uint8_t) * count);
        break;
      }
      case UNIFORM:{
        void_ptr = malloc(sizeof(struct unif_quant));
        struct unif_quant * tmp_ptr3 = (struct unif_quant *) void_ptr;
        tmp_ptr3 -> vec = malloc(sizeof(uint8_t) * count);
        break;
      }
      case HOMOMORPHIC:{
        void_ptr = malloc(sizeof(struct unif_quant));
        struct unif_quant * tmp_ptr4 = (struct unif_quant *) void_ptr;
        tmp_ptr4 -> vec = malloc(sizeof(uint8_t) * count);
        printf("ALLOC time out[0]: %d \n", tmp_ptr4->vec[0]);
        printf("ALLOC time pointer : %p \n", tmp_ptr4);
        break;
      }
      default:{
        break;
      }
    }
  } else if (BITS == 16){
    switch (algo){
      case LLOYD:{
        void_ptr = malloc(sizeof(struct lloyd_max_quant_16));
        struct lloyd_max_quant_16 * tmp_ptr1 = (struct lloyd_max_quant_16 *) void_ptr;
        tmp_ptr1 -> vec = malloc(sizeof(uint16_t) * count);
        break;
      }
      case NON_LINEAR:{
        void_ptr = malloc(sizeof(struct non_linear_quant_16));
        struct non_linear_quant_16 * tmp_ptr2 = (struct non_linear_quant_16 *) void_ptr;
        tmp_ptr2 -> vec = malloc(sizeof(uint16_t) * count);
        break;
      }
      case UNIFORM:{
        void_ptr = malloc(sizeof(struct unif_quant_16));
        struct unif_quant_16 * tmp_ptr3 = (struct unif_quant_16 *) void_ptr;
        tmp_ptr3 -> vec = malloc(sizeof(uint16_t) * count);
        break;
      }
      case HOMOMORPHIC:{
        void_ptr = malloc(sizeof(struct unif_quant_16));
        struct unif_quant_16* tmp_ptr4 = (struct unif_quant_16 *) void_ptr;
        tmp_ptr4 -> vec = malloc(sizeof(uint16_t) * count);
        break;
      }
      default:{
        break;
      }
    }
  }
  return void_ptr;
}


void Free(QUANT algo, void * void_ptr){
  if(BITS==8){
    switch (algo){
      case LLOYD:{
        struct lloyd_max_quant * tmp_ptr1 = (struct lloyd_max_quant *) void_ptr;
        free(tmp_ptr1->vec);
        free(tmp_ptr1);
        break;
      }
      case NON_LINEAR:{
        struct non_linear_quant * tmp_ptr2 = (struct non_linear_quant *) void_ptr;
        free(tmp_ptr2->vec);
        free(tmp_ptr2);
        break;
      }
      case UNIFORM:{
        struct unif_quant * tmp_ptr3 = (struct unif_quant *) void_ptr;
        free(tmp_ptr3->vec);
        free(tmp_ptr3);
        break;
      }
      case HOMOMORPHIC:{
        struct unif_quant * tmp_ptr4 = (struct unif_quant *) void_ptr;
        free(tmp_ptr4->vec);
        free(tmp_ptr4);
        break;
      }
      default:{
        break;
      }
    }
  }else if (BITS==16){
    switch (algo){
      case LLOYD:{
        struct lloyd_max_quant_16 * tmp_ptr1 = (struct lloyd_max_quant_16 *) void_ptr;
        free(tmp_ptr1->vec);
        free(tmp_ptr1);
        break;
      }
      case NON_LINEAR:{
        struct non_linear_quant_16 * tmp_ptr2 = (struct non_linear_quant_16 *) void_ptr;
        free(tmp_ptr2->vec);
        free(tmp_ptr2);
        break;
      }
      case UNIFORM:{
        struct unif_quant_16 * tmp_ptr3 = (struct unif_quant_16 *) void_ptr;
        free(tmp_ptr3->vec);
        free(tmp_ptr3);
        break;
      }
      case HOMOMORPHIC:{
        struct unif_quant_16 * tmp_ptr4 = (struct unif_quant_16 *) void_ptr;
        free(tmp_ptr4->vec);
        free(tmp_ptr4);
        break;
      }
      default:{
        break;
      }
    }
  }
}

