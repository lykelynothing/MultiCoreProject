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
 * UNIFORM = 2 */
int MPI_Allreduce(const void *sendbuf, void *recvbuf,
                  int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    int my_rank, comm_sz;
    //clock_t start, end;
    //double cpu_time;
    int algo;
    int send_algo;
    
    char * quant_env_var;
    char * send_algo_var;

    quant_env_var = getenv("QUANT_ALGO");
    send_algo_var = getenv("SEND_ALGO");

    if (send_algo_var == NULL){
        printf("\nError : no send_algo_var found. export SEND_ALGO = 1 (ring) | 0 (recursive halving)\n");
        return MPI_ERR_OTHER;
    }
    // default will be recursive-halving
    if (strcmp(send_algo_var, "RING") == 0){
        send_algo = 1;
    } else {
        send_algo = 0;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (quant_env_var == NULL){
        printf("\nError : no quant_env_var found. export QUANT_ALGO=LLOYD|NON_LINEAR(NON_LINEAR_TYPE=1|0)|UNFIORM\n");
        return MPI_ERR_OTHER;
    }

    if (strcmp(quant_env_var, "LLOYD") == 0) {
        // when we work with algo we also have to 
        // send the codebook
        algo = 0;
    } else if (strcmp(quant_env_var, "NON_LINEAR") == 0) {
        // also check for type of non_linear
        algo = 1;
    } else if (strcmp(quant_env_var, "UNIFORM") == 0) {
        algo = 2;
    } else if (quant_env_var == NULL){
        // call normal MPI_Allreduce
        return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }

    if (send_algo == 0){
        // need to add rcv buff that will store final sum
        RecursiveHalvingSend(my_rank, comm_sz, count, algo, (float *) sendbuf);
    }
    else if (send_algo == 1){
        RingAllreduce(my_rank, comm_sz, count, algo, (float *) sendbuf);
    }
    /* else other send_algo */
    
    return MPI_SUCCESS;
}

/* Each active node will receive the entire vector of 
 * quantized data, dequantize it, sum it with its own vector, 
 * requantize the partial sum vector and send it to the next process.
 * For now assumes comm_sz is always divisible by 2. */
// TODO: Free stuff
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
            printf("out[%i]=%lf\n", i, partial_sum[i]);
    }

    return MPI_SUCCESS;

    //MPI_Barrier(MPI_COMM_WORLD);
    //free(quantized);
    //free(dequantized);
    //free(struct_ptr);
}

int RingAllreduce(int my_rank, int comm_sz, int dim, int algo, float* my_numbers, float* output_ptr){
    //Homomorphic quantization of the data
    void* struct_ptr = Quantize(my_numbers, dim, algo);
    //The repo here gathers input lenghts given to each rank (dim) and check if they are equal
    //using a normal MPI_Allgather. Wont do that, it seems an errorhandling practice   not useful for us 
    
    //The array will be divided into  N equal-sized chunks. Each process will have a vector of chunk_sizes
    size_t size = dim / comm_sz;
    size_t remainder = dim % comm_sz;
    size_t* sizes = malloc(comm_sz * sizeof(size_t));
    for (int i = 0; i < comm_sz; i++)
        segment_sizes[i] = (i < remainder) ? size + 1 : size;
    
    size_t* segment_ends[comm_sz];
    segment_ends[0] = segment_sizes[0];
    for (int i = 1; i < comm_sz; i++)
        segment_ends[i] = segment_sizes[i] + segment_ends[i-1];

    //output buffer 
    //(the pointer to the pointer will be used for MPI reasons)
    //TODO: free it maybe, idk
    float* output = malloc(dim * sizeof(uint8_t));
    *output_ptr = output;
    
    //copy data from original data to output buffer
    memcpy((void*) output, (void*) struct_ptr->vec, dim * sizeof(uint8_t));
    
    //temporary buffer for incoming data
    uint8_t* buffer = (uint8_t*) malloc(segment_sizes[0]*sizeof(uint8_t));
    
    const size_t recv_from = (my_rank - 1 + comm_sz) % comm_sz;
    const size_t send_to = (my_rank + 1) % comm_sz;
    
    //TODO: send here the pointers to the struct + quantization metadata (not the case for LM) .
    //Need a recieve, but don't get if I can use the same struct ptr just used. Gonna ask Luca
    
    MPI_Status recv_status;
    MPI_Request recv_req;

    for (int i = 0; i < comm_sz; i++){
        int recv_chunk = (rank - i - 1 + comm_sz) % comm_sz;
        int send_chunk = (rank - i + comm_sz) % comm_sz;
        //segment to send and recv
        uint8_t* segment_send = &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);

        MPI_Irecv(buffer, segment_sizes[recv_chunk], MPI_UINT8_T, recv_from, 0, MPI_COMM_WORLD, &recv_req);
        MPI_Send(segment_send, segment_sizes[send_chunk], MPI_UINT8_T, send_to, 0, MPI_COMM_WORLD);
        
        uint8_t* segment_update =  &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);
        
        MPI_Wait(&recv_req, &recv_status);
        
        //TODO: this is for Homomorphic compression, (and still needs an overflow control) need to write also for non Homomorphic compression
        for(size_t i = 0; i < segment_sizes[recv_chunk]; i++)
            segment_update[i] += buffer[i];
    }
    
    for (size_t i = 0; i < size_t (comm_sz - 1); ++i){
        int recv_chunk = (rank - i - 1 + comm_sz) % comm_sz;
        int send_chunk = (rank - i + comm_sz) % comm_sz;
        //segment to send and recv
        uint8_t* segment_send = &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);
        uint8_t* segment_recv = &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);

        MPI_Sendrecv(segment_send, segment_sizes[send_chunk], MPI_UINT8_T, send_to, 0, / 
                    segment_recv, segment_sizes[recv_chunk], MPI_UINT8_T, recv_from, 0, / 
                    MPI_COMM_WORLD, &recv_status);
    }
    
    free(buffer);
    
    //TODO: Homomorphic Dequantization of the datas
}



/* Function takes float vector and quantizes it according to
a
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
        default:
            printf("ERROR!! Quant algo not valid\n");
            return NULL;
    }
    
    return struct_ptr;
}

/* Same as Dequantize vector but for 2 vectors */
/*
void DequantizeTwoVectors(void * struct_ptr1, void * struct_ptr2, float * dequantized_1, 
                            float * dequantized_2, int algo, int dim) {
    switch (algo)
    {
    case 0:
        struct lloyd_max_quant *str1 = (struct lloyd_max_quant *)struct_ptr1;
        struct lloyd_max_quant *str2 = (struct lloyd_max_quant *)struct_ptr2;
        dequantized_1 = LloydMaxDequantizer(str1, dim);
        dequantized_2 = LloydMaxDequantizer(str2, dim);
    case 1:
        struct non_linear_quant *str1 = (struct non_linear_quant *)struct_ptr1;
        struct non_linear_quant *str2 = (struct non_linear_quant *)struct_ptr2;
        dequantized_1 = NonLinearDequantization(str1, dim);
        dequantized_2 = NonLinearDequantization(str2, dim);
    case 2:
        struct unif_quant *str1 = (struct unif_quant *)struct_ptr1;
        struct unif_quant *str2 = (struct unif_quant *)struct_ptr2;
        dequantized_1 = UniformRangedDequantization(str1, dim);
        dequantized_2 = UniformRangedDequantization(str2, dim);
    }
}*/

/* Writes into dequantized_1 the quantized array present into 
 * struct_ptr1.vec */
void DequantizeVector(void * struct_ptr1, float ** dequantized_1, int algo, int dim){
    switch (algo){
        case 0:
            struct lloyd_max_quant *str1 = (struct lloyd_max_quant *)struct_ptr1;
            *dequantized_1 = LloydMaxDequantizer(str1, dim);
            break;
        case 1:
            struct non_linear_quant *str2 = (struct non_linear_quant *)struct_ptr1;
            *dequantized_1 = NonLinearDequantization(str2, dim);
            break;
        case 2:
            struct unif_quant *str3 = (struct unif_quant *)struct_ptr1;
            *dequantized_1 = UniformRangedDequantization(str3, dim);
            break;
        default:
            printf("ERROR!! Quant algo not valid\n");
            break;
    }
}

/* Calls mpi to receive struct and quantized vector. Handles
 * all kinds of struct and attaches to vec field the received
 * quantized vector (uint8) */
// THE FUNCTION DOESN'T FREE THE MEMORY IT ALLOCATES,
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
            printf("ERROR!! Quant algo not valid\n");
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
            printf("ERROR!! Quant algo not valid\n");
            break;
    }

    return MPI_SUCCESS;
}

