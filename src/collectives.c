#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <time.h>

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
    clock_t start, end;
    double cpu_time;
    int algo;
    // TODO NEED TO SET ALGO VARIABLE

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    return 1;
}

/* Each active node will receive the entire vector of 
 * quantized data, dequantize it, sum it with its own vector, 
 * requantize the partial sum vector and send it to the next process.
 * For now assumes comm_sz is always divisible by 2. */
// TODO Free stuff
int RecursiveHalvingSend(int my_rank, int comm_sz, int dim, int algo, float * my_numbers) { 
    /*
    // first dequantize
    float * dequantized_1 = malloc(sizeof(float) * dim);
    float * dequantized_2 = malloc(sizeof(float) * dim);
    float * partial_sum = malloc(sizeof(float) * dim);
    int i;

    // dequantize two vectors
    DequantizeTwoVectors(struct_ptr1, struct_ptr2, dequantized_1, dequantized_2, algo, dim);

    // sum two dequantized vectors
    for (i = 0; i < dim; i++)
        partial_sum[i] = dequantized_1[i] + dequantized_2[i];
    
    // quantize partial sum vector
    void * quantized_partial_sum_struct = Quantize(partial_sum, dim, algo);
    uint8_t * quantized_partial_sum;
    // used to extrapolate quantized vector from sum
    FromStructToVec(algo, quantized_partial_sum_struct, quantized_partial_sum, dim);

    free(dequantized_1);
    free(dequantized_2);
    free(partial_sum); */

    int remaining = comm_sz;
    int half;
    // used to store quantized received bits
    void * struct_ptr;
    // used to store received quantized data
    void * rcv_buf;

    float * dequantized;
    uint8_t * quantized;
    float * partial_sum = my_numbers;
    // need to send struct as well
    while (remaining != 1) {
        half = remaining / 2;
        
        if (my_rank < half) {
            int source = half + my_rank;
            // receive struct and quantized array
            MPI_Recv(rcv_buf, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(struct_ptr, 1, MPI_STRUCT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // once received dequantize it and do sum
            // need to attach vector to struct

            DequantizeVector(struct_ptr, rcv_buf, dequantized, algo, dim);
            // probably need to free the vector allocated by the dequantizer

            // sum my array with received dequantized one
            for (int i = 0; i < dim; i++)
                partial_sum[i] = partial_sum[i] + dequantized[i];
        
        } else {
            int dest = my_rank % half;
            // first quantize
            struct_ptr = Quantize(partial_sum, dim , algo);
            // extract from struct quantized array to send seperately
            FromStructToVec(algo, struct_ptr, quantized, dim);

            MPI_Isend(quantized, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Isend(struct_ptr, 1, MPI_STRUCT, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        remaining = remaining / 2;
    }

    free(quantized);
    free(dequantized);
    free(struct_ptr);
    free(rcv_buf);
}

/* Function takes float vector and quantizes it according to
 * the kind of algorithm specified by int algo. Returns the 
 * struct containing the quantized vector */
void * Quantize(float * sendbuf, int count, int algo){
    
    void * struct_ptr;

    uint8_t *results = malloc(sizeof(uint8_t) * count);

    if (algo = 0)
    {
        struct_ptr = LloydMaxQuantizer(sendbuf, count);
    }
    else if (algo = 1)
    {
        char *string_type_env = getenv("NON_LINEAR_TYPE");
        int type;

        if (string_type_env != NULL)
            type = atoi(string_type_env);
        else
        {
            printf("\nERROR : Couldn't find a type env_var, aborting...\n\n");
            return;
        }

        struct_ptr = NonLinearQuantization(sendbuf, count, type);
    }

    else if (algo = 2)
        struct_ptr = UniformRangedQuantization(sendbuf, count);

    else 
        return;
    
    return struct_ptr;
}

/* Same as Dequantize vector but for 2 vectors */
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
}

/* Writes into dequantized_1 the dequantized array present into 
 * struct_ptr1.vec */

// TODO modify this so that it can dequantize by receiving the array of uints8 
// directly and only use the right struct fields
void DequantizeVector(void * struct_ptr1, float * dequantized_1, uint8_t * quantized, int algo, int dim){
    switch (algo)
    {
    case 0:
        struct lloyd_max_quant *str1 = (struct lloyd_max_quant *)struct_ptr1;
        dequantized_1 = LloydMaxDequantizer2(str1, dim, quantized);
    case 1:
        struct non_linear_quant *str1 = (struct non_linear_quant *)struct_ptr1;
        dequantized_1 = NonLinearDequantization2(str1, dim, quantized);
    case 2:
        struct unif_quant *str1 = (struct unif_quant *)struct_ptr1;
        dequantized_1 = UniformRangedDequantization2(str1, dim, quantized);
    }
}
/* Extracts from quantized_partial_sum_struct 
 * the quantized vector in the struct's vec field */
void FromStructToVec(int algo, void *quantized_partial_sum_struct, uint8_t *quantized_partial_sum, int dim)
{
    int i;
    int dim;

    switch (algo)
    {
    case 0:
        struct lloyd_max_quant *res = (struct lloyd_max_quant *)quantized_partial_sum_struct;
        for (i = 0; i < dim; i++)
            quantized_partial_sum[i] = res->vec[i].number;
    case 1:
        struct non_linear_quant *res = (struct non_linear_quant *)quantized_partial_sum_struct;
        for (i = 0; i < dim; i++)
            quantized_partial_sum[i] = res->vec[i].number;
    case 2:
        struct unif_quant *res = (struct unif_quant *)quantized_partial_sum_struct;
        for (i = 0; i < dim; i++)
            quantized_partial_sum[i] = res->vec[i].number;
    }
}
/* Makes struct_ptr vec field point to received quantized array */
void AttachStructToArray(int algo, uint8_t *quantized, void *struct_ptr)
{
    switch (algo)
    {
    case 0:
        struct lloyd_max_quant *res = (struct lloyd_max_quant *) struct_ptr;
    case 1:
        struct non_linear_quant *res = (struct non_linear_quant *) struct_ptr;
    case 2:
        struct unif_quant *res = (struct unif_quant *) struct_ptr;
    }
}