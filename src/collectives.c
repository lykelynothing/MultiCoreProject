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
 * requantize the partial sum vector and send it to the next process */
// TODO Free shit
int RecursiveHalvingSend(int my_rank, int comm_sz, int dim, int algo, void * struct_ptr1, void * struct_ptr2) { 

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
    free(partial_sum);

    int remaining = comm_sz;
    int half;
    uint8_t * rcv_buff = malloc(sizeof(u_int8_t) * dim);

    while (remaining != 1) {
        half = comm_sz / 2;
        
        if (my_rank < half) {
            // TODO put right source
            int source = half + my_rank;
            MPI_Recv((void *) rcv_buff, dim, MPI_UINT8_T, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            // TODO check if thiscan be non-blocking
            // TODO put right dest
            int dest = my_rank % half;
            MPI_Isend((void *)quantized_partial_sum, dim, MPI_UINT8_T, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        remaining = remaining / 2;

    }
}

/* Function takes float vector and quantizes it according to
 * the kind of algorithm specified by int algo */
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

void DequantizeTwoVectors(void * struct_ptr1, void * struct_ptr2, float * dequantized_1, float * dequantized_2, int algo, int dim){
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