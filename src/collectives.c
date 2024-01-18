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
#include "vector_quantizer.h"

/* Custom MPI_Allreduce that will intercept any calls to it.
 * Will look for the environment variable "QUANT_ALGO" and choose
 * the quantization algorithm accordingly. Once the sendbuf is quantized,
 * it executes a normal Allreduce collective throufh PMPI_Allreduce. */
// TODO: free all this shit
int MPI_Allreduce(const void * sendbuf, void * recvbuf, 
        int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    clock_t start, end;
    double cpu_time;

        char * env_var = getenv("QUANT_ALGO");
        printf("Env_var = %s \n", env_var);

        if (env_var == NULL)
                return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);

        struct compressed * vec_struct;
        uint8_t * results = malloc(sizeof(uint8_t) * count);

        if (strcmp(env_var, "LLOYD") == 0){
                struct lloyd_max_quant * local_res;
                local_res = LloydMaxQuantizer((float *) sendbuf, count);
                vec_struct = local_res -> vec;

        } else if (strcmp(env_var, "NON_LINEAR") == 0){
                char *string_type_env = getenv("NON_LINEAR_TYPE");
                int type;

                if (string_type_env != NULL)
                        type = atoi(string_type_env);
                else {
                        printf("\nERROR : Couldn't find a type env_var, aborting...\n\n");
                        return MPI_ERR_UNKNOWN;
                }

                struct non_linear_quant * local_res;
                local_res = NonLinearQuantization((float *) sendbuf, count, type);
                vec_struct = local_res -> vec;

        } else if (strcmp(env_var, "UNIFORM") == 0){
                struct unif_quant * local_res;
                local_res = UniformRangedQuantization((float *) sendbuf, count);
                vec_struct = local_res -> vec;
        
        } else if (strcmp(env_var, "NULL") == 0)            //i think this is not usefull due to previous if(env_var== NULL) but not sure
                start = clock();
        int res = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
        end = clock();
        cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("CPU time %lf \n", cpu_time);
        return res;
    
        for (int i = 0; i < count; i++)
                results[i] = vec_struct[i].number;

    start = clock();
        int res = PMPI_Allreduce((void *) results, (void *) recvbuf, count, MPI_UINT64_T, op, comm);
    end = clock();
    cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time %lf \n", cpu_time);
    return res;
}
