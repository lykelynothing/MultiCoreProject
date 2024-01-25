#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>

#include "tools.h"
#include "uniform_quantizer.h"
#include "collectives.h"

void ScatterUniform(struct unif_quant* in, size_t size, MPI_Comm comm, uint8_t** output_ptr){
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  
  int remainder = size % comm_sz;
  int div = size / comm_sz;
  const size_t my_segment_sz = (rank < remainder) ? div + 1 : div;

  size_t* segment_sizes = (size_t*) malloc(comm_sz * sizeof(size_t));
  for (int i = 0; i < comm_sz; i++)
    segment_sizes[i] = (i < remainder) ? div + 1 : div;

  size_t* segment_ends = (size_t*) malloc(comm_sz * sizeof(size_t));
  segment_ends[0] = segment_sizes[0];
  for (int i = 1; i < comm_sz; i++)
    segment_ends[i] = segment_ends[i-1] + segment_sizes[i];
  
  uint8_t* output = malloc(size*sizeof(uint8_t));
  *output_ptr = output;

  memcpy(output, in->vec, size);

  uint8_t* buffer = malloc(segment_sizes[0]*sizeof(uint8_t));

  int send_to = (rank + 1) % comm_sz;
  int recv_from = (rank - 1 + comm_sz) % comm_sz;
  
  MPI_Status recv_status;
  MPI_Request recv_request;

  for (int i = 0; i < comm_sz; i++){
    int recv_chunk = (rank - i - 1 + comm_sz) % comm_sz;
    int send_chunk = (rank - i + comm_sz) % comm_sz;
    struct compressed * segment_sent = &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);

    MPI_Irecv(recv_buffer, segment_sizes[recv_chunk], MPI_Byte, recv_from, 0, comm, &recv_request);
    MPI_Send(segment_sent, segment_sizes[send_chunk], MPI_Byte, send_to, 0, comm);

    uint8_t *segment_update = &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);

    MPI_Wait(&recv_req, &recv_status);

    //dequantize

    for(size_t i = 0; i < segment_sizes[recv_chunk]; i++){
      segment_update[i] += buffer[i];
    }
  } 
}

/*int MPI_Allreduce_ring(const void *sendbuf, void *recvbuf, int count,
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
 
  float* tempbuf = malloc(count * sizeof(datatype));
  
  for (int i = 0; i < count; i++)
    recvbuf[i] = sendbuf[i];

  for (int step = 1; step < size; step++0){
    int send_to = (rank + step) % size;
    int recv_from = (rank - step + size) % size;
    
    MPI_Request send_request, recv_request;
    MPI_Isend(sendbuf, count, datatype, send_to, 0, comm, &send_request);
    MPI_Irecv(tempbuf, count, datatype, recv_from, 0, comm, &recv_request);

    MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

    //dequantize here

    for (int i = 0; i < count; i++)
      recv_buf[i] += temp_buf[i];

    //quantize again here 
  }

  free(tempbuf);
  
  return MPI_SUCCESS;
}
*/ 

