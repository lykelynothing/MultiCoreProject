#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>

#include "tools.h"
#include "uniform_quantizer.h"

void ScatterUniform(struct unif_quant* in, size_t size, MPI_Comm comm, uint8_t** output_ptr){
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  
  int remainder = size % comm_sz;
  int div = size / comm_sz;

  size_t* segment_sizes = (size_t*) malloc(comm_sz * sizeof(size_t));
  for (int i = 0; i < comm_sz; i++)
    segment_sizes[i] = (i < remainder) ? (size_t) div + 1 : (size_t) div;

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
  MPI_Request recv_req;
  MPI_Datatype MPI_Unif = UnifQuantType();

  for (int i = 0; i < comm_sz; i++){
    int recv_chunk = (rank - i - 1 + comm_sz) % comm_sz;
    int send_chunk = (rank - i + comm_sz) % comm_sz;
    uint8_t* segment_sent = &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);
    
    MPI_Irecv(buffer, segment_sizes[recv_chunk], MPI_UINT8_T, recv_from, 0, comm, &recv_req);
    MPI_Send(segment_sent, segment_sizes[send_chunk], MPI_UINT8_T, send_to, 0, comm);

    uint8_t *segment_update = &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);

    MPI_Wait(&recv_req, &recv_status);

    //dequantize

    //for(size_t i = 0; i < segment_sizes[recv_chunk]; i++){
    //  segment_update[i] += buffer[i];
    //}
  }

  /*for (size_t i = 0; i < size_t(size - 1); ++i) {
    int send_chunk = (rank - i + 1 + comm_sz) % comm_sz;
    int recv_chunk = (rank - i + comm_sz) % comm_sz;
    // Segment to send - at every iteration we send segment (r+1-i)
    float* segment_send = &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);

    // Segment to recv - at every iteration we receive segment (r-i)
    float* segment_recv = &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);
    MPI_Sendrecv(segment_send, segment_sizes[send_chunk], datatype, send_to, 0, segment_recv, segment_sizes[recv_chunk], datatype, recv_from, 0, MPI_COMM_WORLD, &recv_status);
  }*/

  MPI_Type_free(&MPI_Unif);
  free(buffer);
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

