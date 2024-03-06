#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "homomorphic_quantizer.h"
#include "tools.h"
#include "tools_collectives.h"

/* Imprementation of the ring allreduce algorithm
 * The algorithm is divided into two parts and each send and recieve is done in
 * a fashon:
 * -  in the first part, the vector of each process is divided into comm_sz
 * parts and each one of tose parts is sent to a different process (like an
 * MPI_SCATTER), but a reduction is done by each process while is recieving
 * those parts of the vector
 * -  in the second part the reduced part of the vectors are then gathered by
 * all processes in what is effectively a ring MPI_Allgather*/
int RingAllreduce(int my_rank, int comm_sz, float *data, int dim,
                  float **output_ptr, QUANT algo) {
  if (algo == LLOYD || algo == NON_LINEAR) {
    if (my_rank == 0)
      printf("ERROR!! Non Linear Quantization and Lloyd Quantization NOT "
             "supported by RingAllreduction!!\nreturning . . .");
    return MPI_ERR_OTHER;
  }

  void *void_ptr = Allocate(algo, dim);
  void_ptr = Quantize(data, (int)dim, algo, void_ptr);
  struct unif_quant *quantized_data = (struct unif_quant *)void_ptr;

  // The array will be divided into  N equal-sized chunks
  size_t size = dim / comm_sz;
  size_t remainder = dim % size;

  // Segment sizes and segment ends arrays are used to work as indexes for
  // scattering and gathering the array.
  size_t *segment_sizes = malloc(comm_sz * sizeof(size_t));
  for (int i = 0; i < comm_sz; i++)
    segment_sizes[i] = (i < remainder) ? size + 1 : size;
  size_t *segment_ends = malloc(comm_sz * sizeof(size_t));
  segment_ends[0] = segment_sizes[0];
  for (int i = 1; i < comm_sz; i++)
    segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];

  // This will store the total output and will be used when sending parts of the
  // array The quantized array gets first copied.
  uint8_t *output = malloc(dim * sizeof(uint8_t));
  for (int i = 0; i < dim; i++)
    output[i] = quantized_data->vec[i];

  // Temporary buffer for incoming data
  uint8_t *buffer = malloc(segment_sizes[0] * sizeof(uint8_t));

  const size_t recv_from = (my_rank - 1 + comm_sz) % comm_sz;
  const size_t send_to = (my_rank + 1) % comm_sz;

  // Status and request are needed due to the use of MPI_Irecv
  MPI_Status recv_status;
  MPI_Request recv_req;

  MPI_Datatype datatype = MPI_UINT8_T;

  // Scattering of the array. In each iteration each process recieves from its
  // "recv_from" process a chunk of the scattered array.
  for (int i = 0; i < comm_sz - 1; i++) {
    int recv_chunk = (my_rank - i - 1 + comm_sz) % comm_sz;
    int send_chunk = (my_rank - i + comm_sz) % comm_sz;
    uint8_t *segment_send =
        &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);

    // The Irecv is used because each process will encounter the recv part first
    // and we don't want it to stop here without sending.
    MPI_Irecv(buffer, segment_sizes[recv_chunk], datatype, recv_from, 0,
              MPI_COMM_WORLD, &recv_req);
    MPI_Send(segment_send, segment_sizes[send_chunk], datatype, send_to, 0,
             MPI_COMM_WORLD);

    uint8_t *segment_update =
        &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);

    // Wait for recv to complete before reduction
    MPI_Wait(&recv_req, &recv_status);

    for (size_t i = 0; i < segment_sizes[recv_chunk]; i++)
      segment_update[i] += buffer[i];
  }

  // Gathering of the reduced array. The structure of this gathering is exactly
  // the same as the scattering just done but without immediate communications.
  for (size_t i = 0; i < comm_sz - 1; ++i) {
    int send_chunk = (my_rank - i + 1 + comm_sz) % comm_sz;
    int recv_chunk = (my_rank - i + comm_sz) % comm_sz;

    uint8_t *segment_send =
        &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);
    uint8_t *segment_recv =
        &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);

    MPI_Sendrecv(segment_send, segment_sizes[send_chunk], datatype, send_to, 0,
                 segment_recv, segment_sizes[recv_chunk], datatype, recv_from,
                 0, MPI_COMM_WORLD, &recv_status);
  }

  // TODO: change it with Dequantize
  *output_ptr = HomomorphicDequantization(output, quantized_data->min,
                                          quantized_data->max, comm_sz, dim, 1,
                                          *output_ptr);

  // Free of temporary data
  free(buffer);
  free(segment_sizes);
  free(segment_ends);
  free(quantized_data->vec);
  free(quantized_data);

  return MPI_SUCCESS;
}

int RingAllreduce_16(int my_rank, int comm_sz, float *data, size_t dim,
                     float **output_ptr, QUANT algo) {
  if (algo == LLOYD || algo == NON_LINEAR) {
    if (my_rank == 0)
      printf("ERROR!! Non Linear Quantization and Lloyd Quantization NOT "
             "supported by RingAllreduction!!\nreturning . . .");
    return MPI_ERR_OTHER;
  }

  void *void_ptr = Allocate(algo, dim);
  void_ptr = Quantize(data, (int)dim, algo, void_ptr);
  struct unif_quant_16 *quantized_data = (struct unif_quant_16 *)void_ptr;

  size_t size = dim / comm_sz;
  size_t remainder = dim % size;

  size_t *segment_sizes = malloc(comm_sz * sizeof(size_t));
  for (int i = 0; i < comm_sz; i++)
    segment_sizes[i] = (i < remainder) ? size + 1 : size;
  size_t *segment_ends = malloc(comm_sz * sizeof(size_t));
  segment_ends[0] = segment_sizes[0];
  for (int i = 1; i < comm_sz; i++)
    segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];

  uint16_t *output = malloc(dim * sizeof(uint16_t));
  for (int i = 0; i < dim; i++)
    output[i] = quantized_data->vec[i];
  uint16_t *buffer = malloc(segment_sizes[0] * sizeof(uint16_t));

  const size_t recv_from = (my_rank - 1 + comm_sz) % comm_sz;
  const size_t send_to = (my_rank + 1) % comm_sz;

  MPI_Status recv_status;
  MPI_Request recv_req;
  MPI_Datatype datatype = MPI_UINT16_T;

  for (int i = 0; i < comm_sz - 1; i++) {
    int recv_chunk = (my_rank - i - 1 + comm_sz) % comm_sz;
    int send_chunk = (my_rank - i + comm_sz) % comm_sz;

    uint16_t *segment_send =
        &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);

    MPI_Irecv(buffer, segment_sizes[recv_chunk], datatype, recv_from, 0,
              MPI_COMM_WORLD, &recv_req);
    MPI_Send(segment_send, segment_sizes[send_chunk], datatype, send_to, 0,
             MPI_COMM_WORLD);

    uint16_t *segment_update =
        &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);

    MPI_Wait(&recv_req, &recv_status);

    for (size_t i = 0; i < segment_sizes[recv_chunk]; i++)
      segment_update[i] += buffer[i];
  }
  for (size_t i = 0; i < comm_sz - 1; ++i) {
    int send_chunk = (my_rank - i + 1 + comm_sz) % comm_sz;
    int recv_chunk = (my_rank - i + comm_sz) % comm_sz;

    uint16_t *segment_send =
        &(output[segment_ends[send_chunk] - segment_sizes[send_chunk]]);
    uint16_t *segment_recv =
        &(output[segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);

    MPI_Sendrecv(segment_send, segment_sizes[send_chunk], datatype, send_to, 0,
                 segment_recv, segment_sizes[recv_chunk], datatype, recv_from,
                 0, MPI_COMM_WORLD, &recv_status);
  }

  // TODO: change it with Deallocate
  *output_ptr = HomomorphicDequantization_16(output, quantized_data->min,
                                             quantized_data->max, comm_sz, dim,
                                             1, *output_ptr);

  free(buffer);
  free(segment_sizes);
  free(segment_ends);
  free(quantized_data->vec);
  free(quantized_data);
  return MPI_SUCCESS;
}
