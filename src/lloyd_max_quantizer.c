#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "tools.h"

#define ITERATIONS 5

//				ASSIGNMENT STEP
/* The function takes as input a float value and return as output a	*
 * uint64_t value corresponding to the index of the closest (abs value	*
 * distance).								*/
uint8_t NearestCodeword(float element, float *codebook) {
  uint64_t nearest_cluster = REPR_RANGE + 1;
  float min_dist = INFINITY;

  for (int i = 0; i < REPR_RANGE; i++) {
    float dist = fabsf(codebook[i] - element);
    if (dist < min_dist) {
      min_dist = dist;
      nearest_cluster = i;
    }
  }

  if (nearest_cluster != REPR_RANGE + 1)
    return (uint8_t)nearest_cluster;
  else {
    printf("ERROR! Nearest cluster = NULL\n");
    return 0;
  }
}

// Same as before but for UINT16
uint16_t NearestCodeword_16(float element, float *codebook) {
  uint64_t nearest_cluster = REPR_RANGE + 1;
  float min_dist = INFINITY;
  for (int i = 0; i < REPR_RANGE; i++) {
    float dist = fabsf(codebook[i] - element);
    if (dist < min_dist) {
      min_dist = dist;
      nearest_cluster = i;
    }
  }
  if (nearest_cluster != REPR_RANGE + 1)
    return (uint16_t)nearest_cluster;
  else {
    printf("ERROR! Nearest cluster = NULL\n");
    return 0;
  }
}

//				UPDATE STEP
/* The function takes as input the original input vector, the output	*
 * (assignment) vector, the codebook, the input lenght and the codebook	*
 * lenght.								*
 * It reset the codebook to 0s to then updates it with the mean of the	*
 * assigned vectors.							*/
void UpdateCodebook(float *input, uint8_t *assignments, float *codebook,
                    size_t input_size) {
  // creates a counts vector that will count how many vector there are for each
  // assignment
  int counts[REPR_RANGE];
#pragma omp parallel
  {
// set the counts to 0 and reset the codebook
#pragma omp for
    for (int i = 0; i < REPR_RANGE; i++) {
      codebook[i] = 0;
      counts[i] = 0;
    }

    // each component of the input vector is added to the corresponding codebook
    //(and the corresponding count will be updated)
#pragma omp for
    for (int i = 0; i < input_size; i++) {
      int cluster = (int)assignments[i];
#pragma omp atomic
      counts[cluster]++;
#pragma omp atomic
      codebook[cluster] += input[i];
    }

    // for all the codebook value (with count > 0) the mean is taken
#pragma omp for
    for (int i = 0; i < REPR_RANGE; i++) {
      if (counts[i] != 0)
        codebook[i] /= counts[i];
    }
  }
}

// Same as before but for UINT16
void UpdateCodebook_16(float *input, uint16_t *assignments, float *codebook,
                       size_t input_size) {
  int counts[REPR_RANGE];
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < REPR_RANGE; i++) {
      codebook[i] = 0;
      counts[i] = 0;
    }
#pragma omp for
    for (int i = 0; i < input_size; i++) {
      int cluster = (int)assignments[i];
#pragma omp atomic
      counts[cluster]++;
#pragma omp atomic
      codebook[cluster] += input[i];
    }
#pragma omp for
    for (int i = 0; i < REPR_RANGE; i++) {
      if (counts[i] != 0)
        codebook[i] /= counts[i];
    }
  }
}

/* LLoyd-Max method (also called k-means method) is an iterative method for   /
 * quantization that aims to find a better suited quantization interval given /
 * a dataset.                                                                 /
 * (INITIALIZATION) A random codebook (containing random values that will be  /
 * the thresholds) is generated. Those values are also called centroids.      /
 * (ASSIGNMENT STEP) Each datapoint is assigned to a centroid.                /
 * (UPDATE STEP) Each centroid gets updated with the mean of its assigned     /
 * datapoint.                                                                 /
 * Repeat those two many times untill you have a low enough error function or /
 * you've done a predefined number of iterations.                             /
 * Once you've done that return the codebook of centroids and the quantized   /
 * vector.                                                                    */
struct lloyd_max_quant *LloydMaxQuantizer(float *in, size_t input_size,
                                          void *struct_ptr) {
  // define and allocate memory for struct and vector inside the struct
  struct lloyd_max_quant *out = (struct lloyd_max_quant *)struct_ptr;
  MinMax(in, input_size, &(out->min), &(out->max), 0);

  // Initialization
  out->codebook = RandFloatGenerator(REPR_RANGE, out->min, out->max);

  for (int i = 0; i < ITERATIONS; i++) {
#pragma omp parallel for
    for (int j = 0; j < input_size; j++)
      // Assignment step
      out->vec[j] = NearestCodeword(in[j], out->codebook);
    // Update step
    UpdateCodebook(in, out->vec, out->codebook, input_size);
  }

  return out;
}

// Same as before but for UINT16
struct lloyd_max_quant_16 *LloydMaxQuantizer_16(float *in, size_t input_size,
                                                void *struct_ptr) {
  struct lloyd_max_quant_16 *out = (struct lloyd_max_quant_16 *)struct_ptr;
  MinMax(in, input_size, &(out->min), &(out->max), 0);

  out->codebook = RandFloatGenerator(REPR_RANGE, out->min, out->max);

  for (int i = 0; i < ITERATIONS; i++) {
#pragma omp parallel for
    for (int j = 0; j < input_size; j++)
      out->vec[j] = NearestCodeword_16(in[j], out->codebook);
    UpdateCodebook_16(in, out->vec, out->codebook, input_size);
  }
  return out;
}

/* The quantized vector is confronted with its codebook and the restored values
 * are the corresponding codebook values*/
float *LloydMaxDequantizer(struct lloyd_max_quant *in, size_t input_size,
                           float *out) {
#pragma omp parallel for
  for (int i = 0; i < input_size; i++) {
    int cluster_index = (int)in->vec[i];

    if (cluster_index >= 0 && cluster_index < REPR_RANGE)
      out[i] = in->codebook[cluster_index];
    else
      printf("ERROR: invalid cluster index %d at position %d\n", cluster_index,
             i);
  }

  return out;
}

// Same as before but for UINT16
float *LloydMaxDequantizer_16(struct lloyd_max_quant_16 *in, size_t input_size,
                              float *out) {
#pragma omp parallel for
  for (int i = 0; i < input_size; i++) {
    int cluster_index = (int)in->vec[i];
    if (cluster_index >= 0 && cluster_index < REPR_RANGE)
      out[i] = in->codebook[cluster_index];
    else
      printf("ERROR: invalid cluster index %d at position %d\n", cluster_index,
             i);
  }
  return out;
}
