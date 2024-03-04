#ifndef TOOLS_H
#define TOOLS_H

#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>

extern int BITS;
extern int REPR_RANGE;
extern float MIN_R, MAX_R;

typedef enum {
  INT,
  FLOAT,
  UINT8,
  UINT16,
} TYPE;

typedef enum {
  RING,
  REC_HALVING,
} SEND;

typedef enum {
  UNIFORM,
  HOMOMORPHIC,
  NON_LINEAR,
  LLOYD,
  KNOWN_RANGE,
} QUANT;

struct unif_quant {
  float min;
  float max;
  uint8_t *vec;
};

struct non_linear_quant {
  float min;
  float max;
  int type;
  uint8_t *vec;
};

struct lloyd_max_quant {
  float min;
  float max;
  float *codebook;
  uint8_t *vec;
};

// NOW FOR UINT16
struct unif_quant_16 {
  float min;
  float max;
  uint16_t *vec;
};

struct non_linear_quant_16 {
  float min;
  float max;
  int type;
  uint16_t *vec;
};

struct lloyd_max_quant_16 {
  float min;
  float max;
  float *codebook;
  uint16_t *vec;
};

float *RandFloatGenerator(size_t lenght, float lowerbound, float upperbound);

void GetEnvVariables(SEND *send_algo, QUANT *quant_algo);

void MinMax(float *vec, size_t lenght, float *min, float *max, int hom_flag);

float sign(float x);

float MeanSquaredError(float *v1, float *v2, size_t lenght);

float NormalizedMSE(float *v1, float *v2, size_t lenght);

void ProcessPrinter(void *obj, size_t lenght, int my_rank, int comm_sz, TYPE t);

#endif
