#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "tools.h"

#define MU 255
#define ALPHA 87.6

void MuLawCompander(float *in, size_t input_size) {
  // some useful values to have stored (to not compute them every time)
  float ln_one_plus_mu = logf(1.0 + MU);

#pragma omp parallel for
  for (int i = 0; i < input_size; i++)
    in[i] = sign(in[i]) * (logf(1 + MU * fabsf(in[i])) / ln_one_plus_mu);
}

void MuLawExpander(float *in, size_t input_size) {
  // some useful values to have stored (to not compute them every time)
  float one_over_mu = 1.0 / MU;
  float one_plus_mu = MU + 1.0;

#pragma omp parallel for
  for (int i = 0; i < input_size; i++)
    in[i] = sign(in[i]) * one_over_mu * (powf(one_plus_mu, fabsf(in[i])) - 1.0);
}

void ALawCompander(float *in, size_t input_size) {
  // some useful values to have stored (to not compute them every time)
  float condition = 1.0 / ALPHA;
  float one_plus_ln_a = 1.0 + logf(ALPHA);
  float one_over_one_plus_ln_a = 1.0 / one_plus_ln_a;
  float alpha_over_one_plus_ln_a = ALPHA * one_over_one_plus_ln_a;

#pragma omp parallel for
  for (int i = 0; i < input_size; i++) {
    float abs = fabsf(in[i]);
    in[i] = (abs < condition)
                ? sign(in[i]) * abs * alpha_over_one_plus_ln_a
                : sign(in[i]) * ((1 + logf(ALPHA * abs)) / one_plus_ln_a);
  }
}

void ALawExpander(float *in, size_t input_size) {
  // some useful values to have stored (to not compute them every time)
  float one_over_alpha = 1.0 / ALPHA;
  float one_plus_ln_a = 1.0 + logf(ALPHA);
  float condition = 1.0 / one_plus_ln_a;

#pragma omp parallel for
  for (int i = 0; i < input_size; i++) {
    float abs = fabsf(in[i]);
    in[i] = (abs < condition)
                ? sign(in[i]) * abs * one_over_alpha * one_plus_ln_a
                : sign(in[i]) * one_over_alpha *
                      powf(M_E, -1.0 + abs * one_plus_ln_a);
  }
}

float *RangeReducer(float *in, size_t input_size, float *min, float *max) {
  float *out = malloc(input_size * sizeof(float));
  MinMax(in, input_size, min, max, 0);
  float range = *max - *min;
  float one_over_range = 1 / range;

#pragma omp parallel for
  for (int i = 0; i < input_size; i++)
    out[i] = ((in[i] - *min) * 2) * one_over_range - 1;

  return out;
}

void RangeRestorer(float *in, size_t input_size, float min, float max) {
  float range = max - min;
  float onehalf = 0.5;

#pragma omp parallel for
  for (int i = 0; i < input_size; i++)
    in[i] = ((in[i] + 1) * range) * onehalf + min;
}

uint8_t *NormalizedSymmetricQuantization(float *in, size_t input_size) {
  uint8_t *out = (uint8_t *)malloc(input_size * sizeof(uint8_t));

  float this_range = (REPR_RANGE - 1) / 2;
  for (int i = 0; i < input_size; i++)
    out[i] = (uint8_t)((in[i] + 1) * this_range);

  return out;
}

uint16_t *NormalizedSymmetricQuantization_16(float *in, size_t input_size) {
  uint16_t *out = (uint16_t *)malloc(input_size * sizeof(uint16_t));

  float this_range = (REPR_RANGE - 1) / 2;
  for (int i = 0; i < input_size; i++)
    out[i] = (uint16_t)((in[i] + 1) * this_range);

  return out;
}

float *NormalizedSymmetricDequantization(uint8_t *in, size_t input_size) {
  float *out = malloc(input_size * sizeof(float));

  float this_range = (REPR_RANGE - 1) / 2;
  for (int i = 0; i < input_size; i++)
    out[i] = ((float)in[i]) / this_range - 1.0;

  return out;
}

float *NormalizedSymmetricDequantization_16(uint16_t *in, size_t input_size) {
  float *out = malloc(input_size * sizeof(float));

  float this_range = (REPR_RANGE - 1) / 2;
  for (int i = 0; i < input_size; i++)
    out[i] = ((float)in[i]) / this_range - 1.0;

  return out;
}

/* In non linear quantization the whole vector gets squished to -1, 1 interval,
 * then a non linear function is called to deform the resulting vector and then
 * a normal uniform quantization is done.
 * In our program, we've implemented two non linear functions for companding,
 * called mu-law and alpha-law. Those methods of quantization are primarely used
 * in signal processing, therefore their presence in this program is more a
 * proof of concept (since we generate uniform random distributed arrays.*/
struct non_linear_quant *NonLinearQuantization(float *in, size_t input_size,
                                               int type, void *struct_ptr) {
  float min, max;
  float *temp = RangeReducer(in, input_size, &min, &max);
  switch (type) {
  case 1: {
    MuLawCompander(temp, input_size);
    break;
  }
  case 2: {
    ALawCompander(temp, input_size);
    break;
  }
  default: {
    printf("ERROR:\tcompanding type not valid\n");
    free(temp);
    return NULL;
  }
  }

  struct non_linear_quant *out = (struct non_linear_quant *)struct_ptr;
  out->vec = NormalizedSymmetricQuantization(temp, input_size);
  out->min = min;
  out->max = max;
  out->type = type;

  free(temp);

  return out;
}

/*Same but for uint16_t*/
struct non_linear_quant_16 *NonLinearQuantization_16(float *in,
                                                     size_t input_size,
                                                     int type,
                                                     void *struct_ptr) {
  float min, max;
  float *temp = RangeReducer(in, input_size, &min, &max);

  switch (type) {
  case 1: {
    MuLawCompander(temp, input_size);
    break;
  }
  case 2: {
    ALawCompander(temp, input_size);
    break;
  }
  default: {
    printf("\tERROR\t companding type not valid\n");
    free(temp);
    return NULL;
  }
  }

  struct non_linear_quant_16 *out = (struct non_linear_quant_16 *)struct_ptr;
  out->vec = NormalizedSymmetricQuantization_16(temp, input_size);
  out->min = min;
  out->max = max;
  out->type = type;

  free(temp);

  return out;
}

/* Inverse proccess of non_linear_quant, here the compressed data gets
 * dequantized, then it passes through the inverse of the companding function,
 * then its range gets restored to pre-quantization values*/
float *NonLinearDequantization(struct non_linear_quant *in, size_t input_size,
                               float *out, int type) {
  out = NormalizedSymmetricDequantization(in->vec, input_size);
  switch (type) {
  case 1: {
    MuLawExpander(out, input_size);
    break;
  }
  case 2: {
    ALawExpander(out, input_size);
    break;
  }
  default: {
    printf("\nERROR:\t companding type not valid\n\n");
    return NULL;
  }
  }

  RangeRestorer(out, input_size, in->min, in->max);
  return out;
}

// Same for uint16_t
float *NonLinearDequantization_16(struct non_linear_quant_16 *in,
                                  size_t input_size, float *out, int type) {
  out = NormalizedSymmetricDequantization_16(in->vec, input_size);
  switch (type) {
  case 1: {
    MuLawExpander(out, input_size);
    break;
  }
  case 2: {
    ALawExpander(out, input_size);
    break;
  }
  default: {
    printf("\tERROR\t companding type not valid\n");
    return NULL;
  }
  }

  RangeRestorer(out, input_size, in->min, in->max);
  return out;
}
