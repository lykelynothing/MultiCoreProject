#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "tools.h"
#include "uniform_quantizer.h"


void HomMinMax(float* vec, size_t lenght, float* min, float* max){
	
	float minimum = vec[0];
	float maximum = vec[0];

	#pragma omp parallel for default(none) shared(vec, lenght) \
		reduction(min: minimum) reduction(max: maximum)
	for(int i = 1; i < lenght; i++){
		if(vec[i] < minimum)		minimum = vec[i];
		else if(vec[i] > maximum)	maximum = vec[i];
	}
	
	*min = -minimum;
	*max = maximum;
}


struct unif_quant* HomomorphicQuantization(float* input, size_t input_size, MPI_Comm comm){
  struct unif_quant* out = (struct unif_quant*) malloc(sizeof(struct unif_quant));
  out -> vec = (uint8_t*) malloc(input_size * sizeof(uint8_t));

  float min_max[2];
  HomMinMax(in, input_size, min_max[0], min_max[1]);
  
  MPI_Allreduce(MPI_IN_PLACE, min_max, 2, MPI_FLOAT, MPI_MAX, comm)
	
  out -> min = -min_max[0];
  out -> max = min_max[1];

  //calculate the steps of the quantization
	float range = out->max - out->min;
	float min = out->min;
  float step = range / REPR_RANGE;

	#pragma omp parallel for
	for(int i = 0; i < input_size; i++){
		float quant = floor((in[i] - min) / step);
		out->vec[i] = (uint8_t) quant;
	}

	return out;
}


float* HomomorphicDequantization(uint16_t* quantized, float min, float max, int comm_sz, size_t input_size){
  float range = max - min;
  float step = range / REPR_RANGE;
  float new_min = comm_sz * min;

  float* out = malloc(input_size * sizeof(float));

  for(size_t i = 0; i < input_size; i++)
    out[i] = ((float)quantized[i])*step + new_min;
  
  return out;
}

