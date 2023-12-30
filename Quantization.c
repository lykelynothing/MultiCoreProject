#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>		// used for the definition of int8 (int8_t in the library), not really necessary since we can use char
#include <omp.h>
#include <math.h>		// used for floor function, not really necessary since we can do bitwise operation
#include <time.h>		// used to random generate vectors

/* \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
 * \\\\\\\\\\\\\\\\\\\\		ROADMAP		\\\\\\\\\\\\\\\\\\\\\\\\\
 * \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
 *
 * We have done a uniform affine quantization algorithm (which we need
 * to finish to set up).
 *
 * Some more precise and robust quantization error measures are needed
 * because MSE is subjective to sample magnitude (vector with bigger
 * values will have a bigger MSE regardless of the magnitude of the error
 * itself).
 *
 * An idea could be to divide the MSE with the dimension of the range [a,b].
 *
 * Another possible measure is expectation:
 * E(e)=I(e^2 p(e) de)			interval is [a,b]
 *					p(e)=1/[a,b] if p is uniform
 * legenda: e-> error E->expectation I->integral p->probability
 * 
 * Expectation of quantization error power is:
 * D=E((x-Q(x))^2)
 * legenda: Q(x)->quantization function (quantization followed by dequantization)
 *
 * We need to make a symmetric one, so that we can improve its efficiency.
 * And a way to reliably set up the quantization interval.
 * 
 * Next step are to implement other quantization algorithms.
 * 
 * Non linear quantization (companding) algorithm is a must.
 * It consists in apllying a non linear function to our vector to "boost
 * small values" or values that simply need more sampling and then applying
 * a uniform quantization.
 *
 * Standardized non linear function for companding are A-Low and mu-law.
 *
 * Lloyd-Max is another must. In the Lloyd-Max most liekely values get
 * a smaller quantization step (compared to Companding for which smaller
 * values get a smaller step).
 * With Lloyd-Max we want to minimaze Expecation of quantization error power.
 * We must set decision boundaries bk that works as values to round to.
 * Once we set up decision boundaries, the Lloyd-Max becomes practically a 
 * 1-NN algorithm.
 * For this reason decision boundaries must be set in the middle of two
 * reconstruction values.
 * Reconstruction values thus are:
 * yk=I(x p(x) dx) / I(p(x) dx)			interval is [bk-1, bk]
 * legenda: I->integral bk->decison boundaries yk->reconstruction values
 * Reconstruction values are also called centroids.
 * Since bk and yk are intertwined, and we know k (range of quantized values)
 * we use an algorithm similar to Masi's k-means algorithm to set those 
 * boundaries.
 * The algorithm in question:
 * ///////////////////////	LLOYD-MAX	////////////////////////////
 * 1) Start (initialize the iteration) with a random assignment of k
 *	reconstruction values (codewords)
 * 2) Using the reconstruction values, compute the boundary values as mid-points
 *	between 2 reconstruction values / codewords (nearest neighbour rule).
 * 3) Using the pdf of our signal and the boundary values, compute new
 *	reconstruction values (codewords) as centroids over the quantisation
 *	areas (conditional expectation/centroid).
 * 4) Go to 2) until update is sufficiently small (< epsilon).
 *
 * The third must have quantization algorithm is the Vector Quantizer (VQ)
 * with Linde-Buzo-Gray (LBG) algorithm.
 * It is a multi-dimensional generalization on Lloyd-Max algorithm:
 * instead of quantizing every value into another one "statically", we group 
 * each value of our vector into a group of n<N (N%n==0) elements (basically
 * creating an N dimensional vector space from our vector of original size N).
 * It's best case scenario is when our starting vector data has some
 * correlation but it's surplisingly very good also when we have uncorrelated
 * data.
 * Basically you use a M-NN algorithm to set vector as decision boundaries
 * (codevectora) when quantizing the datas.
 * To dequantize it you need the collection of codevectors (codebook) and the
 * size of it determines also the size and precision of the quantization.
 * To obtain a codebook you can use the LBG algorithm.
 * ///////////////////////	LBG ALGORITHM	/////////////////////////////
 * 1) Start (initialize the iteration) with a random assignment of N/n 
 *	n-dimensional codevectors Yk
 * 2) Using the codewords, compute the decision boundary Bk as the set of all
 *	points with equal distance between 2 reconstruction values/codevectors
 *	(the such constructed regions are also called Voronoi-regions), using
 *	the nearest neighbour rule. To assign a vector to a specific region,
 *	we use the nearest neighbour rule directly. We simply test which
 *	codevector is closest to the observed vector.
 * 3) Using the pdf of our signal and the decision boundary (Voronoi region),
 *	compute new codevectors Yk as centroids (center of mass) or conditional
 *	expectation over the quantisation areas (the Voronoi region).
 * 4) Go to 2) until update is sufficiently small (< epsilon).
 *
 *
 * We can attempt to use PCA to create our quantization algorithm.
 * (see how much time we still have)
 *
 * Then it's time to implemet ALLTOALL and ALLREDUCE MPI collectives.
 *
 * Then we need to set up how to select algorithm.
 *
 * Then we need to adjuct our algorithms with various quantization lenght
 * and give the user a way to select this lenght.
 *
 * We surely need a vector dataset and a way to make it work with our program,
 * because generating random float vector forever could generate some biases
 * on the paper research.
 *
 * Then we need to use a way to measure efficency of those collectives.
 *
 * Then we do experiments and we write the paper.
 *
 * ////////////////////////////////////////////////////////////////////////////////
 * ///////////////////		FILE SEPARATION			///////////////////
 * ///////////////////		     (idea)			///////////////////
 * ////////////////////////////////////////////////////////////////////////////////
 * all error metrics and vector analytics----------->	measures.c 
 * affine quantization------------------------------>	uniform_quantizer.c 
 * symmetric quantization--------------------------->	uniform_quantizer.c
 * non-linear quantization-------------------------->	non_linear_quantizer.c 
 * lloyd-max algorirthm----------------------------->	vector_quantizer.c 
 * VQ/LBG algotithm--------------------------------->	vector_quantizer.c 
 * allreduce---------------------------------------->	collective.c 
 * alltoall----------------------------------------->	collective.c 
 * 
 * \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
 * \\\\\\\\\\\\\\\\\\\\		FROM HERE TAKE EVERYTHING	\\\\\\\\\\\\\\\\\\\
 * \\\\\\\\\\\\\\\\\\\\		WITH A PINCCH OF SALT		\\\\\\\\\\\\\\\\\\\
 * \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
 *
 * Rough ideas:
 * 
 * offset = mean of the vector???????
 * utilizing symmetric quantization discard offset, so can be better
 *
 * SCALE SHOULD BE CHOSEN WITH THE AIM OF MAXIMIZING DATA VARIANCE
 * how we pre-process it/fine tune it? So what shound [a,b] be?
 * - [min,max] when you  have something that looks like a normal distribution
 * - [min observed, max observed] when normal distribution and vector is too big
 * - entropy
 * - mean squared error
 * 
 * Can we use PCA to precompute those datas (i.e. find a range thar preserves the most variance)?
 * Is it computationally heavy? Is it useful?
 */

struct q_val{
	float mean;
	float min;
	float max;
};

int thread_count;

void RandFloatGenerator(float* list, int lenght, float upperbound);

void AffineQuantization(int8_t* out, float* in, int lenght, float scale, float offset);

void AffineDequantization(float* out, int8_t* in, int lenght, float scale, float offset);

void PrintFloatVec(float* vec, int lenght, char* prompt);

void PrintInt8Vec(int8_t* vec, int lenght, char* prompt);

float MeanSquaredError(float* v1, float* v2, int lenght);

float VectorMean(float* vec, int lenght);

void VectorDatas(float* vec, int lenght, struct q_val* out);

int main(int argc, char** argv){
	srand((unsigned int) time(NULL));
	thread_count = 8;

	int dim = strtol(argv[1], NULL, 10);
	float* list = malloc(dim*sizeof(float));

	float scale = 10.0;
	RandFloatGenerator(list, dim, 256.0*scale);

	struct q_val datas;
	VectorDatas(list, dim, &datas);

	int8_t* quantized_data = malloc(dim*sizeof(int8_t));
	
	AffineQuantization(quantized_data, list, dim, scale, 0);

	float* dequantized_data = malloc(dim*sizeof(float));

	AffineDequantization(dequantized_data, quantized_data, dim, scale, 0);

	char* p1 = "The float list is:";
	PrintFloatVec(list, dim,p1);

	printf("Mean: %f \tMin: %f\t Max: %f\n\n\n", datas.mean, datas.min, datas.max);
	
	char* p2 = "The quantized int8 list is:";
	PrintInt8Vec(quantized_data, dim, p2);

	char* p3 = "The dequantized float list is:";
	PrintFloatVec(dequantized_data, dim, p3);
	
	printf("The mean squared error between the two lists is: %f \n", MeanSquaredError(list, dequantized_data, dim));
	free(list);
	free(quantized_data);
	free(dequantized_data);

	return 0;
}


/* generates a random vector of float parallelizing the work	*
 * on more threads and using rand_r (a thread safe version of	*
 * time.h rand that requires an explicit seed) and an		*
 * upperbound. The values are then zero meaned.			*
 *								*
 * input: pointer to float array, array lenght, upperbound	*/
void RandFloatGenerator(float* list, int lenght, float upperbound){
	int i=0;
	unsigned int seed;
	float off=upperbound/2;
	#pragma omp parallel num_threads(thread_count)\
		default(none) shared(list, lenght, upperbound, off) private(i, seed)
	{
		seed=rand();
		#pragma omp for
		for(i=0; i<lenght; i++)
			list[i] = (float) rand_r(&seed) / (float) (RAND_MAX/upperbound) - off;
	}
}

//		FLOAT32---->INT8				//
/* Applies the affine quantization rounding to closest number	*
 * and utilizing more thread in a parallel for.			*
 * The result is written into an int8 array called out.		*
 *								*
 * input:	output pointer,					* 
 *		input float pointer,				*
 *		array lenght,					*
 *		scale,						*
 *		offset						*/
void AffineQuantization(int8_t* out, float* in, int lenght, float scale, float offset){
	#pragma omp parallel for default(none) shared(in, out, lenght, scale, offset)
	for(int i=0; i<lenght; i++){
		float quant = floor(in[i]/scale + offset + 0.5);
		out[i] = (quant < -128) ? INT8_MIN : (quant > 127) ? INT8_MAX : (int8_t) quant;
	}
}


//		INT8---->FLOAT32				//
/* Dequantize the int8 input array into an ourput float array	*
 * utilizing more thread in a parallel for.			*
 *								*
 * input:	output pointer,					*
 *		input pointer,					*
 *		input lenght,					*
 *		scale,						*
 *		offset						*/
void AffineDequantization(float* out, int8_t* in, int lenght, float scale, float offset){
	#pragma omp parallel for default(none) shared(out, in, lenght, scale, offset)
	for(int i=0; i<lenght; i++)
		out[i] = (((float) in[i]) + offset)*scale;
}


//MeanSquaredError between two float arrays
float MeanSquaredError(float* v1, float* v2, int lenght){
	float out=0;
	#pragma omp parallel for default(none) shared(v1, v2, lenght) reduction(+:out)
	for(int i = 0; i<lenght; i++)
		out+= (v1[i]-v2[i])*(v1[i]-v2[i]);

	out = out/(float)lenght;
	
	return out;
}


//Mean of a float array, can be used as offset of affine reduction
float VectorMean(float* vec, int lenght){
	float out = 0;
	float len = (float) len;
	#pragma omp parallel for default(none) shared(vec,lenght, len) reduction(+:out)
	for(int i = 0; i<lenght; i++)
		out+= vec[i]/len;

	return out;
}

/* Utilizes a struct to store datas about the vector to		*
 * quantize. The desired data to store are the mean, the	*
 * highest value and the lowest value.				*
 * OpenMP is used to parallelize the work and three reduction	*
 * variables are declared to do the job.			*
 * Those data are then used to calculate the range [a,b] to use	*
 * in the affine or symmetric quantization.			*
 *								*
 * input:	input vector pointer,				*
 *		input vector lenght,				*
 *		output structure pointer			*/
void VectorDatas(float* vec, int lenght, struct q_val* out){
	float len = (float) lenght;
	float mean = vec[0]/len;
	float minimum = vec[0];
	float maximum = vec[0];
	#pragma omp parallel for default(none) shared(vec, lenght, len) \
		reduction(+: mean) reduction(min: minimum) reduction(max: maximum)
	for(int i=1; i<lenght;i++){
		mean += vec[i]/len;
		if(vec[i] < minimum)		minimum = vec[i];
		else if(vec[i] > maximum)	maximum = vec[i];
	}

	out->mean = mean;
	out->min = minimum;
	out->max = maximum;
}

//Prints a vector of float values, has the option of including a text
void PrintFloatVec(float* vec, int lenght, char* prompt){
	printf("%s \n", prompt);
	for (int i = 0; i<lenght; i++) printf("%f \t", vec[i]);
	printf("\n\n\n");
}

//Prints a vector of int values, has the option of including a text
void PrintInt8Vec(int8_t* vec, int lenght, char* prompt){
	printf("%s \n", prompt);
	for (int i = 0; i<lenght; i++) printf("%d \t", vec[i]);
	printf("\n\n\n");
}
