// Andrew Gloster
// May 2018
// Example of y direction periodic 2D code with user function

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------

#include <cmath>
#include <iostream>
#include <cstdio>
#include "cuda.h"
#include "omp.h"

// ---------------------------------------------------------------------
// Custom libraries and headers
// ---------------------------------------------------------------------

#include "cuSten/cuSten.h"

// ---------------------------------------------------------------------
// MACROS
// ---------------------------------------------------------------------

#define BLOCK_X 4
#define BLOCK_Y 4

// ---------------------------------------------------------------------
// Function Declaration
// ---------------------------------------------------------------------

__inline__ __device__ double centralDiff(double* data, double* coe, int loc, int nx)
{	
	double result = 0.0;

	#pragma unroll
	for (int i = 0; i < 9; i++)
	{
		result += coe[i] * data[(loc - 4 * nx) + i * nx];
	}

	return result;
}

__device__ devArg1Y devFunc = centralDiff;

// ---------------------------------------------------------------------
// Main Program
// ---------------------------------------------------------------------

int main()
{	
	// Set the device number
	int deviceNum = 0;

	// Declare Domain Size
	int nx = 64;
	int ny = 64;

	double ly = 2 * M_PI;

	// Domain spacings
	double dy = ly / (double) (ny);

	// Set the number of tiles per device
	int numTiles = 4;

	// Set the number of streams per device
	int numStreams = 3;

	// Initial Conditions
	double* dataOld;
	double* dataNew;
	double* answer;

	// -----------------------------
	// Allocate the memory 
	// -----------------------------

	cudaMallocManaged(&dataOld, nx * ny * sizeof(double));
	cudaMallocManaged(&dataNew, nx * ny * sizeof(double));
	cudaMallocManaged(&answer, nx * ny * sizeof(double));

	// -----------------------------
	// Set the initial conditions
	// -----------------------------

	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			dataOld[j * nx + i] = sin(j * dy);
			dataNew[j * nx + i] = 0.0;
			answer[j * nx + i] = - sin(j * dy);
		}
	}


	// // Ensure all the above is completed
	cudaDeviceSynchronize();

	// -----------------------------
	// Set the stencil to compute
	// -----------------------------

	int numSten = 9;
	int numStenTop = 4;
	int numStenBottom = 4;

	int numCoe = 9;

	double* coe;
	cudaMallocManaged(&coe, numCoe * sizeof(double));

	coe[0] = - (1.0 / 560.0) * 1.0 / pow(dy, 2.0);
	coe[1] = (8.0 / 315.0) * 1.0 / pow(dy, 2.0);
	coe[2] = - (1.0 / 5.0) * 1.0 / pow(dy, 2.0);
	coe[3] = (8.0 / 5.0) * 1.0 / pow(dy, 2.0);
	coe[4] = - (205.0 / 72.0) * 1.0 / pow(dy, 2.0);
	coe[5] = (8.0 / 5.0) * 1.0 / pow(dy, 2.0);
	coe[6] = - (1.0 / 5.0) * 1.0 / pow(dy, 2.0);
	coe[7] = (8.0 / 315.0) * 1.0 / pow(dy, 2.0);
	coe[8] = - (1.0 / 560.0) * 1.0 / pow(dy, 2.0);

	// -----------------------------
	// Set up device
	// -----------------------------

	// Number of points per device, subdividing in y
	int nxDevice = nx;
	int nyDevice = ny;

	// Set up the compute device structs
	cuSten_t yDirCompute;

	double* func;
	cudaMemcpyFromSymbol(&func, devFunc, sizeof(devArg1Y));

	// Initialise the instance of the stencil
	custenCreate2DYpFun(&yDirCompute, deviceNum, numStreams, numTiles, nxDevice, nyDevice, BLOCK_X, BLOCK_Y, dataNew, dataOld, coe, numSten, numStenTop, numStenBottom, numCoe, func);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// -----------------------------
	// Compute
	// -----------------------------

	// Run the computation
	custenCompute2DYpFun(&yDirCompute, 0);

	// Synchronise at the end to ensure everything is complete
	cudaDeviceSynchronize();

	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			printf("%lf %lf %lf %d %d \n", dataOld[j * nx + i], dataNew[j * nx + i], answer[j * nx + i], i, j);
		}
	}

	// -----------------------------
	// Destroy struct and free memory
	// -----------------------------

	// Destroy struct
	custenDestroy2DYpFun(&yDirCompute);

	// Free memory at the end
	cudaFree(dataOld);
	cudaFree(dataNew);
	cudaFree(answer);
	cudaFree(coe);
	
	// Return 0 when the program completes
	return 0;
}