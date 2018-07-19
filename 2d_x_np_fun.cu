// Andrew Gloster
// May 2018
// Example of x direction non periodic 2D code

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

#define BLOCK_X 8
#define BLOCK_Y 8


// ---------------------------------------------------------------------
// Main Program
// ---------------------------------------------------------------------


__inline__ __device__ double square(double* data, double* coe, int loc)
{	
	return (data[loc - 1] - 2 * data[loc] + data[loc + 1]) * coe[0];	
}

__device__ devArg1X devfunc = square;

int main()
{	
	// Set the device number
	int deviceNum = 0;

	// Declare Domain Size
	int nx = 128;
	int ny = 64;

	double lx = 2 * M_PI;

	// Domain spacings
	double dx = lx / (double) (nx);

	// Set the number of tiles per device
	int numTiles = 1;

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
			dataOld[j * nx + i] = sin(i * dx);
			dataNew[j * nx + i] = 0.0;
			answer[j * nx + i] =- sin(i * dx);
		}
	}


	// // Ensure all the above is completed
	cudaDeviceSynchronize();

	// -----------------------------
	// Set the stencil to compute
	// -----------------------------

	int numSten = 3;
	int numStenLeft = 1;
	int numStenRight = 1;

	int numCoe = 1;

	double* coe;
	cudaMallocManaged(&coe, numCoe * sizeof(double));

	coe[0] = 1.0 / pow(dx, 2.0);

	// -----------------------------
	// Set up device
	// -----------------------------

	// Number of points per device, subdividing in y
	int nxDevice = nx;
	int nyDevice = ny;

	// Set up the compute device structs
	cuSten_t xDirCompute;

	double* func;
	cudaMemcpyFromSymbol(&func, devfunc, sizeof(devArg1X));

	// Initialise the instance of the stencil
	custenCreate2DXnpFun(&xDirCompute, deviceNum, numStreams, numTiles, nxDevice, nyDevice, BLOCK_X, BLOCK_Y, dataNew, dataOld, coe, numSten, numStenLeft, numStenRight, numCoe, func);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// -----------------------------
	// Compute
	// -----------------------------
	
	// Run the computation
	custenCompute2DXnpFun(&xDirCompute, 0);

	// Synchronise at the end to ensure everything is complete
	cudaDeviceSynchronize();

	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			printf("%lf %lf %d \n", dataNew[j * nx + i], answer[j * nx + i], i);
		}
	}

	// -----------------------------
	// Destroy struct and free memory
	// -----------------------------

	// Destroy struct
	custenDestroy2DXnpFun(&xDirCompute);

	// Free memory at the end
	cudaFree(dataOld);
	cudaFree(dataNew);
	cudaFree(answer);
	cudaFree(coe);
	
	// Return 0 when the program completes
	return 0;
}