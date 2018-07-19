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

#define BLOCK_X 32
#define BLOCK_Y 32

// ---------------------------------------------------------------------
// Main Program
// ---------------------------------------------------------------------

int main()
{	
	// Set the device number
	int deviceNum = 0;

	// Declare Domain Size
	int nx = 2048;
	int ny = 1024;

	double lx = 2 * M_PI;

	// Domain spacings
	double dx = lx / (double) (nx);

	// Set the number of tiles per device
	int numTiles = 2;

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
			answer[j * nx + i] = - sin(i * dx);
		}
	}


	// // Ensure all the above is completed
	cudaDeviceSynchronize();

	// -----------------------------
	// Set the stencil to compute
	// -----------------------------

	int numSten = 9;
	int numStenLeft = 4;
	int numStenRight = 4;

	double* weights;
	cudaMallocManaged(&weights, numSten * sizeof(double));

	weights[0] = - (1.0 / 560.0) * 1.0 / pow(dx, 2.0);
	weights[1] = (8.0 / 315.0) * 1.0 / pow(dx, 2.0);
	weights[2] = - (1.0 / 5.0) * 1.0 / pow(dx, 2.0);
	weights[3] = (8.0 / 5.0) * 1.0 / pow(dx, 2.0);
	weights[4] = - (205.0 / 72.0) * 1.0 / pow(dx, 2.0);
	weights[5] = (8.0 / 5.0) * 1.0 / pow(dx, 2.0);
	weights[6] = - (1.0 / 5.0) * 1.0 / pow(dx, 2.0);
	weights[7] = (8.0 / 315.0) * 1.0 / pow(dx, 2.0);
	weights[8] = - (1.0 / 560.0) * 1.0 / pow(dx, 2.0);

	// -----------------------------
	// Set up device
	// -----------------------------

	// Number of points per device, subdividing in y
	int nxDevice = nx;
	int nyDevice = ny;

	// Set up the compute device structs
	cuSten_t xDirCompute;

	// Initialise the instance of the stencil
	custenCreate2DXp(&xDirCompute, deviceNum, numStreams, numTiles, nxDevice, nyDevice, BLOCK_X, BLOCK_Y, dataNew, dataOld, weights, numSten, numStenLeft, numStenRight);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// -----------------------------
	// Compute
	// -----------------------------

	// Run the computation
	custenCompute2DXp(&xDirCompute, 0);

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
	custenDestroy2DXp(&xDirCompute);

	// Free memory at the end
	cudaFree(dataOld);
	cudaFree(dataNew);
	cudaFree(answer);
	cudaFree(weights);

	// Return 0 when the program completes
	return 0;
}