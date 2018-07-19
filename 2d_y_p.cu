// Andrew Gloster
// May 2018
// Example of y direction periodic 2D code

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

#define BLOCK_X 16
#define BLOCK_Y 16


// ---------------------------------------------------------------------
// Main Program
// ---------------------------------------------------------------------

int main()
{	
	// Set the device number
	int deviceNum = 0;

	// Declare Domain Size
	int nx = 512;
	int ny = 512;

	double ly = 2 * M_PI;

	// Domain spacings
	double dy = ly / (double) (nx);

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
			answer[j * nx + i] =- sin(j * dy);
		}
	}


	// // Ensure all the above is completed
	cudaDeviceSynchronize();

	// -----------------------------
	// Set the stencil to compute
	// -----------------------------

	int numSten = 3;
	int numStenTop = 1;
	int numStenBottom = 1;

	double* weights;
	cudaMallocManaged(&weights, numSten * sizeof(double));

	weights[0] = 1.0 / pow(dy, 2.0);
	weights[1] = - 2.0 / pow(dy, 2.0);
	weights[2] = 1.0 / pow(dy, 2.0);

	// -----------------------------
	// Set up device
	// -----------------------------

	// Number of points per device, subdividing in y
	int nxDevice = nx;
	int nyDevice = ny;

	// Set up the compute device structs
	cuSten_t yDirCompute;

	// Initialise the instance of the stencil
	custenCreate2DYp(&yDirCompute, deviceNum, numStreams, numTiles, nxDevice, nyDevice, BLOCK_X, BLOCK_Y, dataNew, dataOld, weights, numSten, numStenTop, numStenBottom);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// -----------------------------
	// Compute
	// -----------------------------

	// Run the computation
	custenCompute2DYp(&yDirCompute, 0);

	// // Synchronise at the end to ensure everything is complete
	cudaDeviceSynchronize();

	for (int j = 0; j < 256; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			printf("%lf %lf %d %d \n", dataNew[j * nx + i], answer[j * nx + i], i, j);
		}
	}

	// -----------------------------
	// Destroy struct and free memory
	// -----------------------------

	// Destroy struct
	custenDestroy2DYp(&yDirCompute);

	// Free memory at the end
	cudaFree(dataOld);
	cudaFree(dataNew);
	cudaFree(answer);
	cudaFree(weights);
	
	// Return 0 when the program completes
	return 0;
}