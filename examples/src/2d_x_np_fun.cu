// Andrew Gloster
// May 2018
// Example of x direction non periodic 2D code

//   Copyright 2018 Andrew Gloster

//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at

//       http://www.apache.org/licenses/LICENSE-2.0

//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.


// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------

#include <cmath>
#include <iostream>
#include <cstdio>
#include "cuda.h"

// ---------------------------------------------------------------------
// cuSten - Note the file position is relative
// ---------------------------------------------------------------------

#include "../../cuSten/cuSten.h"

// ---------------------------------------------------------------------
// MACROS
// ---------------------------------------------------------------------

#define BLOCK_X 32
#define BLOCK_Y 32

// ---------------------------------------------------------------------
// Function pointer definition
// ---------------------------------------------------------------------

// Data -- Coefficients -- Stencil Centre Index
typedef double (*devArg1X)(double*, double*, int);

__inline__ __device__ double CentralDifference(double* data, double* coe, int loc)
{	
	return (data[loc - 1] - 2 * data[loc] + data[loc + 1]) * coe[0];	
}

__device__ devArg1X devfunc = CentralDifference;

// ---------------------------------------------------------------------
// Main Program
// ---------------------------------------------------------------------

int main()
{	
	// Set the device number
	int deviceNum = 0;

	// Declare Domain Size
	int nx = 8192;
	int ny = 8192;

	double lx = 2 * M_PI;

	// Domain spacings
	double dx = lx / (double) (nx);

	// Set the number of tiles per device
	int numTiles = 4;

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

	// Set up the compute device structs
	cuSten_t<double> xDirCompute;

	// Copy the function pointer to the device
	double* func;
	cudaMemcpyFromSymbol(&func, devfunc, sizeof(devArg1X));

	// Initialise the instance of the stencil
	cuStenCreate2DXnpFun(&xDirCompute, deviceNum, numTiles, nx, ny, BLOCK_X, BLOCK_Y, dataNew, dataOld, coe, numSten, numStenLeft, numStenRight, numCoe, func);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// -----------------------------
	// Compute
	// -----------------------------
	
	// Run the computation
	cuStenCompute2DXnpFun(&xDirCompute, HOST);

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
	cuStenDestroy2DXnpFun(&xDirCompute);

	// Free memory at the end
	cudaFree(dataOld);
	cudaFree(dataNew);
	cudaFree(answer);
	cudaFree(coe);
	
	// Return 0 when the program completes
	return 0;
}
