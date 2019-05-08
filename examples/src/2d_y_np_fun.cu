// Andrew Gloster
// January 2018
// Examples - 2D y direction - non periodic - user function

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

#define BLOCK_X 8
#define BLOCK_Y 8

// ---------------------------------------------------------------------
// Function pointer definition
// ---------------------------------------------------------------------

// Data -- Coefficients -- Current node index -- Jump
typedef double (*devArg1Y)(double*, double*, int, int);

// ---------------------------------------------------------------------
// Function Declaration
// ---------------------------------------------------------------------

__inline__ __device__ double centralDiff(double* data, double* coe, int loc, int jump)
{	
	double result = 0.0;

	#pragma unroll
	for (int i = 0; i < 9; i++)
	{
		result += coe[i] * data[(loc - 4 * jump) + i * jump];
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
	int nx = 512;
	int ny = 512;

	double ly = 2 * M_PI;

	// Domain spacings
	double dy = ly / (double) (ny);

	// Set the number of tiles per device
	int numTiles = 2;

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

	// Set up the compute device structs
	cuSten_t<double> yDirCompute;

	double* func;
	cudaMemcpyFromSymbol(&func, devFunc, sizeof(devArg1Y));

	// Initialise the instance of the stencil
	cuStenCreate2DYnpFun(&yDirCompute, deviceNum, numTiles, nx, ny, BLOCK_X, BLOCK_Y, dataNew, dataOld, coe, numSten, numStenTop, numStenBottom, func);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// -----------------------------
	// Compute
	// -----------------------------

	// Run the computation
	cuStenCompute2DYnpFun(&yDirCompute, HOST);

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
	// cuStenDestroy2DYpFun(&yDirCompute);

	// Free memory at the end
	cudaFree(dataOld);
	cudaFree(dataNew);
	cudaFree(answer);
	cudaFree(coe);
	
	// Return 0 when the program completes
	return 0;
}
