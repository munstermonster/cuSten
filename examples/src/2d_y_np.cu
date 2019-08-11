// Andrew Gloster
// July 2018
// Examples - 2D y direction - non periodic

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
			answer[j * nx + i] =- sin(j * dy);
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

	double* weights;
	cudaMallocManaged(&weights, numSten * sizeof(double));

	weights[0] = - (1.0 / 560.0) * 1.0 / pow(dy, 2.0);
	weights[1] = (8.0 / 315.0) * 1.0 / pow(dy, 2.0);
	weights[2] = - (1.0 / 5.0) * 1.0 / pow(dy, 2.0);
	weights[3] = (8.0 / 5.0) * 1.0 / pow(dy, 2.0);
	weights[4] = - (205.0 / 72.0) * 1.0 / pow(dy, 2.0);
	weights[5] = (8.0 / 5.0) * 1.0 / pow(dy, 2.0);
	weights[6] = - (1.0 / 5.0) * 1.0 / pow(dy, 2.0);
	weights[7] = (8.0 / 315.0) * 1.0 / pow(dy, 2.0);
	weights[8] = - (1.0 / 560.0) * 1.0 / pow(dy, 2.0);
	// -----------------------------
	// Set up device
	// -----------------------------

	// Set up the compute device structs
	cuSten_t yDirCompute;

	// Initialise the instance of the stencil
	cuStenCreate2DYnp(&yDirCompute, deviceNum, numTiles, nx, ny, BLOCK_X, BLOCK_Y, dataNew, dataOld, weights, numSten, numStenTop, numStenBottom);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// -----------------------------
	// Compute
	// -----------------------------

	// Run the computation
	cuStenCompute2DYnp(&yDirCompute, HOST);

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
	cuStenDestroy2DYpFun(&yDirCompute);

	// Free memory at the end
	cudaFree(dataOld);
	cudaFree(dataNew);
	cudaFree(answer);
	cudaFree(weights);
	
	// Return 0 when the program completes
	return 0;
}