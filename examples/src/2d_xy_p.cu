// Andrew Gloster
// May 2018
// Example of xy direction periodic 2D code

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
	double ly = 2 * M_PI;

	// Domain spacings
	double dx = lx / (double) (nx);
	double dy = ly / (double) (ny);

	// Set the number of tiles per device
	int numTiles = 1;

	// Initial Conditions
	double* dataInput;
	double* dataOutput;
	double* answer;

	// -----------------------------
	// Allocate the memory 
	// -----------------------------

	cudaMallocManaged(&dataInput, nx * ny * sizeof(double));
	cudaMallocManaged(&dataOutput, nx * ny * sizeof(double));
	cudaMallocManaged(&answer, nx * ny * sizeof(double));

	// -----------------------------
	// Set the initial conditions
	// -----------------------------

	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			dataInput[j * nx + i] = sin(i * dx) * cos(j * dy);
			dataOutput[j * nx + i] = 0.0;
			answer[j * nx + i] = - cos(i * dx) * sin(j * dy);
		}
	}

	// Ensure all the above is completed
	cudaDeviceSynchronize();

	// -----------------------------
	// Set the stencil to compute
	// -----------------------------

	int numStenHoriz = 3;
	int numStenLeft = 1;
	int numStenRight = 1;

	int numStenVert = 3;
	int numStenTop = 1;
	int numStenBottom = 1;

	double* weights;
	cudaMallocManaged(&weights, numStenHoriz * numStenVert * sizeof(double));

	double sigma = 1.0 / (4.0 * dx * dy);

	weights[0] = 1.0 * sigma;
	weights[1] = 0.0 * sigma;
	weights[2] = - 1.0 * sigma;
	weights[3] = 0.0 * sigma;
	weights[4] = 0.0 * sigma;
	weights[5] = 0.0 * sigma;
	weights[6] = - 1.0 * sigma;
	weights[7] = 0.0 * sigma;
	weights[8] = 1.0 * sigma;

	// -----------------------------
	// Set up device
	// -----------------------------

	// Set up the compute device structs
	cuSten_t xyDirCompute;

	// Initialise the instance of the stencil
	cuStenCreate2DXYp(
		&xyDirCompute,

		deviceNum,

		numTiles,

		nx,
		ny,

		BLOCK_X,
		BLOCK_Y,

		dataOutput,
		dataInput,
		weights,

		numStenHoriz,
		numStenLeft,
		numStenRight,

		numStenVert,
		numStenTop,
		numStenBottom
	);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// -----------------------------
	// Compute
	// -----------------------------

	// Run the computation
	cuStenCompute2DXYp(&xyDirCompute, HOST);

	// // Synchronise at the end to ensure everything is complete
	cudaDeviceSynchronize();

	// -----------------------------
	// Destroy struct and free memory
	// -----------------------------

	// Destroy struct
	cuStenDestroy2DXYp(&xyDirCompute);

	// Free memory at the end
	cudaFree(dataInput);
	cudaFree(dataOutput);
	cudaFree(answer);
	cudaFree(weights);
	
	// Return 0 when the program completes
	return 0;
}