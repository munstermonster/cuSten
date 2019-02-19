// Andrew Gloster
// January 2019
// Example of xy direction non periodic 2D code with user function

//   Copyright 2019 Andrew Gloster

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

/*! \var typedef double (*devArg1X)(double*, double*, int);
    \brief The function pointer containing the user defined function to be applied <br>
    Input 1: The pointer to input data to the function <br>
    Input 2: The pointer to the coefficients provided by the user <br>
    Input 3: The current index position (centre of the stencil to be applied) <br>
	Input 4: Value to be used to jump between rows. (j + 1, j - 1 etc.) <br>
	Input 5: Size of stencil in x direction <br>
	Input 6: Size of stencil in y direction
*/

// Data -- Coefficients -- Current node index -- Jump -- Points in x -- Points in y
typedef double (*devArg1XY)(double*, double*, int, int, int, int);

__inline__ __device__ double centralDiff(double* data, double* coe, int loc, int jump, int nx, int ny)
{	
	double result = 0.0;
	int temp;
	int count = 0;

	#pragma unroll
	for (int j = 0; j < ny; j++)
	{
		temp = loc + j * jump;

		for (int i = 0; i < nx; i++)
		{
			result += coe[count] * data[temp + i];

			count ++;
		}
	}

	return result;
}

__device__ devArg1XY devFunc = centralDiff;

// ---------------------------------------------------------------------
// Main Program
// ---------------------------------------------------------------------

int main()
{	
	// Set the device number
	int deviceNum = 0;

	// Declare Domain Size
	int nx = 128;
	int ny = 128;

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

	// Loop indexes
	int temp;
	int index;

	for (int j = 0; j < ny; j++)
	{
		temp = j * nx;
		for (int i = 0; i < nx; i++)
		{
			index = temp + i;

			dataInput[index] = sin(i * dx) * cos(j * dy);
			dataOutput[index] = 0.0;
			answer[index] = - cos(i * dx) * sin(j * dy);
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

	double* coe;
	cudaMallocManaged(&coe, numStenHoriz * numStenVert * sizeof(double));

	double sigma = 1.0 / (4.0 * dx * dy);

	coe[0] = 1.0 * sigma;
	coe[1] = 0.0 * sigma;
	coe[2] = - 1.0 * sigma;
	coe[3] = 0.0 * sigma;
	coe[4] = 0.0 * sigma;
	coe[5] = 0.0 * sigma;
	coe[6] = - 1.0 * sigma;
	coe[7] = 0.0 * sigma;
	coe[8] = 1.0 * sigma;

	// -----------------------------
	// Set up device
	// -----------------------------

	// Set up the compute device structs
	cuSten_t xyDirCompute;

	// Copy the function to device memory
	double* func;
	cudaMemcpyFromSymbol(&func, devFunc, sizeof(devArg1XY));

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// Initialise the instance of the stencil
	custenCreate2DXYnpFun(
		&xyDirCompute,

		deviceNum,

		numTiles,

		nx,
		ny,

		BLOCK_X,
		BLOCK_Y,

		dataOutput,
		dataInput,
		coe,

		numStenHoriz,
		numStenLeft,
		numStenRight,

		numStenVert,
		numStenTop,
		numStenBottom,

		func
	);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// -----------------------------
	// Compute
	// -----------------------------

	// Run the computation
	custenCompute2DXYnpFun(&xyDirCompute, 0);

	// // Synchronise at the end to ensure everything is complete
	cudaDeviceSynchronize();

	for (int j = 0; j < ny; j++)
	{
		temp = j * nx;
		for (int i = 0; i < nx; i++)
		{
			index = temp + i;

			printf("%lf %lf %lf %d %d \n", dataOutput[index], answer[index], dataInput[index], i, j);
		}
	}

	// -----------------------------
	// // Destroy struct and free memory
	// // -----------------------------

	// Destroy struct
	custenDestroy2DXYnpFun(&xyDirCompute);

	// Free memory at the end
	cudaFree(dataInput);
	cudaFree(dataOutput);
	cudaFree(answer);
	cudaFree(coe);
	
	// Return 0 when the program completes
	return 0;
}