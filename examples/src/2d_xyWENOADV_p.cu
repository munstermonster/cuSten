// Andrew Gloster
// November 2018
// Example of advection in 2D with upwinding WENO

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

#define BLOCK_X 4
#define BLOCK_Y 4

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
	double* u;
	double* v;

	// -----------------------------
	// Allocate the memory 
	// -----------------------------

	cudaMallocManaged(&dataInput, nx * ny * sizeof(double));
	cudaMallocManaged(&dataOutput, nx * ny * sizeof(double));

	cudaMallocManaged(&u, nx * ny * sizeof(double));
	cudaMallocManaged(&v, nx * ny * sizeof(double));

	// -----------------------------
	// Set the initial conditions
	// -----------------------------

	// Indexing
	int temp;
	int index;

	for (int j = 0; j < ny; j++)
	{
		temp = j * nx;

		for (int i = 0; i < nx; i++)
		{
			index = temp + i;

			dataInput[index] = cos(i * dx) * sin(j * dy);
			dataOutput[index] = 0.0;

			u[index] = sin(j * dy);
			v[index] = - sin(i * dx);
		}
	}

	// Ensure all the above is completed
	cudaDeviceSynchronize();

	// -----------------------------
	// Set up device
	// -----------------------------

	// Number of points per device, subdividing in y
	int nxDevice = nx;
	int nyDevice = ny;

	// Set up the compute device structs
	cuSten_t xyWENOCompute;

	// Initialise the instance of the stencil
	custenCreate2DXYWENOADVp(	
		&xyWENOCompute,

		deviceNum,

		numTiles,

		nxDevice,
		nyDevice,

		BLOCK_X,
		BLOCK_Y,

		dx,
		dy,

		u,
		v,

		dataOutput,

		dataInput
	);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// -----------------------------
	// Compute
	// -----------------------------

	// Run the computation
	custenCompute2DXYWENOADVp(&xyWENOCompute, 0);

	// // Synchronise at the end to ensure everything is complete
	cudaDeviceSynchronize();

	// -----------------------------
	// Destroy struct and free memory
	// -----------------------------

	// Destroy struct
	custenDestroy2DXYWENOADVp(&xyWENOCompute);

	// Free memory at the end
	cudaFree(dataInput);
	cudaFree(dataOutput);

	cudaFree(u);
	cudaFree(v);
	
	// Return 0 when the program completes
	return 0;
}