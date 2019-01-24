// Andrew Gloster
// May 2018
// Kernel to apply a y direction stencil on a 2D grid - periodic

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

#include <iostream>
#include <cstdio>

// ---------------------------------------------------------------------
// Custom libraries and headers
// ---------------------------------------------------------------------

#include "../util/util.h"
#include "../struct/cuSten_struct_type.h"

// ---------------------------------------------------------------------
// Function pointer definition
// ---------------------------------------------------------------------

// Data -- Coefficients -- Current node index -- Jump -- Points in x -- Points in y
typedef double (*devArg1XY)(double*, double*, int, int, int, int);

// ---------------------------------------------------------------------
//  Kernel Definition
// ---------------------------------------------------------------------

__global__ void kernel2DXYpFun
(
	double* dataOutput,  				// Answer data

	double* dataInput,					// Input data

	double* boundaryTop, 				// Data for the top boundary
	double* boundaryBottom,				// Data for the bottom boundary

	const double* coe,       			// Stencil coefficients for use in function

	double* func,						// User defined function

	const int numSten,					// Stencil total size 

	const int numStenHoriz,				// Number of points in a horizontal stencil
	const int numStenLeft,				// Number of points on left of stencil
	const int numStenRight,				// Number of points on right of stencil

	const int numStenVert,				// Number of points in a vertical stencil
	const int numStenTop,				// Number of points on top of stencil
	const int numStenBottom,			// Number of points on bottom of stencil

	const int nxLocal,					// Number of points in shared memory in x direction
	const int nyLocal,					// Number of points in shared memory in y direction

	const int BLOCK_X, 					// Number of threads in block in x	
	const int BLOCK_Y,					// Number of threads in block in y

	const int nxDevice,					// Total number of points in x on the tile being computed
	const int nyTile					// Number of y direction points on tile
)
{	
	// -----------------------------	
	// Allocate the shared memory
	// -----------------------------

	extern __shared__ int memory[];

	double* arrayLocal = (double*)&memory;
	double* coeLocal = (double*)&arrayLocal[nxLocal * nyLocal];

	// Move the weigths into shared memory
	#pragma unroll
	for (int k = 0; k < numSten; k++)
	{
		coeLocal[k] = coe[k];
	}

	// -----------------------------
	// Set the indexing
	// -----------------------------

	// True matrix index
    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

	// Local matrix index
	int localIdx = threadIdx.x + numStenLeft;
	int localIdy = threadIdx.y + numStenTop;

	// Local sum variable
	double sum = 0.0;

	// Set index for summing stencil
	int stenSet;

	// -----------------------------
	// Set interior
	// -----------------------------

	arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nxDevice + globalIdx];

	// -----------------------------
	// Set x boundaries
	// -----------------------------

	// If block is in the interior
	if (blockIdx.x != 0 && blockIdx.x != nxDevice / BLOCK_X - 1)
	{
		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nxDevice + (globalIdx - numStenLeft)];
		}

		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nxDevice + globalIdx + BLOCK_X];
		}
	}

	// If block is on the left boundary
	if (blockIdx.x == 0)
	{
		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nxDevice + (nxDevice - numStenLeft + threadIdx.x)];
		}

		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nxDevice + globalIdx + BLOCK_X];
		}
	}

	// Set the right boundary blocks
	if (blockIdx.x == nxDevice / BLOCK_X - 1)
	{
		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nxDevice + (globalIdx - numStenLeft)];
		}

		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nxDevice + threadIdx.x];
		}
	}

	// -----------------------------
	// Set y boundaries
	// -----------------------------

	// Set interior y boundary
	if (blockIdx.y != 0 && blockIdx.y != nyTile / BLOCK_Y - 1)
	{
		if (threadIdx.y < numStenTop )
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nxDevice + globalIdx];
		}

		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + globalIdx];
		}
	}

	// Set top y boundary
	if (blockIdx.y == 0)
	{
		if (threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = boundaryTop[threadIdx.y * nxDevice + globalIdx];
		}

		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + globalIdx];
		}
	}

	// Set bottom y boundary
	if (blockIdx.y == nyTile / BLOCK_Y - 1)
	{
		if (threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nxDevice + globalIdx];
		}

		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = boundaryBottom[threadIdx.y * nxDevice + globalIdx];
		}
	}

	// -----------------------------
	// Corners - Interior of tile
	// -----------------------------

	// Set interior y boundary
	if (blockIdx.y != 0 && blockIdx.y != nyTile / BLOCK_Y - 1)
	{
		// If block is in the interior
		if (blockIdx.x != 0 && blockIdx.x != nxDevice / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nxDevice + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nxDevice + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + (globalIdx + BLOCK_X)];
			}
		}

		// If block is on the left boundary
		if (blockIdx.x == 0)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nxDevice + (nxDevice - numStenLeft + threadIdx.x)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nxDevice + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + (nxDevice - numStenLeft + threadIdx.x)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + (globalIdx + BLOCK_X)];
			}			
		}

		// If block is on the right boundary
		if (blockIdx.x == nxDevice / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nxDevice + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nxDevice + threadIdx.x];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + threadIdx.x];
			}
		}
	}

	// -----------------------------
	// Corners - Top of tile
	// -----------------------------

	// Set top y boundary
	if (blockIdx.y == 0)
	{
		// If block is in the interior
		if (blockIdx.x != 0 && blockIdx.x != nxDevice / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = boundaryTop[threadIdx.y * nxDevice + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = boundaryTop[threadIdx.y * nxDevice + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + (globalIdx + BLOCK_X)];
			}
		}

		// If block is on the left boundary
		if (blockIdx.x == 0)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = boundaryTop[threadIdx.y * nxDevice + (nxDevice - numStenLeft + threadIdx.x)];

			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = boundaryTop[threadIdx.y * nxDevice + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + (nxDevice - numStenLeft + threadIdx.x)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + (globalIdx + BLOCK_X)];
			}
		}

		// If block is on the right boundary
		if (blockIdx.x == nxDevice / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = boundaryTop[threadIdx.y * nxDevice + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = boundaryTop[threadIdx.y * nxDevice + threadIdx.x];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nxDevice + threadIdx.x];
			}
		}
	}

	// -----------------------------
	// Corners - Bottom of tile
	// -----------------------------

	// Set bottom y boundary
	if (blockIdx.y == nyTile / BLOCK_Y - 1)
	{
		// If block is in the interior
		if (blockIdx.x != 0 && blockIdx.x != nxDevice / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nxDevice + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nxDevice + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = boundaryBottom[threadIdx.y * nxDevice + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] =  boundaryBottom[threadIdx.y * nxDevice + (globalIdx + BLOCK_X)];

			}
		}

		// If block is on the left boundary
		if (blockIdx.x == 0)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nxDevice + (nxDevice - numStenLeft + threadIdx.x)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nxDevice + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = boundaryBottom[threadIdx.y * nxDevice + (nxDevice - numStenLeft + threadIdx.x)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] =  boundaryBottom[threadIdx.y * nxDevice + (globalIdx + BLOCK_X)];
			}		
		}

		// If block is on the right boundary
		if (blockIdx.x == nxDevice / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nxDevice + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nxDevice + threadIdx.x];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = boundaryBottom[threadIdx.y * nxDevice + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] =  boundaryBottom[threadIdx.y * nxDevice + threadIdx.x];
			}
		}
	}

	// -----------------------------
	// Compute the stencil
	// -----------------------------

	__syncthreads();

	stenSet = (localIdy - numStenTop) * nxLocal + (localIdx - numStenLeft);

	__syncthreads();


	sum = ((devArg1XY)func)(arrayLocal, coeLocal, stenSet, nxLocal, numStenHoriz, numStenVert);
	
	__syncthreads();

	// -----------------------------
	// Copy back to global
	// -----------------------------

	// printf("%lf \n", sum);
	dataOutput[globalIdy * nxDevice + globalIdx] = sum;

}

// ---------------------------------------------------------------------
// Function to compute kernel
// ---------------------------------------------------------------------

void custenCompute2DXYpFun
(
	cuSten_t* pt_cuSten,

	bool offload
)
{	
	// 	Buffer used for error checking
	char msgStringBuffer[1024];

	// Set current active compute device
	cudaSetDevice(pt_cuSten->deviceNum);
	sprintf(msgStringBuffer, "Setting current device to GPU %d", pt_cuSten->deviceNum);
	checkError(msgStringBuffer);

	dim3 blockDim(pt_cuSten->BLOCK_X, pt_cuSten->BLOCK_Y);
	dim3 gridDim(pt_cuSten->xGrid, pt_cuSten->yGrid);

	// Load the weights
	cudaMemPrefetchAsync(pt_cuSten->coe, pt_cuSten->numSten * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Ensure the current stream is free
	cudaStreamSynchronize(pt_cuSten->streams[1]);

	// Prefetch the tile data
	cudaMemPrefetchAsync(pt_cuSten->dataInput[0], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->dataOutput[0], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Prefetch the boundary data
	cudaMemPrefetchAsync(pt_cuSten->boundaryTop[0], pt_cuSten->numBoundaryTop * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[0], pt_cuSten->numBoundaryTop * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Record the event
	cudaEventRecord(pt_cuSten->events[1], pt_cuSten->streams[1]);

	// Temporary stream and event used for permuting
	cudaStream_t ts;
	cudaEvent_t te;

	// Loop over the tiles
	for (int tile = 0; tile < pt_cuSten->numTiles; tile++)
	{
		// Synchronise the events to ensure computation overlaps
		cudaEventSynchronize(pt_cuSten->events[0]);
		cudaEventSynchronize(pt_cuSten->events[1]);

		// Preform the computation on the current tile
		kernel2DXYpFun<<<gridDim, blockDim, pt_cuSten->mem_shared, pt_cuSten->streams[0]>>>(
			pt_cuSten->dataOutput[tile],
		
			pt_cuSten->dataInput[tile], 

			pt_cuSten->boundaryTop[tile], 
			pt_cuSten->boundaryBottom[tile], 

			pt_cuSten->coe,

			pt_cuSten->devFunc, 

			pt_cuSten->numSten,

			pt_cuSten->numStenHoriz,
			pt_cuSten->numStenLeft,
			pt_cuSten->numStenRight,

			pt_cuSten->numStenVert,
			pt_cuSten->numStenTop, 
			pt_cuSten->numStenBottom,

			pt_cuSten->nxLocal, 
			pt_cuSten->nyLocal,

			pt_cuSten->BLOCK_X, 
			pt_cuSten->BLOCK_Y, 

			pt_cuSten->nxDevice, 
			pt_cuSten->nyTile
		);

		sprintf(msgStringBuffer, "Error computing tile %d on GPU %d", tile, pt_cuSten->deviceNum);
		checkError(msgStringBuffer);	

		cudaEventRecord(pt_cuSten->events[0], pt_cuSten->streams[0]);

		// Offload should the user want to
		if (offload == 1)
		{
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[0]);
	    	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[0]);
		}

		// Load the next set of data
    	if (tile < pt_cuSten->numTiles - 1)
    	{
    		cudaStreamSynchronize(pt_cuSten->streams[1]);

    		// Prefetch the tiles
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile + 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
		 	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile + 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

		 	// Prefetch the next boundaries
		 	cudaMemPrefetchAsync(pt_cuSten->boundaryTop[tile + 1], pt_cuSten->numBoundaryTop * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
			cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[tile + 1], pt_cuSten->numBoundaryBottom * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

			cudaEventRecord(pt_cuSten->events[1], pt_cuSten->streams[1]);
    	}

    	// Permute streams
    	for (int i = 0; i < pt_cuSten->numStreams - 1; i++)
    	{
    		ts = pt_cuSten->streams[i];
    		pt_cuSten->streams[i] = pt_cuSten->streams[i + 1];	
    		pt_cuSten->streams[i + 1] = ts;    			
    	}

    	// Permute events
		te = pt_cuSten->events[0]; pt_cuSten->events[0] = pt_cuSten->events[1]; pt_cuSten->events[1] = te; 
    }
}

// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------