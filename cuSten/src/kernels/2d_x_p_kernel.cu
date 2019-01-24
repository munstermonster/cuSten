// Andrew Gloster
// May 2018
// Kernel to apply an x direction stencil on a 2D grid - non periodic

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
//  Kernel Definition
// ---------------------------------------------------------------------

__global__ void kernel2DXp
(

	double* dataOutput,  					// Answer data

	double* dataInput,					// Input data

	const double* d_weights,       		// Stencil weights

	const int numSten,					// Stencil width
	const int numStenLeft,				// Number of points to the left
	const int numStenRight,				// Number of points to the right

	const int nxLocal,					// Number of points in shared memory in x direction
	const int nyLocal,					// Number of points in shared memory in y direction

	const int BLOCK_X,					// Number of threads in block in y

	const int nxDevice					// Number of points in x on the device
)
{	
	// -----------------------------	
	// Allocate the shared memory
	// -----------------------------

	extern __shared__ int memory[];
	
	double* arrayLocal = (double*)&memory;
	double* weigthsLocal = (double*)&arrayLocal[nxLocal * nyLocal];

	// Move the weigths into shared memory
	#pragma unroll
	for (int k = 0; k < numSten; k++)
	{
		weigthsLocal[k] = d_weights[k];
	}

	// -----------------------------
	// Set the indexing
	// -----------------------------

	// True matrix index
    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

	// Local matrix index
	int localIdx = threadIdx.x + numStenLeft;
	int localIdy = threadIdx.y;

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
		arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nxDevice + globalIdx];

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
		arrayLocal[localIdy * nxLocal + threadIdx.x + numStenLeft] = dataInput[globalIdy * nxDevice + globalIdx];

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
	// Compute the stencil
	// -----------------------------

	__syncthreads();

	stenSet = localIdy * nxLocal + threadIdx.x;

	#pragma unroll
	for (int k = 0; k < numSten; k++)
	{
		sum += weigthsLocal[k] * arrayLocal[stenSet + k];
	}

	__syncthreads();

	// -----------------------------
	// Copy back to global
	// -----------------------------

	dataOutput[globalIdy * nxDevice + globalIdx] = sum;
}

// ---------------------------------------------------------------------
// Function to compute kernel
// ---------------------------------------------------------------------

void custenCompute2DXp
(
	cuSten_t* pt_cuSten,

	bool offload
)
{	
	// Buffer used for error checking
	char msgStringBuffer[1024];

	// Set current active compute device
	cudaSetDevice(pt_cuSten->deviceNum);
	sprintf(msgStringBuffer, "Setting current device to GPU %d", pt_cuSten->deviceNum);
	checkError(msgStringBuffer);

	dim3 blockDim(pt_cuSten->BLOCK_X, pt_cuSten->BLOCK_Y);
	dim3 gridDim(pt_cuSten->xGrid, pt_cuSten->yGrid);

	// Local memory grid sizes
	int local_nx = pt_cuSten->BLOCK_X + pt_cuSten->numStenLeft + pt_cuSten->numStenRight;
	int local_ny = pt_cuSten->BLOCK_Y;

	// Load the weights
	cudaMemPrefetchAsync(pt_cuSten->weights, pt_cuSten->numSten * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Preload the first block
	cudaStreamSynchronize(pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->dataInput[0], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->dataOutput[0], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
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
		kernel2DXp<<<gridDim, blockDim, pt_cuSten->mem_shared, pt_cuSten->streams[0]>>>(pt_cuSten->dataOutput[tile], pt_cuSten->dataInput[tile], pt_cuSten->weights, pt_cuSten->numSten, pt_cuSten->numStenLeft, pt_cuSten->numStenRight, local_nx, local_ny, pt_cuSten->BLOCK_X, pt_cuSten->nxDevice);
		cudaEventRecord(pt_cuSten->events[0], pt_cuSten->streams[0]);

		// Offload should the user want to
		if (offload == 1)
		{
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[0]);
	    	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[0]);
		}

		// Load the next tile
    	if (tile < pt_cuSten->numTiles - 1)
    	{
    		cudaStreamSynchronize(pt_cuSten->streams[1]);
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile + 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
		 	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile + 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
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