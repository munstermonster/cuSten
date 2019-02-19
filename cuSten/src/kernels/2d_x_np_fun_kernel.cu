// Andrew Gloster
// May 2018

// Copyright 2018 Andrew Gloster

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*! \file 2d_x_np_fun_kernel.cu
    Functions to apply a non-periodic user function to a 2D domain, x-direction only.
*/

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

/*! typedef double (*devArg1X)(double*, double*, int);
    \brief The function pointer containing the user defined function to be applied <br>
    Input 1: The pointer to input data to the function <br>
    Input 2: The pointer to the coefficients provided by the user <br>
    Input 3: The current index position (centre of the stencil to be applied)
*/

typedef double (*devArg1X)(double*, double*, int);

// ---------------------------------------------------------------------
//  Kernel Definition
// ---------------------------------------------------------------------

/*! \fun static __global__ void kernel2DXnpFun
    \brief Device function to apply the stencil to the data and output the answer.
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function
	\param coe Pointer to coefficients to be used in the function pointer
	\param func Function pointer to the function created by the user
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
	\param numCoe Number of coefficients used by the user in their function pointer
	\param nxLocal Number of points in sharded memory in the x direction
	\param nyLocal Number of points in sharded memory in the y direction
	\param BLOCK_X Size of thread block in the x direction
	\param nx Total number of points in the x direction
*/

__global__ void kernel2DXnpFun
(
	double* dataOutput,  				
	double* dataInput,					
	double* coe,						
	double* func,						
	const int numStenLeft,				
	const int numStenRight,				
	const int numCoe,				
	const int nxLocal,					
	const int nyLocal,					
	const int BLOCK_X,					
	const int nx
)
{	
	// Allocate the shared memory
	extern __shared__ int memory[];

	double* arrayLocal = (double*)&memory;
	double* coeLocal = (double*)&arrayLocal[nxLocal * nyLocal];

	// Move the weigths into shared memory
	#pragma unroll
	for (int k = 0; k < numCoe; k++)
	{
		coeLocal[k] = coe[k];
	}

	// True matrix index
    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

	// Local matrix index
	int localIdx = threadIdx.x + numStenLeft;
	int localIdy = threadIdx.y;

	// Set index for summing stencil
	int stenSet;

	// Set all interior blocks
	if (blockIdx.x != 0 && blockIdx.x != nx / (BLOCK_X) - 1)
	{
		arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

		// Left
		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (globalIdx - numStenLeft)];
		}

		// Right
		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nx + globalIdx + BLOCK_X];
		}

		__syncthreads();

		stenSet = localIdy * nxLocal + localIdx;

		dataOutput[globalIdy * nx + globalIdx] = ((devArg1X)func)(arrayLocal, coeLocal, stenSet);
	}

	// Set all left boundary blocks
	if (blockIdx.x == 0)
	{
		arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + globalIdx];

		// Right
		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x + BLOCK_X] = dataInput[globalIdy * nx + globalIdx + BLOCK_X];
		}

		__syncthreads();

		if (threadIdx.x >= numStenLeft)
		{
			stenSet = localIdy * nxLocal + threadIdx.x;

			dataOutput[globalIdy * nx + globalIdx] = ((devArg1X)func)(arrayLocal, coeLocal, stenSet);
		}
	}

	// Set the right boundary blocks
	if (blockIdx.x == nx / BLOCK_X - 1)
	{
		arrayLocal[localIdy * nxLocal + threadIdx.x + numStenLeft] = dataInput[globalIdy * nx + globalIdx];

		// Left
		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (globalIdx - numStenLeft)];
		}

		__syncthreads();

		if (threadIdx.x < BLOCK_X - numStenRight)
		{
			stenSet = localIdy * nxLocal + localIdx;

			dataOutput[globalIdy * nx + globalIdx] = ((devArg1X)func)(arrayLocal, coeLocal, stenSet);
		}
	}
}

// ---------------------------------------------------------------------
// Function to compute kernel
// ---------------------------------------------------------------------

/*! \fun void custenCompute2DXnpFun
    \brief Function called by user to compute the stencil.
    \param pt_cuSten Pointer to cuSten data type which contains all the necessary input
	\param offload Boolean set by user to 1 if they wish to move the data back to the host after completing computation, 0 otherwise
*/

void custenCompute2DXnpFun
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

	// Load the weights
	cudaMemPrefetchAsync(pt_cuSten->coe, pt_cuSten->numCoe * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Preload the first block
	cudaStreamSynchronize(pt_cuSten->streams[1]);

	// Prefetch the tile data
	cudaMemPrefetchAsync(pt_cuSten->dataInput[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->dataOutput[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Record the event
	cudaEventRecord(pt_cuSten->events[0], pt_cuSten->streams[1]);

	// Temporary stream and event used for permuting
	cudaStream_t ts;
	cudaEvent_t te;

	// Loop over the tiles
	for (int tile = 0; tile < pt_cuSten->numTiles; tile++)
	{
		// Synchronise the events to ensure computation overlaps
		cudaEventSynchronize(pt_cuSten->events[0]);

		// Preform the computation on the current tile
		kernel2DXnpFun<<<gridDim, blockDim, pt_cuSten->mem_shared, pt_cuSten->streams[0]>>>
		(
			pt_cuSten->dataOutput[tile],
		 	pt_cuSten->dataInput[tile], 

		 	pt_cuSten->coe, 
		 	pt_cuSten->devFunc, 

		 	pt_cuSten->numStenLeft, 
		 	pt_cuSten->numStenRight, 
		 	pt_cuSten->numCoe, 
		 	pt_cuSten->nxLocal, 
		 	pt_cuSten->nyLocal, 
		 	pt_cuSten->BLOCK_X, 
		 	pt_cuSten->nx
		 );

		// Error checking 
		sprintf(msgStringBuffer, "Error computing tile %d on GPU %d", tile, pt_cuSten->deviceNum);
		checkError(msgStringBuffer);	

		// Offload should the user want to
		if (offload == 1)
		{
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[0]);
	    	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[0]);
		}

		// Load the next tile
    	if (tile < pt_cuSten->numTiles - 1)
    	{
    		// Ensure the steam is free to load the data
    		cudaStreamSynchronize(pt_cuSten->streams[1]);

      		// Prefetch the necessary tiles  		
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile + 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
		 	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile + 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

			// Record the event
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