// Andrew Gloster
// May 2018

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

/*! \file 2d_y_np_kernel.cu
    Kernel to apply a y direction stencil on a 2D grid - periodic
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
//  Kernel Definition
// ---------------------------------------------------------------------

/*! \fun __global__ void kernel2DYp
    \brief Device function to apply the stencil to the data and output the answer.
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function
	\param boundaryTop Pointer to data in the top boundary of the current tile
	\param boundaryBottom Pointer to data in the bottom boundary of the current tile
	\param weights Pointer to coefficients to be used in stencil
	\param numSten Total number of points in the stencil
	\param numStenTop Number of points on top of stencil
	\param numStenBottom Number of points on bottom of stencil
	\param nxLocal Number of points in sharded memory in the x direction
	\param nyLocal Number of points in sharded memory in the y direction
	\param BLOCK_Y Size of thread block in the y direction
	\param nx Total number of points in the x direction
	\param nyTile Number of y direction points on tile
	\param tileTop Check if the current tile is at the top of the domain
	\param tileBottom Check if the current tile is at the bottom of the domain
*/

__global__ void kernel2DYp
(
	double* dataOutput,  					
	double* dataInput,					
	double* boundaryTop, 				
	double* boundaryBottom,				
	const double* weights,       		
	const int numSten,					
	const int numStenTop,				
	const int numStenBottom,			
	const int nxLocal,					
	const int nyLocal,					
	const int BLOCK_Y,					
	const int nx,					
	const int nyTile
)
{	
	// Allocate the shared memory
	extern __shared__ int memory[];

	double* arrayLocal = (double*)&memory;
	double* weigthsLocal = (double*)&arrayLocal[nxLocal * nyLocal];

	// Move the weigths into shared memory
	#pragma unroll
	for (int k = 0; k < numSten; k++)
	{
		weigthsLocal[k] = weights[k];
	}

	// True matrix index
    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

	// Local matrix index
	int localIdx = threadIdx.x;
	int localIdy = threadIdx.y + numStenTop;

	// Local sum variable
	double sum = 0.0;

	// Set index for summing stencil
	int stenSet;

	// Set all interior blocks
	if (blockIdx.y != 0 && blockIdx.y != nyTile / (BLOCK_Y) - 1)
	{
		arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

		if (threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
		}

		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
		}

		__syncthreads();


		stenSet = threadIdx.y * nxLocal + threadIdx.x;

		#pragma unroll
		for (int k = 0; k < numSten; k++)
		{
			sum += weigthsLocal[k] * arrayLocal[stenSet + k * nxLocal];
		}

		__syncthreads();

		dataOutput[globalIdy * nx + globalIdx] = sum;
	}

	// // Set all top boundary blocks
	if (blockIdx.y == 0)
	{
		arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

		if (threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = boundaryTop[threadIdx.y * nx + globalIdx];
		}

		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
		}

		__syncthreads();

		stenSet = threadIdx.y * nxLocal + threadIdx.x;

		#pragma unroll
		for (int k = 0; k < numSten; k++)
		{
			sum += weigthsLocal[k] * arrayLocal[stenSet + k * nxLocal];
		}

		__syncthreads();

		dataOutput[globalIdy * nx + globalIdx] = sum;
	}

	// Set the bottom boundary blocks
	if (blockIdx.y == nyTile / BLOCK_Y - 1)
	{
		arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

		if (threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
		}

		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = boundaryBottom[threadIdx.y * nx + globalIdx];
		}

		__syncthreads();

		stenSet = threadIdx.y * nxLocal + threadIdx.x;

		#pragma unroll
		for (int k = 0; k < numSten; k++)
		{
			sum += weigthsLocal[k] * arrayLocal[stenSet + k * nxLocal];
		}
		
		__syncthreads();

		dataOutput[globalIdy * nx + globalIdx] = sum;
	}
}

// ---------------------------------------------------------------------
// Function to compute kernel
// ---------------------------------------------------------------------

/*! \fun void cuStenCompute2DYp
    \brief Function called by user to compute the stencil.
    \param pt_cuSten Pointer to cuSten data type which contains all the necessary input
	\param offload Set to HOST to move data back to CPU or DEVICE to keep on the GPU
*/

void cuStenCompute2DYp
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
	cudaMemPrefetchAsync(pt_cuSten->weights, pt_cuSten->numSten * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Ensure the current stream is free
	cudaStreamSynchronize(pt_cuSten->streams[1]);

	// Prefetch the tile data
	cudaMemPrefetchAsync(pt_cuSten->dataInput[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->dataOutput[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Prefetch the boundary data
	cudaMemPrefetchAsync(pt_cuSten->boundaryTop[0], pt_cuSten->numBoundaryTop * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[0], pt_cuSten->numBoundaryTop * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

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
		kernel2DYp<<<gridDim, blockDim, pt_cuSten->mem_shared, pt_cuSten->streams[0]>>>(pt_cuSten->dataOutput[tile], pt_cuSten->dataInput[tile], pt_cuSten->boundaryTop[tile], pt_cuSten->boundaryBottom[tile], pt_cuSten->weights, pt_cuSten->numSten, pt_cuSten->numStenTop, pt_cuSten->numStenBottom,  pt_cuSten->nxLocal, pt_cuSten->nyLocal, pt_cuSten->BLOCK_Y, pt_cuSten->nx, pt_cuSten->nyTile);
		sprintf(msgStringBuffer, "Error computing tile %d on GPU %d", tile, pt_cuSten->deviceNum);
		checkError(msgStringBuffer);	

		// Offload should the user want to
		if (offload == 1)
		{
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[0]);
	    	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[0]);
		}

		// Load the next set of data
    	if (tile < pt_cuSten->numTiles - 1)
    	{
    		// Ensure the stream is free
    		cudaStreamSynchronize(pt_cuSten->streams[1]);

    		// Prefetch the tiles
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile + 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
		 	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile + 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

		 	// Prefetch the next boundaries
		 	cudaMemPrefetchAsync(pt_cuSten->boundaryTop[tile + 1], pt_cuSten->numBoundaryTop * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
			cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[tile + 1], pt_cuSten->numBoundaryBottom * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

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