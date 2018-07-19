// Andrew Gloster
// May 2018
// Kernel to apply an x direction stencil on a 2D grid - non periodic

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------

#include <iostream>
#include <cstdio>

// ---------------------------------------------------------------------
// Custom libraries and headers
// ---------------------------------------------------------------------

#include "../util/util.h"
#include "../structs/cuSten_struct_type.h"
#include "../DeviceFunctions.h"

// ---------------------------------------------------------------------
//  Kernel Definition
// ---------------------------------------------------------------------

__global__ void kernel2DXpFun
(
	double* dataNew,  					// Answer data

	double* dataOld,					// Input data

	double* coe,						// User defined coefficients		

	double* func,						// The user input function

	const int numStenLeft,				// Number of points to the left
	const int numStenRight,				// Number of points to the right

	const int numCoe,					// Number of user defined coefficients

	const int nxLocal,					// Number of points in shared memory in x direction
	const int nyLocal,					// Number of points in shared memory in y direction

	const int BLOCK_X,					// Number of threads in block in y

	const int nx						// Total number of points in x
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
		arrayLocal[localIdy * nxLocal + localIdx] = dataOld[globalIdy * nx + globalIdx];

		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataOld[globalIdy * nx + (globalIdx - numStenLeft)];
		}

		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataOld[globalIdy * nx + globalIdx + BLOCK_X];
		}

		__syncthreads();

		stenSet = localIdy * nxLocal + localIdx;

		dataNew[globalIdy * nx + globalIdx] = ((devArg1X)func)(arrayLocal, coeLocal, stenSet);
	}

	// Set all left boundary blocks
	if (blockIdx.x == 0)
	{
		arrayLocal[localIdy * nxLocal + localIdx] = dataOld[globalIdy * nx + globalIdx];

		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataOld[globalIdy * nx + (nx - numStenLeft + threadIdx.x)];
		}

		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataOld[globalIdy * nx + globalIdx + BLOCK_X];
		}

		__syncthreads();

		stenSet = localIdy * nxLocal + localIdx;

		dataNew[globalIdy * nx + globalIdx] = ((devArg1X)func)(arrayLocal, coeLocal, stenSet);

	}

	// Set the right boundary blocks
	if (blockIdx.x == nx / BLOCK_X - 1)
	{
		arrayLocal[localIdy * nxLocal + threadIdx.x + numStenLeft] = dataOld[globalIdy * nx + globalIdx];

		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataOld[globalIdy * nx + (globalIdx - numStenLeft)];
		}

		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataOld[globalIdy * nx + threadIdx.x];
		}

		__syncthreads();

		stenSet = localIdy * nxLocal + localIdx;

		dataNew[globalIdy * nx + globalIdx] = ((devArg1X)func)(arrayLocal, coeLocal, stenSet);
	}
}

// ---------------------------------------------------------------------
// Function to compute kernel
// ---------------------------------------------------------------------

void custenCompute2DXpFun
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

	// Preload the first block
	cudaStreamSynchronize(pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->dataInput[0], pt_cuSten->nxTile * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->dataOutput[0], pt_cuSten->nxTile * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
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
		kernel2DXpFun<<<gridDim, blockDim, pt_cuSten->mem_shared, pt_cuSten->streams[0]>>>(pt_cuSten->dataOutput[tile], pt_cuSten->dataInput[tile], pt_cuSten->coe, pt_cuSten->devFunc, pt_cuSten->numStenLeft, pt_cuSten->numStenRight, pt_cuSten->numCoe, pt_cuSten->nxLocal, pt_cuSten->nyLocal, pt_cuSten->BLOCK_X, pt_cuSten->nxTile);
		cudaEventRecord(pt_cuSten->events[0], pt_cuSten->streams[0]);

		// Offload should the user want to
		if (offload == 1)
		{
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile], pt_cuSten->nxTile * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[0]);
	    	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile], pt_cuSten->nxTile * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[0]);
		}

		// Load the next tile
    	if (tile < pt_cuSten->numTiles - 1)
    	{
    		cudaStreamSynchronize(pt_cuSten->streams[1]);
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile + 1], pt_cuSten->nxTile * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
		 	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile + 1], pt_cuSten->nxTile * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
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