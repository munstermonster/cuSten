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

/*! \file 2d_xy_p_fun_kernel.cu
    Kernel to apply a xy direction user function on a 2D grid - periodic
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

/*! typedef double (*devArg1XY)(double*, double*, int, int, int, int);
    \brief The function pointer containing the user defined function to be applied <br>
    Input 1: The pointer to input data to the function <br>
    Input 2: The pointer to the coefficients provided by the user <br>
    Input 3: The current index position (centre of the stencil to be applied) <br>
	Input 4: Value to be used to jump between rows. (j + 1, j - 1 etc.) <br>
	Input 5: Size of stencil in x direction <br>
	Input 6: Size of stencil in y direction
*/

typedef double (*devArg1XY)(double*, double*, int, int, int, int);

// ---------------------------------------------------------------------
//  Kernel Definition
// ---------------------------------------------------------------------

/*! \fun __global__ void kernel2DXYp
    \brief Device function to apply the stencil to the data and output the answer.
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function
	\param boundaryTop Pointer to data in the top boundary of the current tile
	\param boundaryBottom Pointer to data in the bottom boundary of the current tile
	\param coe Pointer to coefficients to be used in the function pointer
	\param func Function pointer to the function created by the user
	\param numSten Stencil total size 
	\param numStenHoriz Number of points in a horizontal stencil
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
	\param numStenVert Number of points in a vertical stencil
	\param numStenTop Number of points on top of stencil
	\param numStenBottom Number of points on bottom of stencil
	\param nxLocal Number of points in sharded memory in the x direction
	\param nyLocal Number of points in sharded memory in the y direction
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the x direction
	\param nx Total number of points in the x direction
	\param nyTile Number of y direction points on tile
	\param tileTop Check if the current tile is at the top of the domain
	\param tileBottom Check if the current tile is at the bottom of the domain
*/

__global__ void kernel2DXYpFun
(
	double* dataOutput,  				
	double* dataInput,					
	double* boundaryTop, 				
	double* boundaryBottom,				
	const double* coe,       			
	double* func,						
	const int numSten,					
	const int numStenHoriz,				
	const int numStenLeft,				
	const int numStenRight,				
	const int numStenVert,				
	const int numStenTop,				
	const int numStenBottom,			
	const int nxLocal,					
	const int nyLocal,					
	const int BLOCK_X, 					
	const int BLOCK_Y,					
	const int nx,					
	const int nyTile					
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

	arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

	// -----------------------------
	// Set x boundaries
	// -----------------------------

	// If block is in the interior
	if (blockIdx.x != 0 && blockIdx.x != nx / BLOCK_X - 1)
	{
		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (globalIdx - numStenLeft)];
		}

		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nx + globalIdx + BLOCK_X];
		}
	}

	// If block is on the left boundary
	if (blockIdx.x == 0)
	{
		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (nx - numStenLeft + threadIdx.x)];
		}

		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nx + globalIdx + BLOCK_X];
		}
	}

	// Set the right boundary blocks
	if (blockIdx.x == nx / BLOCK_X - 1)
	{
		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (globalIdx - numStenLeft)];
		}

		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nx + threadIdx.x];
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
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
		}

		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
		}
	}

	// Set top y boundary
	if (blockIdx.y == 0)
	{
		if (threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = boundaryTop[threadIdx.y * nx + globalIdx];
		}

		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
		}
	}

	// Set bottom y boundary
	if (blockIdx.y == nyTile / BLOCK_Y - 1)
	{
		if (threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
		}

		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = boundaryBottom[threadIdx.y * nx + globalIdx];
		}
	}

	// -----------------------------
	// Corners - Interior of tile
	// -----------------------------

	// Set interior y boundary
	if (blockIdx.y != 0 && blockIdx.y != nyTile / BLOCK_Y - 1)
	{
		// If block is in the interior
		if (blockIdx.x != 0 && blockIdx.x != nx / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx + BLOCK_X)];
			}
		}

		// If block is on the left boundary
		if (blockIdx.x == 0)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (nx - numStenLeft + threadIdx.x)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (nx - numStenLeft + threadIdx.x)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx + BLOCK_X)];
			}			
		}

		// If block is on the right boundary
		if (blockIdx.x == nx / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + threadIdx.x];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + threadIdx.x];
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
		if (blockIdx.x != 0 && blockIdx.x != nx / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = boundaryTop[threadIdx.y * nx + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = boundaryTop[threadIdx.y * nx + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx + BLOCK_X)];
			}
		}

		// If block is on the left boundary
		if (blockIdx.x == 0)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = boundaryTop[threadIdx.y * nx + (nx - numStenLeft + threadIdx.x)];

			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = boundaryTop[threadIdx.y * nx + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (nx - numStenLeft + threadIdx.x)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx + BLOCK_X)];
			}
		}

		// If block is on the right boundary
		if (blockIdx.x == nx / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = boundaryTop[threadIdx.y * nx + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = boundaryTop[threadIdx.y * nx + threadIdx.x];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + threadIdx.x];
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
		if (blockIdx.x != 0 && blockIdx.x != nx / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = boundaryBottom[threadIdx.y * nx + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] =  boundaryBottom[threadIdx.y * nx + (globalIdx + BLOCK_X)];

			}
		}

		// If block is on the left boundary
		if (blockIdx.x == 0)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (nx - numStenLeft + threadIdx.x)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx + BLOCK_X)];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = boundaryBottom[threadIdx.y * nx + (nx - numStenLeft + threadIdx.x)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] =  boundaryBottom[threadIdx.y * nx + (globalIdx + BLOCK_X)];
			}		
		}

		// If block is on the right boundary
		if (blockIdx.x == nx / BLOCK_X - 1)
		{
			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx - numStenLeft)];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + threadIdx.x];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = boundaryBottom[threadIdx.y * nx + (globalIdx - numStenLeft)];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] =  boundaryBottom[threadIdx.y * nx + threadIdx.x];
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
	dataOutput[globalIdy * nx + globalIdx] = sum;

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
	cudaMemPrefetchAsync(pt_cuSten->dataInput[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->dataOutput[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Prefetch the boundary data
	cudaMemPrefetchAsync(pt_cuSten->boundaryTop[0], pt_cuSten->numBoundaryTop * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[0], pt_cuSten->numBoundaryTop * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	sprintf(msgStringBuffer, "Can't prefectch %d", pt_cuSten->deviceNum);
	checkError(msgStringBuffer);

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

			pt_cuSten->nx, 
			pt_cuSten->nyTile
		);

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
			// Ensure the current stream is free
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