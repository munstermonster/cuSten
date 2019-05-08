// Andrew Gloster
// July 2018

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

/*! \file 2d_y_np_fun_kernel.cu
    Kernel to apply a y direction stencil on a 2D grid - non periodic
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

/*! typedef elemType (*devArg1Y)(elemType*, elemType*, int, int);
    \brief The function pointer containing the user defined function to be applied <br>
    Input 1: The pointer to input data to the function <br>
    Input 2: The pointer to the coefficients provided by the user <br>
    Input 3: The current index position (centre of the stencil to be applied) <br>
	Input 4: Value to be used to jump between rows. (j + 1, j - 1 etc.)
*/

template <typename elemType>
struct templateFunc
{
	typedef elemType (*devArg1Y)(elemType*, elemType*, int, int);
};

// ---------------------------------------------------------------------
//  Kernel Definition
// ---------------------------------------------------------------------

/*! \fun __global__ void kernel2DYnpFun
    \brief Device function to apply the stencil to the data and output the answer.
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function
	\param boundaryTop Pointer to data in the top boundary of the current tile
	\param boundaryBottom Pointer to data in the bottom boundary of the current tile
	\param coe Pointer to coefficients to be used in the function pointer
	\param func Function pointer to the function created by the user
	\param numSten Stencil total size 
	\param numStenTop Number of points on top of stencil
	\param numStenBottom Number of points on bottom of stencil
	\param nxLocal Number of points in sharded memory in the x direction
	\param nyLocal Number of points in sharded memory in the y direction
	\param BLOCK_Y Size of thread block in the x direction
	\param nx Total number of points in the x direction
	\param nyTile Number of y direction points on tile
	\param tileTop Check if the current tile is at the top of the domain
	\param tileBottom Check if the current tile is at the bottom of the domain
*/

template <typename elemType>
__global__ void kernel2DYnpFun
(
	elemType* dataOutput,  					
	elemType* dataInput,					
	elemType* boundaryTop, 				
	elemType* boundaryBottom,				
	const elemType* coe,       					
	const elemType* func,					
	const int numSten,					
	const int numStenTop,				
	const int numStenBottom,			
	const int nxLocal,					
	const int nyLocal,					
	const int BLOCK_Y,					
	const int nx,					
	const int nyTile,					
	const int tileTop,					
	const int tileBottom				
)
{	
	// Allocate the shared memory
	extern __shared__ int memory[];

	elemType* arrayLocal = (elemType*)&memory;
	elemType* coeLocal = (elemType*)&arrayLocal[nxLocal * nyLocal];

	// Move the weigths into shared memory
	#pragma unroll
	for (int k = 0; k < numSten; k++)
	{
		coeLocal[k] = coe[k];
	}

	// True matrix index
	int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

	// Local matrix index
	int localIdx = threadIdx.x;
	int localIdy = threadIdx.y + numStenTop;

	// Local sum variable
	elemType sum = 0.0;

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

		stenSet = localIdy * nxLocal + localIdx;

		sum = ((typename templateFunc<elemType>::devArg1Y)func)(arrayLocal, coeLocal, stenSet, nxLocal);

		__syncthreads();

		dataOutput[globalIdy * nx + globalIdx] = sum;
	}

	// Set all top boundary blocks
	if (blockIdx.y == 0)
	{
		if (tileTop != 1)
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

			stenSet = localIdy * nxLocal + localIdx;

			sum = ((typename templateFunc<elemType>::devArg1Y)func)(arrayLocal, coeLocal, stenSet, nxLocal);

			__syncthreads();

			dataOutput[globalIdy * nx + globalIdx] = sum;
		}
		else
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

			if (threadIdx.y < numStenBottom)
			{
				arrayLocal[(threadIdx.y + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
			}

			__syncthreads();

			stenSet = localIdy * nxLocal + localIdx;

			sum = ((typename templateFunc<elemType>::devArg1Y)func)(arrayLocal, coeLocal, stenSet, nxLocal);

			__syncthreads();

			if (threadIdx.y < BLOCK_Y - numStenTop)
			{
				dataOutput[(globalIdy + numStenTop) * nx + globalIdx] = sum;
			}
		}
	}


	// Set the bottom boundary blocks
	if (blockIdx.y == nyTile / BLOCK_Y - 1)
	{
		if (tileBottom != 1)
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

			stenSet = localIdy * nxLocal + localIdx;

			sum = ((typename templateFunc<elemType>::devArg1Y)func)(arrayLocal, coeLocal, stenSet, nxLocal);
			
			__syncthreads();

			dataOutput[globalIdy * nx + globalIdx] = sum;
		}
		else
		{
			arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

			if (threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
			}

			__syncthreads();

			stenSet = localIdy * nxLocal + localIdx;

			sum = ((typename templateFunc<elemType>::devArg1Y)func)(arrayLocal, coeLocal, stenSet, nxLocal);

			__syncthreads();

			if (threadIdx.y < BLOCK_Y - numStenBottom)
			{
				dataOutput[globalIdy * nx + globalIdx] = sum;
			}
		}
	}
}

// ---------------------------------------------------------------------
// Function to compute kernel
// ---------------------------------------------------------------------

/*! \fun void cuStenCompute2DYnpFun
    \brief Function called by user to compute the stencil for 2D xy direction non periodic with user function
    \param pt_cuSten Pointer to cuSten data type which contains all the necessary input
	\param offload Set to HOST to move data back to CPU or DEVICE to keep on the GPU
*/

template <typename elemType>
void cuStenCompute2DYnpFun
(
	cuSten_t<elemType>* pt_cuSten,
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
	cudaMemPrefetchAsync(pt_cuSten->coe, pt_cuSten->numSten * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Ensure the current stream is free
	cudaStreamSynchronize(pt_cuSten->streams[1]);

	// Prefetch the tile data
	cudaMemPrefetchAsync(pt_cuSten->dataInput[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->dataOutput[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Prefetch the boundary data
	cudaMemPrefetchAsync(pt_cuSten->boundaryTop[0], pt_cuSten->numBoundaryTop * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[0], pt_cuSten->numBoundaryTop * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Record the event
	cudaEventRecord(pt_cuSten->events[0], pt_cuSten->streams[1]);

	// Temporary stream and event used for permuting
	cudaStream_t ts;
	cudaEvent_t te;

	// Tile positions
	int tileTop;
	int tileBottom;

	// Loop over the tiles
	for (int tile = 0; tile < pt_cuSten->numTiles; tile++)
	{
		// Set the variables that describe the current tile position
		if (pt_cuSten->numTiles == 1)
		{
			tileTop = 1;
			tileBottom = 1;
		}
		else
		{
			if (tile == 0)
			{
				tileTop = 1;
				tileBottom = 0;
			}
			else if (tile == pt_cuSten->numTiles - 1)
			{
				tileTop = 0;
				tileBottom = 1;
			}
			else
			{
				tileTop = 0;
				tileBottom = 0;
			}
		}

		// Synchronise the events to ensure computation overlaps
		cudaEventSynchronize(pt_cuSten->events[0]);

		// Preform the computation on the current tile
		kernel2DYnpFun<<<gridDim, blockDim, pt_cuSten->mem_shared, pt_cuSten->streams[0]>>>
		(
			pt_cuSten->dataOutput[tile], 
			pt_cuSten->dataInput[tile],
			pt_cuSten->boundaryTop[tile], 
			pt_cuSten->boundaryBottom[tile],
			pt_cuSten->coe, 
			pt_cuSten->devFunc, 
			pt_cuSten->numSten, 
			pt_cuSten->numStenTop, 
			pt_cuSten->numStenBottom,  
			pt_cuSten->nxLocal, 
			pt_cuSten->nyLocal, 
			pt_cuSten->BLOCK_Y, 
			pt_cuSten->nx, 
			pt_cuSten->nyTile, 
			tileTop, 
			tileBottom
		);
		sprintf(msgStringBuffer, "Error computing tile %d on GPU %d", tile, pt_cuSten->deviceNum);
		checkError(msgStringBuffer);	

		// Offload should the user want to
		if (offload == 1)
		{
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[0]);
	    	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[0]);
		}

		// Load the next set of data
    	if (tile < pt_cuSten->numTiles - 1)
    	{
    		// Ensure the stream is free
    		cudaStreamSynchronize(pt_cuSten->streams[1]);

    		// Prefetch the tiles
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile + 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
		 	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile + 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

		 	// Prefetch the next boundaries
		 	cudaMemPrefetchAsync(pt_cuSten->boundaryTop[tile + 1], pt_cuSten->numBoundaryTop * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
			cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[tile + 1], pt_cuSten->numBoundaryBottom * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

			// Record teh event
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
// Explicit instantiation
// ---------------------------------------------------------------------

template
__global__ void kernel2DYnpFun<double>
(
	double*,  					
	double*,					
	double*, 				
	double*,				
	const double*,       					
	const double*,					
	const int,					
	const int,				
	const int,			
	const int,					
	const int,					
	const int,					
	const int,					
	const int,					
	const int,					
	const int				
);

template
void cuStenCompute2DYnpFun<double>
(
	cuSten_t<double>*,
	bool
);

template
__global__ void kernel2DYnpFun<float>
(
	float*,  					
	float*,					
	float*, 				
	float*,				
	const float*,       					
	const float*,					
	const int,					
	const int,				
	const int,			
	const int,					
	const int,					
	const int,					
	const int,					
	const int,					
	const int,					
	const int				
);

template
void cuStenCompute2DYnpFun<float>
(
	cuSten_t<float>*,
	bool
);

// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------
