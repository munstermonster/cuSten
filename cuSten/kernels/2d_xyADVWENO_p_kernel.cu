// Andrew Gloster
// November 2018
// Kernel to apply a xy WENO stencil on a 2D grid - periodic

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
#include "../structs/cuSten_struct_type.h"

// ---------------------------------------------------------------------
// Code to apply the WENO stencil
// ---------------------------------------------------------------------

// Notation from Level Set Methods - Fedkiw
static __device__ double wenoSten
(
	double v1,
	double v2,
	double v3,
	double v4,
	double v5
)
{
	double epsilon = 1e-06;
	double phi1, phi2, phi3;
	double s1, s2, s3;
	double alpha1, alpha2, alpha3;
	double denom;
	double w1, w2, w3;

	phi1 = (1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3;
	phi2 = - (1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4;
	phi3 = (1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5;

	s1 = (13.0 / 12.0) * powf(v1 - 2.0 * v2 + v3, 2.0) + 0.25 * powf(v1 - 4.0 * v2 + 3.0 * v3, 2.0); 
	s2 = (13.0 / 12.0) * powf(v2 - 2.0 * v3 + v4, 2.0) + 0.25 * powf(v2 - v4, 2.0);
	s3 = (13.0 / 12.0) * powf(v3 - 2.0 * v4 + v5, 2.0) + 0.25 * powf(3.0 * v3 - 4.0 * v4 + v5, 2.0);

	alpha1 = 0.1 / powf(s1 + epsilon, 2.0);
	alpha2 = 0.6 / powf(s2 + epsilon, 2.0);
	alpha3 = 0.3 / powf(s3 + epsilon, 2.0);

	denom = 1.0 / (alpha1 + alpha2 + alpha3);

	w1 = alpha1 * denom;
	w2 = alpha2 * denom;
	w3 = alpha3 * denom;

	return phi1 * w1 + phi2 * w2 + phi3 * w3;
}

// ---------------------------------------------------------------------
//  Kernel Definition
// ---------------------------------------------------------------------

// We don't care about corner values
static __global__ void kernel2DXYWENOADVp
(
	double* dataOutput,  				// Answer data for x direction

	double* dataInput,					// Input data

	double* uVel, 						// X direction velocity
	double* vVel,						// Y direction velocity

	double* boundaryTop, 				// Data for the top boundary
	double* boundaryBottom,				// Data for the bottom boundary

	const double coeDx,					// x direction coefficient
	const double coeDy,					// y direction coefficient

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

	// -----------------------------
	// Set the indexing
	// -----------------------------

	// True matrix index
    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

	// Local matrix index
	int localIdx = threadIdx.x + numStenLeft;
	int localIdy = threadIdx.y + numStenTop;

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
	// Compute the stencil
	// -----------------------------

	// Inputs
	double v1, v2, v3, v4, v5;

	// Output
	double Fx, Fy;

	// X direction fluxes
	__syncthreads();

	// Set the index for the i - 3 point
	stenSet = localIdy * nxLocal + (localIdx - numStenLeft);

	if (uVel[globalIdy * nxDevice + globalIdx] > 0.0)
	{
		stenSet += 1;

		v1 = (arrayLocal[stenSet] - arrayLocal[stenSet - 1]) * coeDx;

		stenSet += 1;

		v2 = (arrayLocal[stenSet] - arrayLocal[stenSet - 1]) * coeDx;
		
		stenSet += 1;

		v3 = (arrayLocal[stenSet] - arrayLocal[stenSet - 1]) * coeDx;
		
		stenSet += 1;

		v4 = (arrayLocal[stenSet] - arrayLocal[stenSet - 1]) * coeDx;

		stenSet += 1;

		v5 = (arrayLocal[stenSet] - arrayLocal[stenSet - 1]) * coeDx;
	}

	// Flip for case vel < 0.0
	else
	{
		stenSet += 2;

		v5 = (arrayLocal[stenSet] - arrayLocal[stenSet - 1]) * coeDx;

		stenSet += 1;

		v4 = (arrayLocal[stenSet] - arrayLocal[stenSet - 1]) * coeDx;
		
		stenSet += 1;

		v3 = (arrayLocal[stenSet] - arrayLocal[stenSet - 1]) * coeDx;
		
		stenSet += 1;

		v2 = (arrayLocal[stenSet] - arrayLocal[stenSet - 1]) * coeDx;

		stenSet += 1;

		v1 = (arrayLocal[stenSet] - arrayLocal[stenSet - 1]) * coeDx;
	}

	Fx = wenoSten(v1, v2, v3, v4, v5);

	// Set the index for the j - 3 point
	stenSet = (localIdy - numStenTop) * nxLocal + localIdx;

	if (vVel[globalIdy * nxDevice + globalIdx] > 0.0)
	{
		stenSet += nxLocal;

		v1 = (arrayLocal[stenSet] - arrayLocal[stenSet - nxLocal]) * coeDy;

		stenSet += nxLocal;

		v2 = (arrayLocal[stenSet] - arrayLocal[stenSet - nxLocal]) * coeDy;
		
		stenSet += nxLocal;

		v3 = (arrayLocal[stenSet] - arrayLocal[stenSet - nxLocal]) * coeDy;
		
		stenSet += nxLocal;

		v4 = (arrayLocal[stenSet] - arrayLocal[stenSet - nxLocal]) * coeDy;

		stenSet += nxLocal;

		v5 = (arrayLocal[stenSet] - arrayLocal[stenSet - nxLocal]) * coeDy;
	}

	// Flip for case vel < 0.0
	else
	{
		stenSet += 2 * nxLocal;

		v5 = (arrayLocal[stenSet] - arrayLocal[stenSet - nxLocal]) * coeDy;

		stenSet += nxLocal;

		v4 = (arrayLocal[stenSet] - arrayLocal[stenSet - nxLocal]) * coeDy;
		
		stenSet += nxLocal;

		v3 = (arrayLocal[stenSet] - arrayLocal[stenSet - nxLocal]) * coeDy;
		
		stenSet += nxLocal;

		v2 = (arrayLocal[stenSet] - arrayLocal[stenSet - nxLocal]) * coeDy;

		stenSet += nxLocal;

		v1 = (arrayLocal[stenSet] - arrayLocal[stenSet - nxLocal]) * coeDy;
	}

	Fy = wenoSten(v1, v2, v3, v4, v5);

	// -----------------------------
	// Copy back to global
	// -----------------------------

	__syncthreads();

	dataOutput[globalIdy * nxDevice + globalIdx] = uVel[globalIdy * nxDevice + globalIdx] * Fx +  vVel[globalIdy * nxDevice + globalIdx] * Fy;
}

// ---------------------------------------------------------------------
// Function to compute kernel
// ---------------------------------------------------------------------

void custenCompute2DXYWENOADVp
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

	// Ensure the current stream is free
	cudaStreamSynchronize(pt_cuSten->streams[1]);

	// Prefetch the tile data
	cudaMemPrefetchAsync(pt_cuSten->dataInput[0], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	cudaMemPrefetchAsync(pt_cuSten->dataOutput[0], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	cudaMemPrefetchAsync(pt_cuSten->uVel[0], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->vVel[0], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

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
		// Check tthe data has loaded so we can begin computing
		cudaEventSynchronize(pt_cuSten->events[0]);

		// Preform the computation on the current tile
		kernel2DXYWENOADVp<<<gridDim, blockDim, pt_cuSten->mem_shared, pt_cuSten->streams[0]>>>(
			pt_cuSten->dataOutput[tile],

			pt_cuSten->dataInput[tile], 

			pt_cuSten->uVel[tile], 
			pt_cuSten->vVel[tile], 

			pt_cuSten->boundaryTop[tile], 
			pt_cuSten->boundaryBottom[tile], 

			pt_cuSten->coeDx,
			pt_cuSten->coeDy, 

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

		// Load the next set of data
    	if (tile < pt_cuSten->numTiles - 1)
    	{
    		// Ensure the steam is free to load the data
    		cudaStreamSynchronize(pt_cuSten->streams[1]);

    		// Prefetch the tiles
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile + 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

			cudaMemPrefetchAsync(pt_cuSten->uVel[tile + 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
			cudaMemPrefetchAsync(pt_cuSten->vVel[tile + 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

		 	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile + 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

		 	// Prefetch the next boundaries
		 	cudaMemPrefetchAsync(pt_cuSten->boundaryTop[tile + 1], pt_cuSten->numBoundaryTop * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
			cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[tile + 1], pt_cuSten->numBoundaryBottom * sizeof(double), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

			cudaEventRecord(pt_cuSten->events[1], pt_cuSten->streams[1]);
    	}

		// Offload the previous set of tiles, this is to ensure we don't get page faults
		if (offload == 1 && tile > 0)
		{
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile - 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
	    	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile - 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
			cudaMemPrefetchAsync(pt_cuSten->uVel[tile - 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
			cudaMemPrefetchAsync(pt_cuSten->vVel[tile - 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
			cudaMemPrefetchAsync(pt_cuSten->boundaryTop[tile - 1], pt_cuSten->numBoundaryTop * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
			cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[tile - 1],  pt_cuSten->numBoundaryBottom * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
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

    // Offload the final set
	if (offload == 1)
	{
		cudaMemPrefetchAsync(pt_cuSten->dataOutput[pt_cuSten->numTiles - 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
		cudaMemPrefetchAsync(pt_cuSten->dataInput[pt_cuSten->numTiles - 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
		cudaMemPrefetchAsync(pt_cuSten->uVel[pt_cuSten->numTiles - 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
		cudaMemPrefetchAsync(pt_cuSten->vVel[pt_cuSten->numTiles - 1], pt_cuSten->nxDevice * pt_cuSten->nyTile * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
		cudaMemPrefetchAsync(pt_cuSten->boundaryTop[pt_cuSten->numTiles - 1], pt_cuSten->numBoundaryTop * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
		cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[pt_cuSten->numTiles - 1],  pt_cuSten->numBoundaryBottom * sizeof(double), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
	}
}

// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------