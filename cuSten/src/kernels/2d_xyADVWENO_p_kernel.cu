// Andrew Gloster
// November 2018

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


/*! \file 2d_xyADVWENO_p_kernel.cu
    Kernel to apply a xy WENO stencil on a 2D grid - periodic
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
// Code to apply the WENO stencil
// ---------------------------------------------------------------------

/*! \fun static __global__ void kernel2DXnpFun
    \brief Function to apply WENO scheme from Level Set Methods - Fedkiw 
    \param v1 Parameter matching notation from book 
	\param v2 Parameter matching notation from book 
	\param v3 Parameter matching notation from book 
	\param v4 Parameter matching notation from book 
	\param v5 Parameter matching notation from book 
*/

// Notation from Level Set Methods - Fedkiw

template <typename elemType>
__device__ elemType wenoSten
(
	elemType v1,
	elemType v2,
	elemType v3,
	elemType v4,
	elemType v5
)
{
	elemType epsilon = 1e-06;
	elemType phi1, phi2, phi3;
	elemType s1, s2, s3;
	elemType alpha1, alpha2, alpha3;
	elemType denom;
	elemType w1, w2, w3;

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

/*! \fun __global__ void kernel2DXYWENOADVp
    \brief Device function to apply the stencil to the data and output the answer.
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function
	\param uVel Pointer to the x direction advecting velocity 
	\param vVel Pointer to the y direction advecting velocity 
	\param boundaryTop Pointer to data in the top boundary of the current tile
	\param boundaryBottom Pointer to data in the bottom boundary of the current tile
	\param coeDx x direction coefficient
	\param coeDy y direction coefficient
	\param numSten Total number of points in the stencil
	\param numStenHoriz Number of points in a horizontal stencil
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
	\param numStenVert Number of points in a vertical stencil
	\param numStenTop Number of points on top of stencil
	\param numStenBottom Number of points on bottom of stencil
	\param nxLocal Number of points in sharded memory in the x direction
	\param nyLocal Number of points in sharded memory in the y direction
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
	\param nx Total number of points in the x direction
	\param nyTile Number of y direction points on tile
*/

template <typename elemType>
__global__ void kernel2DXYWENOADVp
(
	elemType* dataOutput,  				

	elemType* dataInput,					

	elemType* uVel, 						
	elemType* vVel,						

	elemType* boundaryTop, 				
	elemType* boundaryBottom,				

	const elemType coeDx,					
	const elemType coeDy,					

	const int numSten,					

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

	const int nx,					// Total number of points in x on the tile being computed
	const int nyTile					// Number of y direction points on tile
)
{	
	// -----------------------------	
	// Allocate the shared memory
	// -----------------------------

	extern __shared__ int memory[];

	elemType* arrayLocal = (elemType*)&memory;

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
	// Compute the stencil
	// -----------------------------

	// Inputs
	elemType v1, v2, v3, v4, v5;

	// Output
	elemType Fx, Fy;

	// X direction fluxes
	__syncthreads();

	// Set the index for the i - 3 point
	stenSet = localIdy * nxLocal + (localIdx - numStenLeft);

	if (uVel[globalIdy * nx + globalIdx] > 0.0)
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

	if (vVel[globalIdy * nx + globalIdx] > 0.0)
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

	dataOutput[globalIdy * nx + globalIdx] = uVel[globalIdy * nx + globalIdx] * Fx +  vVel[globalIdy * nx + globalIdx] * Fy;
}

// ---------------------------------------------------------------------
// Function to compute kernel
// ---------------------------------------------------------------------

/*! \fun void cuStenCompute2DXYWENOADVp
    \brief Function called by user to compute the stencil.
    \param pt_cuSten Pointer to cuSten data type which contains all the necessary input
	\param offload Set to HOST to move data back to CPU or DEVICE to keep on the GPU
*/

template <typename elemType>
void cuStenCompute2DXYWENOADVp
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

	// Ensure the current stream is free
	cudaStreamSynchronize(pt_cuSten->streams[1]);

	// Prefetch the tile data
	cudaMemPrefetchAsync(pt_cuSten->dataInput[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->dataOutput[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->uVel[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->vVel[0], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

	// Prefetch the boundary data
	cudaMemPrefetchAsync(pt_cuSten->boundaryTop[0], pt_cuSten->numBoundaryTop * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
	cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[0], pt_cuSten->numBoundaryTop * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

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

			pt_cuSten->nx, 
			pt_cuSten->nyTile
		);

		// Error checking
		sprintf(msgStringBuffer, "Error computing tile %d on GPU %d", tile, pt_cuSten->deviceNum);
		checkError(msgStringBuffer);	

		// Load the next set of data
    	if (tile < pt_cuSten->numTiles - 1)
    	{
    		// Ensure the steam is free to load the data
    		cudaStreamSynchronize(pt_cuSten->streams[1]);

    		// Prefetch the necessary tiles
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile + 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
			cudaMemPrefetchAsync(pt_cuSten->uVel[tile + 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
			cudaMemPrefetchAsync(pt_cuSten->vVel[tile + 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
		 	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile + 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

		 	// Prefetch the next boundaries
		 	cudaMemPrefetchAsync(pt_cuSten->boundaryTop[tile + 1], pt_cuSten->numBoundaryTop * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);
			cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[tile + 1], pt_cuSten->numBoundaryBottom * sizeof(elemType), pt_cuSten->deviceNum, pt_cuSten->streams[1]);

			// Record the event
			cudaEventRecord(pt_cuSten->events[1], pt_cuSten->streams[1]);
    	}

		// Offload the previous set of tiles, this is to ensure we don't get page faults
		if (offload == 1 && tile > 0)
		{
			cudaMemPrefetchAsync(pt_cuSten->dataOutput[tile - 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
	    	cudaMemPrefetchAsync(pt_cuSten->dataInput[tile - 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
			cudaMemPrefetchAsync(pt_cuSten->uVel[tile - 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
			cudaMemPrefetchAsync(pt_cuSten->vVel[tile - 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
			cudaMemPrefetchAsync(pt_cuSten->boundaryTop[tile - 1], pt_cuSten->numBoundaryTop * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
			cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[tile - 1],  pt_cuSten->numBoundaryBottom * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
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
		cudaMemPrefetchAsync(pt_cuSten->dataOutput[pt_cuSten->numTiles - 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
		cudaMemPrefetchAsync(pt_cuSten->dataInput[pt_cuSten->numTiles - 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
		cudaMemPrefetchAsync(pt_cuSten->uVel[pt_cuSten->numTiles - 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
		cudaMemPrefetchAsync(pt_cuSten->vVel[pt_cuSten->numTiles - 1], pt_cuSten->nx * pt_cuSten->nyTile * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
		cudaMemPrefetchAsync(pt_cuSten->boundaryTop[pt_cuSten->numTiles - 1], pt_cuSten->numBoundaryTop * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
		cudaMemPrefetchAsync(pt_cuSten->boundaryBottom[pt_cuSten->numTiles - 1],  pt_cuSten->numBoundaryBottom * sizeof(elemType), cudaCpuDeviceId, pt_cuSten->streams[pt_cuSten->numStreams - 1]);
	}
}

// ---------------------------------------------------------------------
// Explicit instantiation
// ---------------------------------------------------------------------

template
__device__ double wenoSten<double>
(
	double,
	double,
	double,
	double,
	double
);

template
__global__ void kernel2DXYWENOADVp<double>
(
	double*,  				

	double*,					

	double*, 						
	double*,						

	double*, 				
	double*,				

	const double,					
	const double,					

	const int,					

	const int,				
	const int,
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
void cuStenCompute2DXYWENOADVp<double>
(
	cuSten_t<double>*,
	bool
);

template
__device__ float wenoSten<float>
(
	float,
	float,
	float,
	float,
	float
);

template
__global__ void kernel2DXYWENOADVp<float>
(
	float*,  				

	float*,					

	float*, 						
	float*,						

	float*, 				
	float*,				

	const float,					
	const float,					

	const int,					

	const int,				
	const int,
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
void cuStenCompute2DXYWENOADVp<float>
(
	cuSten_t<float>*,
	bool
);

// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------
