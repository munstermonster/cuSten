// Andrew Gloster
// January 2019

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

/*! \file 2d_xy_np_kernel.cu
    Kernel to apply a xy direction stencil on a 2D grid - non periodic
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

/*! \fun __global__ void kernel2DXYnp
    \brief Device function to apply the stencil to the data and output the answer.
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function
	\param boundaryTop Pointer to data in the top boundary of the current tile
	\param boundaryBottom Pointer to data in the bottom boundary of the current tile
	\param weights Pointer to coefficients to be used in stencil
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
	\param tileTop Check if the current tile is at the top of the domain
	\param tileBottom Check if the current tile is at the bottom of the domain
*/

__global__ void kernel2DXYnp
(
	double* dataOutput,  				
	double* dataInput,					
	double* boundaryTop, 				
	double* boundaryBottom,				
	const double* weights,       		
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
	const int nyTile,					
	const int tileTop,					
	const int tileBottom				
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
		weigthsLocal[k] = weights[k];
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

	// Set temporary index for looping
	int temp;

	// Use to loop over indexing in the weighsLocal
	int weight = 0;

	// -----------------------------
	// We divide the domain in 9 - 4x Corners, 4x Edges, 1x Interior
	// -----------------------------

	// -----------------------------
	// (0, 0) - Top Left
	// -----------------------------

	if (blockIdx.x == 0 && blockIdx.y == 0)
	{
		// ----------
		// Copy
		// ----------
	
		// Interior
		arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

		// Right
		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nx + globalIdx + BLOCK_X];
		}

		// Bottom
		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
		}

		// Bottom Right
		if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx + BLOCK_X)];
		}

		// Top
		if (tileTop != 1)
		{	
			// Top
			if (threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + localIdx] = boundaryTop[threadIdx.y * nx + globalIdx];
			}

			// Top Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = boundaryTop[threadIdx.y * nx + (globalIdx + BLOCK_X)];
			}
		}

		// Ensure copying completed
		__syncthreads();

		// ----------
		// Compute
		// ----------

		stenSet = threadIdx.y * nxLocal + threadIdx.x;
		weight = 0;

		for (int j = 0; j < numStenVert; j++) // Allow for the point we're actually at
		{
			temp = j * nxLocal;

			for (int i = 0; i < numStenHoriz; i++) // Allow for the point we're actually at
			{
				sum += weigthsLocal[weight] * arrayLocal[stenSet + temp + i];

				weight++;
			} 
		}

		// Ensure the compute is complete
		__syncthreads();

		// ----------
		// Copy back 
		// ----------

		if (tileTop == 1)
		{
			if (threadIdx.x >= numStenLeft && threadIdx.y >= numStenTop)
			{
				dataOutput[globalIdy * nx + globalIdx] = sum;
			}
		}
		else
		{
			if (threadIdx.x >= numStenLeft)
			{
				dataOutput[globalIdy * nx + globalIdx] = sum;
			}
		}
		
	}

	// -----------------------------
	// (nx / BLOCK_X - 1, 0) - Top Right
	// -----------------------------

	else if (blockIdx.x == nx / BLOCK_X - 1 && blockIdx.y == 0)
	{
		// ----------
		// Copy
		// ----------

		// Interior
		arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

		// Left 
		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (globalIdx - numStenLeft)];
		}

		// Bottom
		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
		}

		// Bottom Left
		if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx - numStenLeft)];
		}

		// Top
		if (tileTop != 1)
		{	
			// Top
			if (threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + localIdx] = boundaryTop[threadIdx.y * nx + globalIdx];
			}

			// Top Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = boundaryTop[threadIdx.y * nx + (globalIdx - numStenLeft)];
			}
		}

		// Ensure copying completed
		__syncthreads();

		// ----------
		// Compute
		// ----------

		stenSet = threadIdx.y * nxLocal + threadIdx.x;
		weight = 0;

		for (int j = 0; j < numStenVert; j++) // Allow for the point we're actually at
		{
			temp = j * nxLocal;

			for (int i = 0; i < numStenHoriz; i++) // Allow for the point we're actually at
			{
				sum += weigthsLocal[weight] * arrayLocal[stenSet + temp + i];

				weight++;
			} 
		}

		// Ensure the compute is complete
		__syncthreads();

		// ----------
		// Copy back 
		// ----------

		if (tileTop == 1)
		{
			if (threadIdx.x < BLOCK_X - numStenRight && threadIdx.y >= numStenTop)
			{
				dataOutput[globalIdy * nx + globalIdx] = sum;
			}
		}
		else
		{
			if (threadIdx.x < BLOCK_X - numStenRight)
			{
				dataOutput[globalIdy * nx + globalIdx] = sum;
			}
		}

	}

	// -----------------------------
	// (0, nyTile / BLOCK_Y - 1) - Bottom Left
	// -----------------------------

	else if (blockIdx.x == 0 && blockIdx.y == nyTile / BLOCK_Y - 1)
	{
		// ----------
		// Copy
		// ----------

		// Interior
		arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

		// Right
		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nx + globalIdx + BLOCK_X];
		}

		// Top
		if (threadIdx.y < numStenTop )
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
		}

		// Top Right
		if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx + BLOCK_X)];
		}

		if (tileBottom != 1)
		{
			// Bottom
			if (threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = boundaryBottom[threadIdx.y * nx + globalIdx];
			}

			// Bottom Right
			if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] =  boundaryBottom[threadIdx.y * nx + (globalIdx + BLOCK_X)];
			}
		}

		// Ensure the copy is complete
		__syncthreads();

		// ----------
		// Compute
		// ----------

		stenSet = threadIdx.y * nxLocal + threadIdx.x;
		weight = 0;

		for (int j = 0; j < numStenVert; j++) // Allow for the point we're actually at
		{
			temp = j * nxLocal;

			for (int i = 0; i < numStenHoriz; i++) // Allow for the point we're actually at
			{
				sum += weigthsLocal[weight] * arrayLocal[stenSet + temp + i];

				weight++;
			} 
		}

		// Ensure the compute is complete
		__syncthreads();

		// ----------
		// Copy back 
		// ----------

		if (tileBottom == 1)
		{
			if (threadIdx.x >= numStenLeft && threadIdx.y < BLOCK_Y - numStenBottom)
			{
				dataOutput[globalIdy * nx + globalIdx] = sum;
			}
		}
		else
		{
			if (threadIdx.x >= numStenLeft)
			{
				dataOutput[globalIdy * nx + globalIdx] = sum;
			}
		}
	}

	// -----------------------------
	// (0, nyTile / BLOCK_Y - 1) - Bottom Right
	// -----------------------------

	else if (blockIdx.x == nx / BLOCK_X - 1 && blockIdx.y == nyTile / BLOCK_Y - 1)
	{
		// ----------
		// Copy
		// ----------

		// Interior
		arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

		// Left 
		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (globalIdx - numStenLeft)];
		}

		// Top
		if (threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
		}

		// Top Left
		if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx - numStenLeft)];
		}

		if (tileBottom != 1)
		{
			// Bottom
			if (threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = boundaryBottom[threadIdx.y * nx + globalIdx];
			}

			// Bottom Left
			if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = boundaryBottom[threadIdx.y * nx + (globalIdx - numStenLeft)];
			}
		}

		// Ensure copying completed
		__syncthreads();

		// ----------
		// Compute
		// ----------

		stenSet = threadIdx.y * nxLocal + threadIdx.x;
		weight = 0;

		for (int j = 0; j < numStenVert; j++) // Allow for the point we're actually at
		{
			temp = j * nxLocal;

			for (int i = 0; i < numStenHoriz; i++) // Allow for the point we're actually at
			{
				sum += weigthsLocal[weight] * arrayLocal[stenSet + temp + i];

				weight++;
			} 
		}

		// Ensure the compute is complete
		__syncthreads();

		// ----------
		// Copy back 
		// ----------

		if (tileBottom == 1)
		{
			if (threadIdx.x < BLOCK_X - numStenRight && threadIdx.y < BLOCK_Y - numStenBottom)
			{
				dataOutput[globalIdy * nx + globalIdx] = sum;
			}
		}
		else
		{
			if (threadIdx.x < BLOCK_X - numStenRight)
			{
				dataOutput[globalIdy * nx + globalIdx] = sum;
			}
		}
	}

	// -----------------------------
	// (_, 0) - Top
	// -----------------------------

	else if (blockIdx.y == 0)
	{
		// ----------
		// Copy
		// ----------
	
		// Interior
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

		// Bottom
		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
		}

		// Bottom Right
		if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx + BLOCK_X)];
		}

		// Bottom Left
		if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx - numStenLeft)];
		}

		if (tileTop != 1)
		{
			// Top
			if (threadIdx.y < numStenTop)
			{
				arrayLocal[threadIdx.y * nxLocal + localIdx] = boundaryTop[threadIdx.y * nx + globalIdx];
			}

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
		}

		// Ensure copying completed
		__syncthreads();

		// ----------
		// Compute
		// ----------

		stenSet = threadIdx.y * nxLocal + threadIdx.x;
		weight = 0;

		for (int j = 0; j < numStenVert; j++) // Allow for the point we're actually at
		{
			temp = j * nxLocal;

			for (int i = 0; i < numStenHoriz; i++) // Allow for the point we're actually at
			{
				sum += weigthsLocal[weight] * arrayLocal[stenSet + temp + i];

				weight++;
			} 
		}

		// Ensure the compute is complete
		__syncthreads();

		// ----------
		// Copy back 
		// ----------

		if (tileTop == 1)
		{
			if (threadIdx.y >= numStenTop)
			{
				dataOutput[globalIdy * nx + globalIdx] = sum;
			}
		}
		else
		{
			dataOutput[globalIdy * nx + globalIdx] = sum;
		}
	}

	// -----------------------------
	// (_, nyTile / BLOCK_Y - 1) - Bottom
	// -----------------------------

	else if (blockIdx.y == nyTile / BLOCK_Y - 1)
	{
		// ----------
		// Copy
		// ----------
	
		// Interior
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

		// Top
		if (threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
		}

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

		if (tileBottom != 1)
		{
			// Bottom
			if (threadIdx.y < numStenBottom)
			{
				arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = boundaryBottom[threadIdx.y * nx + globalIdx];
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

		// Ensure copying completed
		__syncthreads();
		
		// ----------
		// Compute
		// ----------

		stenSet = threadIdx.y * nxLocal + threadIdx.x;
		weight = 0;

		for (int j = 0; j < numStenVert; j++) // Allow for the point we're actually at
		{
			temp = j * nxLocal;

			for (int i = 0; i < numStenHoriz; i++) // Allow for the point we're actually at
			{
				sum += weigthsLocal[weight] * arrayLocal[stenSet + temp + i];

				weight++;
			} 
		}

		// Ensure the compute is complete
		__syncthreads();

		// ----------
		// Copy back 
		// ----------

		if (tileBottom == 1)
		{
			if (threadIdx.y < BLOCK_Y - numStenBottom)
			{
				dataOutput[globalIdy * nx + globalIdx] = sum;
			}
		}
		else
		{
			dataOutput[globalIdy * nx + globalIdx] = sum;
		}	
	}

	// -----------------------------
	// (0, _) - Left
	// -----------------------------

	else if (blockIdx.x == 0)
	{
		// ----------
		// Copy
		// ----------
	
		// Interior
		arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

		// Right
		if (threadIdx.x < numStenRight)
		{
			arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nx + globalIdx + BLOCK_X];
		}

		// Top
		if (threadIdx.y < numStenTop )
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
		}

		// Top Right
		if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx + BLOCK_X)];
		}

		// Bottom
		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
		}

		// Bottom Right
		if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx + BLOCK_X)];
		}

		// Ensure copying completed
		__syncthreads();
		
		// ----------
		// Compute
		// ----------

		stenSet = threadIdx.y * nxLocal + threadIdx.x;
		weight = 0;

		for (int j = 0; j < numStenVert; j++) // Allow for the point we're actually at
		{
			temp = j * nxLocal;

			for (int i = 0; i < numStenHoriz; i++) // Allow for the point we're actually at
			{
				sum += weigthsLocal[weight] * arrayLocal[stenSet + temp + i];

				weight++;
			} 
		}

		// Ensure the compute is complete
		__syncthreads();

		// ----------
		// Copy back 
		// ----------

		if (threadIdx.x >= numStenLeft)
		{
			dataOutput[globalIdy * nx + globalIdx] = sum;
		}	
	}

	// -----------------------------
	// (nx / BLOCK_X - 1, _) - Right
	// -----------------------------

	else if (blockIdx.x == nx / BLOCK_X - 1)
	{
		// ----------
		// Copy
		// ----------
	
		// Interior
		arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

		// Left 
		if (threadIdx.x < numStenLeft)
		{
			arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (globalIdx - numStenLeft)];
		}

		// Top
		if (threadIdx.y < numStenTop )
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
		}

		// Top Left
		if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
		{
			arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx - numStenLeft)];
		}

		// Bottom
		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
		}

		// Bottom Left
		if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx - numStenLeft)];
		}

		// Ensure copying completed
		__syncthreads();
		
		// ----------
		// Compute
		// ----------

		stenSet = threadIdx.y * nxLocal + threadIdx.x;
		weight = 0;

		for (int j = 0; j < numStenVert; j++) // Allow for the point we're actually at
		{
			temp = j * nxLocal;

			for (int i = 0; i < numStenHoriz; i++) // Allow for the point we're actually at
			{
				sum += weigthsLocal[weight] * arrayLocal[stenSet + temp + i];

				weight++;
			} 
		}

		// Ensure the compute is complete
		__syncthreads();

		// ----------
		// Copy back 
		// ----------

		if (threadIdx.x < BLOCK_X - numStenLeft)
		{
			dataOutput[globalIdy * nx + globalIdx] = sum;
		}
	}

	// -----------------------------
	// Interior
	// -----------------------------

	else
	{
		// ----------
		// Copy
		// ----------
	
		// Interior
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

		// Top
		if (threadIdx.y < numStenTop )
		{
			arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
		}

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

		// Bottom
		if (threadIdx.y < numStenBottom)
		{
			arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
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

		// Ensure copying completed
		__syncthreads();
		
		// ----------
		// Compute
		// ----------

		stenSet = threadIdx.y * nxLocal + threadIdx.x;
		weight = 0;

		for (int j = 0; j < numStenVert; j++) // Allow for the point we're actually at
		{
			temp = j * nxLocal;

			for (int i = 0; i < numStenHoriz; i++) // Allow for the point we're actually at
			{
				sum += weigthsLocal[weight] * arrayLocal[stenSet + temp + i];

				weight++;
			} 
		}

		// Ensure the compute is complete
		__syncthreads();

		// ----------
		// Copy back 
		// ----------

		dataOutput[globalIdy * nx + globalIdx] = sum;
	}
}

// ---------------------------------------------------------------------
// Function to compute kernel
// ---------------------------------------------------------------------

/*! \fun void custenCompute2DXYnp
    \brief Function called by user to compute the stencil.
    \param pt_cuSten Pointer to cuSten data type which contains all the necessary input
	\param offload Boolean set by user to 1 if they wish to move the data back to the host after completing computation, 0 otherwise
*/

void custenCompute2DXYnp
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
		kernel2DXYnp<<<gridDim, blockDim, pt_cuSten->mem_shared, pt_cuSten->streams[0]>>>(
			pt_cuSten->dataOutput[tile], 

			pt_cuSten->dataInput[tile], 

			pt_cuSten->boundaryTop[tile], 
			pt_cuSten->boundaryBottom[tile], 

			pt_cuSten->weights, 

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
			pt_cuSten->nyTile,

			tileTop,
			tileBottom
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