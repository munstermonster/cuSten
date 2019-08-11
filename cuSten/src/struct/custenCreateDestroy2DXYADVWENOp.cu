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

/*! \file custenCreateDestroy2DXYADVWENOp.cu
    Functions to create and destroy the required struct for a 2D xy WENO
 	periodic calculation
*/

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------

#include <iostream>

// ---------------------------------------------------------------------
// Custom libraries and headers
// ---------------------------------------------------------------------

#include "cuSten_struct_type.h"
#include "cuSten_struct_functions.h"
#include "../util/util.h"

// ---------------------------------------------------------------------
// Function to create the struct
// ---------------------------------------------------------------------

/*! \fun void cuStenCreate2DXYWENOADVp
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
	\param dx Spacing of grid in x direction
	\param dy Spacing of grid in y direction
	\param u Pointer to u velocity data
	\param v Pointer to v velocity data
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
*/

void cuStenCreate2DXYWENOADVp
(
	cuSten_t* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	double dx,
	double dy,
	double* u,
	double* v,
	double* dataOutput,
	double* dataInput
) 
{
	// Buffer used for error checking
	char msgStringBuffer[1024];

	// Set the device number associated with the struct
  	pt_cuSten->deviceNum = deviceNum;

  	// Set the number of streams
  	pt_cuSten->numStreams = 6;

  	// Set the number of tiles
  	pt_cuSten->numTiles = numTiles;

  	// Set the number points in x on the device
  	pt_cuSten->nx = nx;

  	// Set the number points in y on the device
  	pt_cuSten->ny = ny;

  	// Number of threads in x on the device
	pt_cuSten->BLOCK_X = BLOCK_X;

  	// Number of threads in y on the device
	pt_cuSten->BLOCK_Y = BLOCK_Y;

	// Encode spacings
	pt_cuSten->coeDx = 1.0 / dx;
	pt_cuSten->coeDy = 1.0 / dy;

	// We require i - 3 throught to i + 3 for WENO
	// Set number of points to the left in the stencil
	pt_cuSten->numStenLeft = 3;

	// Set number of points to the right in the stencil
	pt_cuSten->numStenRight = 3;

	// Set number of points in the top the stencil
	pt_cuSten->numStenTop = 3;

	// Set number of points in the bottom of the stencil
	pt_cuSten->numStenBottom = 3;

	// Set size of stencil
	int numStenHoriz = 7;
	int numStenVert = 7;
	
	// Set current active compute device
	cudaSetDevice(pt_cuSten->deviceNum);
	sprintf(msgStringBuffer, "Setting current device to GPU %d", pt_cuSten->deviceNum);
	checkError(msgStringBuffer);	

	// Create memeory for the streams
	pt_cuSten->streams = (cudaStream_t*)malloc(pt_cuSten->numStreams * sizeof(cudaStream_t*));

	// Create the streams
	for (int st = 0; st < pt_cuSten->numStreams; st++)
	{
		cudaStreamCreate(&pt_cuSten->streams[st]);
		sprintf(msgStringBuffer, "Creating stream %d on GPU %d", st, pt_cuSten->deviceNum);
		checkError(msgStringBuffer);	
	}

	// Create memeory for the events
	pt_cuSten->events = (cudaEvent_t*)malloc(2 * sizeof(cudaEvent_t*));

	// Create the events
	for (int ev = 0; ev < 2; ev++)
	{
		cudaEventCreate(&pt_cuSten->events[ev]);
		sprintf(msgStringBuffer, "Creating event %d on GPU %d", ev, pt_cuSten->deviceNum);
		checkError(msgStringBuffer);
	}

	// Set number of points in the stencil
	pt_cuSten->numSten = numStenHoriz * numStenVert;

	// Set local block array sizes - x direction
	pt_cuSten->nxLocal = pt_cuSten->BLOCK_X + pt_cuSten->numStenLeft + pt_cuSten->numStenRight;

	// Set loacl block array sizes - y direction
	pt_cuSten->nyLocal = pt_cuSten->BLOCK_Y + pt_cuSten->numStenTop + pt_cuSten->numStenBottom;

	// Set the amount of shared memory required
	pt_cuSten->mem_shared = (pt_cuSten->nxLocal * pt_cuSten->nyLocal) * sizeof(double) + pt_cuSten->numSten * sizeof(double);

	// Find number of points per tile
	pt_cuSten->nyTile = pt_cuSten->ny / pt_cuSten->numTiles;	

	// Set the grid up
    pt_cuSten->xGrid = (pt_cuSten->nx % pt_cuSten->BLOCK_X == 0) ? (pt_cuSten->nx / pt_cuSten->BLOCK_X) : (pt_cuSten->nx / pt_cuSten->BLOCK_X + 1);
    pt_cuSten->yGrid = (pt_cuSten->nyTile % pt_cuSten->BLOCK_Y == 0) ? (pt_cuSten->nyTile / pt_cuSten->BLOCK_Y) : (pt_cuSten->nyTile / pt_cuSten->BLOCK_Y + 1);

	// Allocate the pointers for each input tile
	pt_cuSten->dataInput = (double**)malloc(pt_cuSten->numTiles * sizeof(double));

	// Allocate the pointers for each output tile
	pt_cuSten->dataOutput = (double**)malloc(pt_cuSten->numTiles * sizeof(double));

	// Allocate the pointers for each input x velocity tile
	pt_cuSten->uVel = (double**)malloc(pt_cuSten->numTiles * sizeof(double));

	// Allocate the pointers for each input v velocity tile
	pt_cuSten->vVel = (double**)malloc(pt_cuSten->numTiles * sizeof(double));

	// // Tile offset index
	int offset = pt_cuSten->nx * pt_cuSten->nyTile;

	// // Match the pointers to the data
	for (int tile = 0; tile < pt_cuSten->numTiles; tile++)
	{	
		// Set the input data
		pt_cuSten->dataInput[tile] = &dataInput[tile * offset];

		// Set the output data
		pt_cuSten->dataOutput[tile] = &dataOutput[tile * offset];

		// Set the u velocity tiles
		pt_cuSten->uVel[tile] = &u[tile * offset];

		// Set the v velocity tiles
		pt_cuSten->vVel[tile] = &v[tile * offset];
	}

	// Create cases depending on what tile numbers - Periodic
	// 1 tile
	// 2 tiles
	// 3 or greater

	// Allocate top boundary memory
	pt_cuSten->boundaryTop = (double**)malloc(pt_cuSten->numTiles * sizeof(double));

	// Allocate bottom boundary memory
	pt_cuSten->boundaryBottom = (double**)malloc(pt_cuSten->numTiles * sizeof(double));

	switch(pt_cuSten->numTiles)
	{
		// One tile only requires single top and bottom to be set
		case 1:
			pt_cuSten->boundaryTop[0] = &dataInput[(pt_cuSten->ny - pt_cuSten->numStenTop) * pt_cuSten->nx];
			pt_cuSten->boundaryBottom[0] = &dataInput[0]; 

			break;

		// Two tiles requires a special case of only setting two tiles
		case 2:
			pt_cuSten->boundaryTop[0] = &dataInput[(pt_cuSten->ny - pt_cuSten->numStenTop) * pt_cuSten->nx];
			pt_cuSten->boundaryBottom[0] = &dataInput[pt_cuSten->nyTile * pt_cuSten->nx];

			pt_cuSten->boundaryTop[1] = &dataInput[(pt_cuSten->nyTile - pt_cuSten->numStenTop) * pt_cuSten->nx];
			pt_cuSten->boundaryBottom[1] = &dataInput[0];

			break;

		// Default case has interiors, so set the top tile, then loop over interior, then set the bottom tile
		default:
			pt_cuSten->boundaryTop[0] = &dataInput[(pt_cuSten->ny - pt_cuSten->numStenTop) * pt_cuSten->nx];
			pt_cuSten->boundaryBottom[0] = &dataInput[pt_cuSten->nyTile * pt_cuSten->nx];

			for (int tile = 1; tile < pt_cuSten->numTiles - 1; tile++)
			{
				pt_cuSten->boundaryTop[tile] = &dataInput[(pt_cuSten->nyTile * tile - pt_cuSten->numStenTop) * pt_cuSten->nx];
				pt_cuSten->boundaryBottom[tile] = &dataInput[(pt_cuSten->nyTile * (tile + 1)) * pt_cuSten->nx];
			}

			pt_cuSten->boundaryTop[pt_cuSten->numTiles - 1] = &dataInput[(pt_cuSten->nyTile * (pt_cuSten->numTiles - 1) - pt_cuSten->numStenTop) * pt_cuSten->nx];
			pt_cuSten->boundaryBottom[pt_cuSten->numTiles - 1] = &dataInput[0];

			break;
	}

	// Number of points in top boundary data
	pt_cuSten->numBoundaryTop = pt_cuSten->numStenTop * pt_cuSten->nx;

	// Number of points in bottom boundary data
	pt_cuSten->numBoundaryBottom = pt_cuSten->numStenBottom * pt_cuSten->nx;

	// Number of points in a horizontal stencil
	pt_cuSten->numStenHoriz = numStenHoriz;

	// Number of points in a vertical stencil
	pt_cuSten->numStenVert = numStenVert;
}

// ---------------------------------------------------------------------
// Swap pointers
// ---------------------------------------------------------------------

/*! \fun void cuStenSwap2DXYWENOADVp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void cuStenSwap2DXYWENOADVp(
	cuSten_t* pt_cuSten,

	double* dataInput
) 
{
	for (int tile = 0; tile < pt_cuSten->numTiles; tile++)
	{	
		// Swap the input and output data
		std::swap(pt_cuSten->dataInput[tile], pt_cuSten->dataOutput[tile]);

		// Update the boundary data
		switch(pt_cuSten->numTiles)
		{
			// One tile only requires single top and bottom to be set
			case 1:
				pt_cuSten->boundaryTop[0] = &dataInput[(pt_cuSten->ny - pt_cuSten->numStenTop) * pt_cuSten->nx];
				pt_cuSten->boundaryBottom[0] = &dataInput[0]; 

				break;

			// Two tiles requires a special case of only setting two tiles
			case 2:
				pt_cuSten->boundaryTop[0] = &dataInput[(pt_cuSten->ny - pt_cuSten->numStenTop) * pt_cuSten->nx];
				pt_cuSten->boundaryBottom[0] = &dataInput[pt_cuSten->nyTile * pt_cuSten->nx];

				pt_cuSten->boundaryTop[1] = &dataInput[(pt_cuSten->nyTile - pt_cuSten->numStenTop) * pt_cuSten->nx];
				pt_cuSten->boundaryBottom[1] = &dataInput[0];

				break;

			// Default case has interiors, so set the top tile, then loop over interior, then set the bottom tile
			default:
				pt_cuSten->boundaryTop[0] = &dataInput[(pt_cuSten->ny - pt_cuSten->numStenTop) * pt_cuSten->nx];
				pt_cuSten->boundaryBottom[0] = &dataInput[pt_cuSten->nyTile * pt_cuSten->nx];

				for (int tile = 1; tile < pt_cuSten->numTiles - 1; tile++)
				{
					pt_cuSten->boundaryTop[tile] = &dataInput[(pt_cuSten->nyTile * tile - pt_cuSten->numStenTop) * pt_cuSten->nx];
					pt_cuSten->boundaryBottom[tile] = &dataInput[(pt_cuSten->nyTile * (tile + 1)) * pt_cuSten->nx];
				}

				pt_cuSten->boundaryTop[pt_cuSten->numTiles - 1] = &dataInput[(pt_cuSten->nyTile * (pt_cuSten->numTiles - 1) - pt_cuSten->numStenTop) * pt_cuSten->nx];
				pt_cuSten->boundaryBottom[pt_cuSten->numTiles - 1] = &dataInput[0];

				break;
		}
	}
}

// ---------------------------------------------------------------------
// Function to destroy the struct
// ---------------------------------------------------------------------

/*! \fun void cuStenDestroy2DXYWENOADVp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void cuStenDestroy2DXYWENOADVp
(
	cuSten_t* pt_cuSten
) 
{
	// Buffer used for error checking
	char msgStringBuffer[1024];

	// Set current active compute device
	cudaSetDevice(pt_cuSten->deviceNum);
	sprintf(msgStringBuffer, "Setting current device to GPU %d", pt_cuSten->deviceNum);
	checkError(msgStringBuffer);	


	// Destroy the streams
	for (int st = 0; st < pt_cuSten->numStreams; st++)
	{
		cudaStreamDestroy(pt_cuSten->streams[st]);
		sprintf(msgStringBuffer, "Destroying stream %d on GPU %d", st, pt_cuSten->deviceNum);
		checkError(msgStringBuffer);	
	}

	// Free the main memory
	free(pt_cuSten->streams);

	// // Create the events
	for (int ev = 0; ev < 2; ev++)
	{
		cudaEventDestroy(pt_cuSten->events[ev]);
		sprintf(msgStringBuffer, "Destroying event %d on GPU %d", ev, pt_cuSten->deviceNum);
		checkError(msgStringBuffer);
	}

	// Free the main memory
	free(pt_cuSten->events);

	// Free the pointers for each input tile
	free(pt_cuSten->dataInput);

	// Free the pointers for each output tile
	free(pt_cuSten->dataOutput);

	// Free the top boundary tile pointers
	free(pt_cuSten->boundaryTop);

	// Free the bottom boundary tile pointers
	free(pt_cuSten->boundaryBottom);
}


// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------