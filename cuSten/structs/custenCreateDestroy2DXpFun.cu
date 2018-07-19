// Andrew Gloster
// May 2018
// Functions to create and destroy the required struct for a 2D x direction
// non periodic calculation

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
// Function to create the struct for a 2D x direction non periodic
// ---------------------------------------------------------------------

void custenCreate2DXpFun(
	cuSten_t* pt_cuSten,

	int deviceNum,

	int numStreams,

	int numTiles,

	int nxDevice,
	int nyDevice,

	int BLOCK_X,
	int BLOCK_Y,

	double* dataNew,
	double* dataOld,
	double* coe,

	int numSten,
	int numStenLeft,
	int numStenRight,

	int numCoe,

	double* func
) 
{
	// Buffer used for error checking
	char msgStringBuffer[1024];

	// Set the device number associated with the struct
  	pt_cuSten->deviceNum = deviceNum;

  	// Set the number of streams
  	pt_cuSten->numStreams = numStreams;

  	// Set the number of tiles
  	pt_cuSten->numTiles = numTiles;

  	// Set the number points in x on the device
  	pt_cuSten->nxDevice = nxDevice;

  	// Set the number points in y on the device
  	pt_cuSten->nyDevice = nyDevice;

  	// Number of threads in x on the device
	pt_cuSten->BLOCK_X = BLOCK_X;

  	// Number of threads in y on the device
	pt_cuSten->BLOCK_Y = BLOCK_Y;

	// Set current active compute device
	cudaSetDevice(pt_cuSten->deviceNum);
	sprintf(msgStringBuffer, "Setting current device to GPU %d", pt_cuSten->deviceNum);
	checkError(msgStringBuffer);	

	// Create memeory for the streams
	pt_cuSten->streams = (cudaStream_t*)malloc(numStreams * sizeof(cudaStream_t*));

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
	pt_cuSten->numSten = numSten;

	// Set number of points to the left in the stencil
	pt_cuSten->numStenLeft = numStenLeft;

	// Set number of points to the right in the stencil
	pt_cuSten->numStenRight = numStenRight;

	// Set the device coefficients pointer
	pt_cuSten->coe = coe;

	// Set number of coefficients
	pt_cuSten->numCoe = numCoe;

	// Local memory grid sizes
	pt_cuSten->nxLocal = pt_cuSten->BLOCK_X;
	pt_cuSten->nyLocal = pt_cuSten->BLOCK_Y + pt_cuSten->numStenTop + pt_cuSten->numStenBottom;

	// Set the amount of shared memory required
	pt_cuSten->mem_shared = pt_cuSten->nxLocal * pt_cuSten->nyLocal * sizeof(double) + numCoe * sizeof(double);

	// Find number of points per tile
	pt_cuSten->nxTile = pt_cuSten->nxDevice;

	// Find number of points per tile
	pt_cuSten->nyTile = pt_cuSten->nyDevice / pt_cuSten->numTiles;	

	// Set the grid up
    pt_cuSten->xGrid = (pt_cuSten->nxTile % pt_cuSten->BLOCK_X == 0) ? (pt_cuSten->nxTile / pt_cuSten->BLOCK_X) : (pt_cuSten->nxTile / pt_cuSten->BLOCK_X + 1);
    pt_cuSten->yGrid = (pt_cuSten->nyTile % pt_cuSten->BLOCK_Y == 0) ? (pt_cuSten->nyTile / pt_cuSten->BLOCK_Y) : (pt_cuSten->nyTile / pt_cuSten->BLOCK_Y + 1);

	// Allocate the pointers for each input tile
	pt_cuSten->dataInput = (double**)malloc(pt_cuSten->numTiles * sizeof(double));

	// Allocate the pointers for each output tile
	pt_cuSten->dataOutput = (double**)malloc(pt_cuSten->numTiles * sizeof(double));

	// // Tile offset index
	int offset = pt_cuSten->nxTile * pt_cuSten->nyTile;

	// Match the pointers to the data
	for (int tile = 0; tile < pt_cuSten->numTiles; tile++)
	{	
		// Set the input data
		pt_cuSten->dataInput[tile] = &dataOld[tile * offset];

		// Set the output data
		pt_cuSten->dataOutput[tile] = &dataNew[tile * offset];
	}



	pt_cuSten->devFunc = func;

}

// ---------------------------------------------------------------------
// Function to destroy the struct for a 2D x direction non periodic
// ---------------------------------------------------------------------

void custenDestroy2DXpFun(
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
}


// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------