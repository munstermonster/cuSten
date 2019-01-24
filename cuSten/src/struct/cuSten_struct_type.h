// Andrew Gloster
// May 2018
// File detailing struct type used in cuSten library

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
// Define Header
// ---------------------------------------------------------------------

#ifndef COMPUTE_H
#define COMPUTE_H



// ---------------------------------------------------------------------
//  Programmer Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
// Define Struct
// ---------------------------------------------------------------------

typedef struct
{
    // Device number 
    int deviceNum;

    // Number of streams
    int numStreams;

    // Number of tiles
    int numTiles;

	// Number of x points on the device
	int nxDevice;
	
	// Number of y points on the device
 	int nyDevice;

	// Number of y points on a tile
 	int nyTile;

	// Number of points in the stencil (or total points when cross derivative)
	int numSten;

	// Number of points to the left in the stencil
	int numStenLeft;
	
	// Number of points to the right in the stencil
	int numStenRight;

	// Number of points in the top of the stencil
	int numStenTop;
	
	// Number of points to the bottom in the stencil
	int numStenBottom;

	// Number of points in a horizontal stencil
	int numStenHoriz;

	// Number of points in a vertical stencil
	int numStenVert;

	// Number of threads in x
	int BLOCK_X;

	// Number of threads in y
	int BLOCK_Y; 

	// Size of grid in x
	int xGrid;

	// Sixe of grid in y
	int yGrid;

	// Amount of shared memory required
	int mem_shared; 

	// Pointers for the input data
	double** dataInput;

	// Pointers for the ouput data
	double** dataOutput;

	// Pointers for the input data
	double** uVel;

	// Pointers for the ouput data
	double** vVel;

	// Pointer for the device weights data
	double* weights;

	// Pointer to coefficients used
	double* coe;

	// Coefficient for WENO - x 
	double coeDx;

	// Coefficient for WENO - y
	double coeDy;

	// Number of coefficients
	int numCoe;

	// Local points in shared memory in x
	int nxLocal;

	// Local points in shared memory in x
	int nyLocal;	

	// Boundary locations for top of tiles
	double** boundaryTop;

	// Boundary locations for bottom of tiles
	double** boundaryBottom;

	// Number of points in top boundary data
	int numBoundaryTop;

	// Number of points in bottom boundary data
	int numBoundaryBottom;

	// Streams to permute through
	cudaStream_t* streams;

	// Events for tracking
	cudaEvent_t* events;

	// Function Pointer
	double* devFunc;

} cuSten_t;

#endif
