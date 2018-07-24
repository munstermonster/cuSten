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

#ifndef STRUCT_FUN_H
#define STRUCT_FUN_H

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  Programmer Libraries and Headers
// ---------------------------------------------------------------------

#include "../DeviceFunctions.h"

// ---------------------------------------------------------------------
//  Header file functions
// ---------------------------------------------------------------------

// ----------------------------------------
// 2D x direction non periodic
// ----------------------------------------

// Create the struct for a 2D x direction non periodic stencil
void custenCreate2DXnp(
	cuSten_t* pt_cuSten,		// Pointer to the user created struct

	int deviceNum,				// GPU device number (important with multi gpu systems)

	int numTiles,				// Number of tiles to use

	int nxDevice,				// Total number of points in x that will be computed on GPU
	int nyDevice,				// Total number of points in y that will be computed on GPU

	int BLOCK_X,				// Size of thread block in x
	int BLOCK_Y,				// Sixe of thread block in y

	double* dataNew,			// Output data
	double* dataOld,			// Input data
	double* weights,			// Stencil weights

	int numSten,				// Size of stencil
	int numStenLeft,			// Number of points to left in stencil
	int numStenRight			// Number of points to right in stencil
);

// Destroy the struct for a 2D x direction non periodic stencil
void custenDestroy2DXnp(
	cuSten_t* pt_cuSten   		// Pointer to user struct
); 

// ----------------------------------------
// 2D x direction periodic
// ----------------------------------------

// Create the struct for a 2D x direction non periodic stencil
void custenCreate2DXp(
	cuSten_t* pt_cuSten,		// Pointer to the user created struct

	int deviceNum,				// GPU device number (important with multi gpu systems)

	int numTiles,				// Number of tiles to use

	int nxDevice,				// Total number of points in x that will be computed on GPU
	int nyDevice,				// Total number of points in y that will be computed on GPU

	int BLOCK_X,				// Size of thread block in x
	int BLOCK_Y,				// Sixe of thread block in y

	double* dataNew,			// Output data
	double* dataOld,			// Input data
	double* weights,			// Stencil weights

	int numSten,				// Size of stencil
	int numStenLeft,			// Number of points to left in stencil
	int numStenRight			// Number of points to right in stencil
);

// Destroy the struct for a 2D x direction non periodic stencil
void custenDestroy2DXp(
	cuSten_t* pt_cuSten   		// Pointer to user struct
); 

// ----------------------------------------
// 2D x direction with non periodic user function
// ----------------------------------------

// Create the struct for a 2D x direction non periodic stencil
void custenCreate2DXnpFun(
	cuSten_t* pt_cuSten,		

	int deviceNum,

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
);

// Destroy the struct for a 2D x direction non periodic stencil with user function
void custenDestroy2DXnpFun(
	cuSten_t* pt_cuSten			// Pointer to user struct
);

// ----------------------------------------
// 2D x direction with periodic user function
// ----------------------------------------

// Create the struct for a 2D x direction non periodic stencil
void custenCreate2DXpFun(
	cuSten_t* pt_cuSten,

	int deviceNum,

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
);

// Destroy the struct for a 2D x direction non periodic stencil with user function
void custenDestroy2DXpFun(
	cuSten_t* pt_cuSten			// Pointer to user struct
);

// ----------------------------------------
// 2D y direction periodic
// ----------------------------------------

// Create the struct for a 2D x direction non periodic stencil
void custenCreate2DYp(
	cuSten_t* pt_cuSten,		// Pointer to the user created struct

	int deviceNum,				// GPU device number (important with multi gpu systems)

	int numTiles,				// Number of tiles to use

	int nxDevice,				// Total number of points in x that will be computed on GPU
	int nyDevice,				// Total number of points in y that will be computed on GPU

	int BLOCK_X,				// Size of thread block in x
	int BLOCK_Y,				// Sixe of thread block in y

	double* dataNew,			// Output data
	double* dataOld,			// Input data
	double* weights,			// Stencil weights

	int numSten,				// Size of stencil
	int numStenTop,				// Number of points in top of stencil
	int numStenBottom			// Number of points in bottom of stencil
);

// Destroy the struct for a 2D x direction non periodic stencil
void custenDestroy2DYp(
	cuSten_t* pt_cuSten   		// Pointer to user struct
); 

// ----------------------------------------
// 2D y direction with periodic user function
// ----------------------------------------

// Create the struct for a 2D y direction periodic with user function
void custenCreate2DYpFun(
	cuSten_t* pt_cuSten,

	int deviceNum,

	int numTiles,

	int nxDevice,
	int nyDevice,

	int BLOCK_X,
	int BLOCK_Y,

	double* dataNew,
	double* dataOld,
	double* coe,

	int numSten,
	int numStenTop,
	int numStenBottom,

	int numCoe,

	double* func	
);

// Swap pointers
void custenSwap2DYpFun(
	cuSten_t* pt_cuSten,

	double* dataOld
);

// Destroy the struct for a 2D y direction periodic stencil with user function
void custenDestroy2DYpFun(
	cuSten_t* pt_cuSten
);

// ----------------------------------------
// 2D y direction with non periodic
// ----------------------------------------

// Function to create the struct for a 2D y direction non periodic
void custenCreate2DYnp(
	cuSten_t* pt_cuSten,		// Pointer to the compute struct type

	int deviceNum,				// Device on which to compute this stencil

	int numTiles,				// Number of tiles to divide the data on the device into

	int nxDevice,				// Number of points in x on the device
	int nyDevice,				// Number of points in y on the device

	int BLOCK_X,				// Number of threads to use in x
	int BLOCK_Y,				// Number of threads to use in y

	double* dataNew,			// Output data
	double* dataOld,			// Input data
	double* weights,			// Arracy containing the weights

	int numSten,				// Number of points in a stencil
	int numStenTop,				// Number of points in the top of the stencil
	int numStenBottom			// Number of points in the bottom of the stencil
);

// Function to destroy the struct for a 2D y direction non periodic
void custenDestroy2DYnp(
	cuSten_t* pt_cuSten			// Pointer to the compute struct type
);

// ---------------------------------------------------------------------
// End of header file functions
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// End of definition
// ---------------------------------------------------------------------

#endif

// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------
