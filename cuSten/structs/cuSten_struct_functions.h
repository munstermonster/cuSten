// Andrew Gloster
// May 2018
// File detailing struct type used in cuSten library

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

	int numStreams,				// Number of streams to create

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

	int numStreams,				// Number of streams to create

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

	int numStreams,				// Number of streams to create

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

// Create the struct for a 2D x direction non periodic stencil
void custenCreate2DYpFun(
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
	int numStenTop,
	int numStenBottom,

	int numCoe,

	double* func	
);

// Swap pointers when timestepping
void custenSwap2DYpFun(
	cuSten_t* pt_cuSten,

	double* dataOld
);

// Destroy the struct for a 2D x direction non periodic stencil with user function
void custenDestroy2DYpFun(
	cuSten_t* pt_cuSten
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
