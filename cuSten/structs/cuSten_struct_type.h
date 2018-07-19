// Andrew Gloster
// May 2018
// File detailing struct type used in cuSten library

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

	// Number of x points on a tile
	int nxTile;
	
	// Number of y points on a tile
 	int nyTile;

	// Number of points in the stencil
	int numSten;

	// Number of points to the left in the stencil
	int numStenLeft;
	
	// Number of points to the right in the stencil
	int numStenRight;

	// Number of points in the top of the stencil
	int numStenTop;
	
	// Number of points to the bottom in the stencil
	int numStenBottom;

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

    // Pointer for the device weights data
    double* weights;

	// Pointer to coefficients used
	double* coe;

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
