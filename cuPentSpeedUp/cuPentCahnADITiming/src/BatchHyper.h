// Andrew Gloster
// June 2018
// Header file for functions used in batch 1D hyperdiffusion code

// ---------------------------------------------------------------------
// Define Header
// ---------------------------------------------------------------------

#ifndef HYPERFUN_H
#define HYPERFUN_H

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  Programmer Libraries and Headers
// ---------------------------------------------------------------------


// ---------------------------------------------------------------------
//  Header file functions
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
// Set the vectors for the LHS - inital single solve - interleaved
// ---------------------------------------------------------------------

__global__ void setMultiLHS
(
	double* dsMulti, 
	double* dlMulti, 
	double* diagMulti, 
	double* duMulti, 
	double* dwMulti, 

	double a,
	double b,
	double c,
	double d,
	double e, 

	int nEq,
	int nSolve
);

// ---------------------------------------------------------------------
// Recover the omega matrix
// ---------------------------------------------------------------------

void findOmega
(
	double* omega,

	double* inv1,
	double* inv2,

	const double a,
	const double b,
	const double c,
	const double d,
	const double e,

	const int nx
);

// ---------------------------------------------------------------------
// Invert Cyclic Matrix
// ---------------------------------------------------------------------

void cyclicInv
(
	double* ds,
	double* dl,
	double* diag,
	double* du,
	double* dw,

	double* inv1Multi,
	double* inv2Multi,

	double* omega,

	double* data,

	double a,
	double b, 
	double d,
	double e,

	int BLOCK_INV,
	int BLOCK_X,
	int BLOCK_Y,

	int size,
	int nx
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