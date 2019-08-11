// Andrew Gloster
// September 2018
// Functions for use in 2D Cahn-Hilliard

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------

#include <stdlib.h>
#include <stdio.h>

// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------

#include "BatchHyper.h"
#include "cuPentBatch.h"
#include "../../../cuSten/cuSten.h"

// ---------------------------------------------------------------------
// Set the vectors for the LHS - inital single solve - interleaved
// ---------------------------------------------------------------------

static void setSingleLHS
(
	double* dsSingle, 
	double* dlSingle, 
	double* diagSingle, 
	double* duSingle, 
	double* dwSingle, 

	double a,
	double b,
	double c,
	double d,
	double e, 

	int nx
)
{
	for(int i = 0; i < nx; i++)
	{
		if (i > 1)
		{
			dsSingle[i] = a;
		}
		if (i > 0)
		{
			dlSingle[i] = b;
		}

		diagSingle[i] = c;

		if (i < nx - 1)
		{
			duSingle[i] = d;
		}

		if (i < nx - 2)
		{
			dwSingle[i] = e;
		}
	}
}

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

	int nx,
	int batchCount
)
{
	// Matrix index
    int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

	// Index access
	int index = globalIdy * batchCount + globalIdx;

	if (globalIdx < batchCount && globalIdy < nx)
	{

		dsMulti[index] = a;

		dlMulti[index] = b;
	
		diagMulti[index] = c;

		duMulti[index] = d;
	
		dwMulti[index] = e;
		
	}
}

// ---------------------------------------------------------------------
// Set up the 1st part of the f vector for initial inversions
// ---------------------------------------------------------------------

static void setInv1
(
	double* inv1,

	double a,
	double d,
	double e,

	int nx 
)
{
	for(int i = 0; i < nx; i++)
	{
		if (i == 0)
		{
			inv1[i] = a;
		}
		else if (i == 1)
		{
			inv1[i] = 0.0;
		}
		else if (i == nx - 2)
		{
			inv1[i] = e;
		}
		else if (i == nx - 1)
		{
			inv1[i] = d;
		}
		else
		{
			inv1[i] = 0.0;
		}
	}
}

// ---------------------------------------------------------------------
// Set up the 2nd part of the f vector for initial inversions
// ---------------------------------------------------------------------

static void setInv2
(
	double* inv2,

	double a,
	double b,
	double e,

	int nx 
)
{
	for(int i = 0; i < nx; i++)
	{
		if (i == 0)
		{
			inv2[i] = b;
		}
		else if (i == 1)
		{
			inv2[i] = a;
		}
		else if (i == nx - 2)
		{
			inv2[i] = 0.0;
		}
		else if (i == nx - 1)
		{
			inv2[i] = e;
		}
		else
		{
			inv2[i] = 0.0;
		}
	}
}

// ---------------------------------------------------------------------
// Solve for the final two points in the system
// ---------------------------------------------------------------------

__global__ static void solveEnd
(
	double* data,

	const double a,
	const double b,
	const double d,
	const double e,

	const double omega_11,
	const double omega_12,
	const double omega_21,
	const double omega_22,

	const int nx,
	const int nBatch
)
{
	// Matrix index
	int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;

	// Last two vectors
	double newNx2; 
	double newNx1;

	// Compute lambda = d^~ - transpose(g) * inverse(E) * d_hat
	newNx2 = data[(nx - 2) * nBatch + globalIdx] - (e * data[globalIdx] + a * data[(nx - 4) * nBatch + globalIdx] + b * data[(nx - 3) * nBatch + globalIdx]);
	newNx1 = data[(nx - 1) * nBatch + globalIdx] - (d * data[globalIdx] + e * data[nBatch + globalIdx] + a * data[(nx - 3) * nBatch + globalIdx]);

	// Compute x^~ = omega * lambda
	data[(nx - 2) * nBatch + globalIdx] = omega_11 * newNx2 + omega_12 * newNx1;
	data[(nx - 1) * nBatch + globalIdx] = omega_21 * newNx2 + omega_22 * newNx1;
}

// ---------------------------------------------------------------------
// Reconstruct the full solution
// ---------------------------------------------------------------------

__global__ static void solveFull
(
	double* data,

	double* inv1,
	double* inv2,

	const int nx,
	const int nBatch
)
{
	// Matrix index
	int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

	// Set values to last two entries in array
	double oldNx2 = data[(nx - 2) * nBatch + globalIdx]; // Two points from end
	double oldNx1 = data[(nx - 1) * nBatch + globalIdx]; // One point from end

	// Set index being computed
	int index = globalIdy * nBatch + globalIdx;

	if (globalIdy < nx - 2)
	{
		data[index] = data[index] - (inv1[index] * oldNx2 + inv2[index] * oldNx1);
	}
}

// ---------------------------------------------------------------------
// Factorise the matrix
// ---------------------------------------------------------------------

static void pentFactor
(
	double* ds,  	// Array containing the lower diagonal, 2 away from the main diagonal. First two elements are 0. Stored in interleaved format.
	double* dl,  	// Array containing the lower diagonal, 1 away from the main diagonal. First elements is 0. Stored in interleaved format.
	double* d, 	 	// Array containing the main diagonal. Stored in interleaved format.
	double* du,	 	// Array containing the upper diagonal, 1 away from the main diagonal. Last element is 0. Stored in interleaved format.
	double* dw,  	// Array containing the upper diagonal, 2 awy from the main diagonal. Last 2 elements are 0. Stored in interleaved format.

	int nx  		// Size of the linear systems, number of unknowns
)
{
	// Indices used to store relative indexes
	int rowCurrent;
	int rowPrevious;
	int rowSecondPrevious;

	// Starting index
    rowCurrent = 0;

	// First Row
	d[rowCurrent] = d[rowCurrent];
	du[rowCurrent] = du[rowCurrent] / d[rowCurrent];
	dw[rowCurrent] = dw[rowCurrent] / d[rowCurrent];

	// Second row index
	rowPrevious = rowCurrent;
	rowCurrent += 1;

	// Second row
	dl[rowCurrent] = dl[rowCurrent];

	d[rowCurrent] = d[rowCurrent] - dl[rowCurrent] * du[rowPrevious];

	du[rowCurrent] = (du[rowCurrent] - dl[rowCurrent] * dw[rowPrevious]) / d[rowCurrent];

	dw[rowCurrent] = dw[rowCurrent] / d[rowCurrent];

	// Interior rows - Note 0 indexing
	for (int i = 2; i < nx - 2; i++)
	{
		rowSecondPrevious = rowCurrent - 1; 
		rowPrevious = rowCurrent;
		rowCurrent += 1;

		dl[rowCurrent] = dl[rowCurrent] - ds[rowCurrent] * du[rowSecondPrevious];
		
		d[rowCurrent] = d[rowCurrent] - ds[rowCurrent] * dw[rowSecondPrevious] - dl[rowCurrent] * du[rowPrevious];

		dw[rowCurrent] = dw[rowCurrent] / d[rowCurrent];

		du[rowCurrent] = (du[rowCurrent] - dl[rowCurrent] * dw[rowPrevious]) / d[rowCurrent];
	}

	// Second last row indexes
	rowSecondPrevious = rowCurrent - 1; 
	rowPrevious = rowCurrent;
	rowCurrent += 1;

	// Second last row
	dl[rowCurrent] = dl[rowCurrent] - ds[rowCurrent] * du[rowSecondPrevious];
	d[rowCurrent] = d[rowCurrent] - ds[rowCurrent] * dw[rowSecondPrevious] - dl[rowCurrent] * du[rowPrevious];
	du[rowCurrent] = (du[rowCurrent] - dl[rowCurrent] * dw[rowPrevious]) / d[rowCurrent];

	// Last row indexes
	rowSecondPrevious = rowCurrent - 1; 
	rowPrevious = rowCurrent;
	rowCurrent += 1;

	// Last row
	dl[rowCurrent] = dl[rowCurrent] - ds[rowCurrent] * du[rowSecondPrevious];
	d[rowCurrent] = d[rowCurrent] - ds[rowCurrent] * dw[rowSecondPrevious] - dl[rowCurrent] * du[rowPrevious];
}

// ---------------------------------------------------------------------
// Solve the matrix
// ---------------------------------------------------------------------

static void pentSolve
(
	double* ds, 	// Array containing updated ds after using pentFactorBatch
	double* dl,		// Array containing updated ds after using pentFactorBatch
	double* d,		// Array containing updated ds after using pentFactorBatch
	double* du,		// Array containing updated ds after using pentFactorBatch
	double* dw,		// Array containing updated ds after using pentFactorBatch
	
	double* b,		// Dense array of RHS stored in interleaved format

	int nx  		// Size of the linear systems, number of unknowns
)
{

	// Indices used to store relative indexes
	int rowCurrent;
	int rowPrevious;
	int rowSecondPrevious;

	int rowAhead;
	int rowSecondAhead;

	// Starting index
    rowCurrent = 0;

	// --------------------------
	// Forward Substitution
	// --------------------------

	// First Row
	b[rowCurrent] = b[rowCurrent] / d[rowCurrent];

	// Second row index
	rowPrevious = rowCurrent;
	rowCurrent += 1;

	// Second row
	b[rowCurrent] = (b[rowCurrent] - dl[rowCurrent] * b[rowPrevious]) / d[rowCurrent];

	// Interior rows - Note 0 indexing
	for (int i = 2; i < nx; i++)
	{
		rowSecondPrevious = rowCurrent - 1; 
		rowPrevious = rowCurrent;
		rowCurrent += 1;

		b[rowCurrent] = (b[rowCurrent] - ds[rowCurrent] * b[rowSecondPrevious] - dl[rowCurrent] * b[rowPrevious]) / d[rowCurrent];	
	}

	// --------------------------
	// Backward Substitution
	// --------------------------

	// Last row
	b[rowCurrent] = b[rowCurrent];

	// Second last row index
	rowAhead = rowCurrent;
	rowCurrent -= 1;

	// Second last row
	b[rowCurrent] = b[rowCurrent] - du[rowCurrent] * b[rowAhead];

	// Interior points - Note row indexing
	for (int i = nx - 3; i >= 0; i -= 1)
	{
		rowSecondAhead = rowCurrent + 1;
		rowAhead = rowCurrent;
		rowCurrent -= 1;

		b[rowCurrent] = b[rowCurrent] - du[rowCurrent] * b[rowAhead] - dw[rowCurrent] * b[rowSecondAhead];
	}
	
}

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
)
{
	// Set the actual size, -2 
	int size = nx - 2;

	// --------------------------
	// Malloc the necessary data for LHS and set up
	// --------------------------

	double* dsSingle = (double*)malloc(size * sizeof(double));
	if (dsSingle == NULL)
	{
		printf("%s \n", "Failed to malloc dsSingle");
	}

	double* dlSingle = (double*)malloc(size * sizeof(double));
	if (dlSingle == NULL)
	{
		printf("%s \n", "Failed to malloc dlSingle");
	}   

	double* diagSingle = (double*)malloc(size * sizeof(double));
	if (diagSingle == NULL)
	{
		printf("%s \n", "Failed to malloc diagSingle");
	}

	double* duSingle = (double*)malloc(size * sizeof(double));
	if (duSingle == NULL)
	{
		printf("%s \n", "Failed to malloc duSingle");
	}

	double* dwSingle = (double*)malloc(size * sizeof(double));
	if (dwSingle == NULL)
	{
		printf("%s \n", "Failed to malloc dwSingle");
	}

	// Set the LHS 
	setSingleLHS(dsSingle, dlSingle, diagSingle, duSingle, dwSingle, a, b, c, d, e, size);

	// Factor the LHS
	pentFactor(dsSingle, dlSingle, diagSingle, duSingle, dwSingle, size);

	// Set f vectors and invert
	setInv1(inv1, a, d, e, size);
	setInv2(inv2, a, b, e, size);

	// Solve the inverses
	pentSolve(dsSingle, dlSingle, diagSingle, duSingle, dwSingle, inv1, size);
	pentSolve(dsSingle, dlSingle, diagSingle, duSingle, dwSingle, inv2, size);

	// Find z
	double z_11 = e * inv1[0] + a * inv1[size - 2] + b * inv1[size - 1]; 
	double z_12 = e * inv2[0] + a * inv2[size - 2] + b * inv2[size - 1]; 
	double z_21 = d * inv1[0] + e * inv1[1] + a * inv1[size - 1];
	double z_22 = d * inv2[0] + e * inv2[1] + a * inv2[size - 1];

	// Find matrix to be inverted
	double y_11 = c - z_11;
	double y_12 = d - z_12;
	double y_21 = b - z_21;
	double y_22 = c - z_22;

	// Invert to find omega
	double det = 1.0 / (y_11 * y_22 - y_21 * y_12);

	omega[0] = y_22 * det;							// 11
	omega[1] = - y_12 * det;						// 12
	omega[2] = - y_21 * det;						// 21
	omega[3] = y_11 * det;							// 22

	// Free memory local to this function
	free(dsSingle);
	free(dlSingle);
	free(diagSingle);
	free(duSingle);
	free(dwSingle);
}

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
)
{
    // Buffer used for error checking
    char msgStringBuffer[1024];

    // Set for inversion
    int gridInv = (nx % BLOCK_INV == 0) ? (nx / BLOCK_INV) : (nx / BLOCK_INV + 1);

    dim3 blockDimInv(BLOCK_INV);
    dim3 gridDimInv(gridInv);

    // Set for any standard grid
    int xGrid = (nx % BLOCK_X == 0) ? (nx / BLOCK_X) : (nx / BLOCK_X + 1);
    int yGrid = (nx % BLOCK_Y == 0) ? (nx / BLOCK_Y) : (nx / BLOCK_Y + 1);

    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim(xGrid, yGrid);

	// Solve system
	pentSolveBatch<<<gridDimInv, blockDimInv>>>(ds, dl,	diag, du, dw, data, size, nx);
    
    sprintf(msgStringBuffer, "Failed to solve system");
    checkError(msgStringBuffer);
	
	// Ensure transpose completed
	cudaDeviceSynchronize();

	// Solve for the end two points
	solveEnd<<<gridDimInv, blockDimInv>>>(data, a, b, d, e, omega[0], omega[1], omega[2], omega[3], nx, nx);
    
    sprintf(msgStringBuffer, "Failed to solve system for end two points");
    checkError(msgStringBuffer);

	// Ensure computation completed
	cudaDeviceSynchronize();

	// Reconstruct the full solution
	solveFull<<<gridDim, blockDim>>>(data, inv1Multi, inv2Multi, nx, nx);
    
    sprintf(msgStringBuffer, "Failed to reconstruct full system");
    checkError(msgStringBuffer);

	// Ensure computation completed
	cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------