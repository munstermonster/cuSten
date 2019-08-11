// Andrew Gloster
// August 2019
// Program to solve the 2D Cahn-Hilliard equation on a periodic domain using the ADI method
// This program is a serial version to allow for timing statistics to be collected

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

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "hdf5.h"
#include <time.h>

//---------------------------------------------------------------------
// Random uniform distribution function generator
//---------------------------------------------------------------------

static double double_rand(double min, double max)
{
    double scale = (double) rand() / (double) RAND_MAX; 
    return min + scale * ( max - min );
}


// ---------------------------------------------------------------------
// Set the vectors for the LHS
// ---------------------------------------------------------------------

static void setLHS
(
	double* ds, 
	double* dl, 
	double* diag, 
	double* du, 
	double* dw, 

	double a,
	double b,
	double c,
	double d,
	double e, 

	int nSolve
)
{
	for(int i = 0; i < nSolve; i++)
	{
		if (i > 1)
		{
			ds[i] = a;
		}
		if (i > 0)
		{
			dl[i] = b;
		}

		diag[i] = c;

		if (i < nSolve - 1)
		{
			du[i] = d;
		}

		if (i < nSolve - 2)
		{
			dw[i] = e;
		}
	}
}

// ---------------------------------------------------------------------
// Factorise the matrix
// ---------------------------------------------------------------------

static void pentFactor
(
	double* ds,  	
	double* dl,  	
	double* d, 	 	
	double* du,	 	
	double* dw,  	

	int nSolve
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
	for (int i = 2; i < nSolve - 2; i++)
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
	double* ds, 	
	double* dl,		
	double* d,		
	double* du,		
	double* dw,		
	
	double* b,		

	int nSolve 
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
	for (int i = 2; i < nSolve; i++)
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
	for (int i = nSolve - 3; i >= 0; i -= 1)
	{
		rowSecondAhead = rowCurrent + 1;
		rowAhead = rowCurrent;
		rowCurrent -= 1;

		b[rowCurrent] = b[rowCurrent] - du[rowCurrent] * b[rowAhead] - dw[rowCurrent] * b[rowSecondAhead];
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
// Recover the omega matrix
// ---------------------------------------------------------------------

void findOmega
(
	double* omega,

	double* inv1,
	double* inv2,

	double a,
	double b,
	double c,
	double d,
	double e,

	int nSolve
)
{
	// --------------------------
	// Malloc the necessary data for LHS and set up
	// --------------------------

	double* dsSingle = (double*)malloc(nSolve * sizeof(double));
	if (dsSingle == NULL)
	{
		printf("%s \n", "Failed to malloc dsSingle");
	}

	double* dlSingle = (double*)malloc(nSolve * sizeof(double));
	if (dlSingle == NULL)
	{
		printf("%s \n", "Failed to malloc dlSingle");
	}   

	double* diagSingle = (double*)malloc(nSolve * sizeof(double));
	if (diagSingle == NULL)
	{
		printf("%s \n", "Failed to malloc diagSingle");
	}

	double* duSingle = (double*)malloc(nSolve * sizeof(double));
	if (duSingle == NULL)
	{
		printf("%s \n", "Failed to malloc duSingle");
	}

	double* dwSingle = (double*)malloc(nSolve * sizeof(double));
	if (dwSingle == NULL)
	{
		printf("%s \n", "Failed to malloc dwSingle");
	}

	// Set the LHS 
	setLHS(dsSingle, dlSingle, diagSingle, duSingle, dwSingle, a, b, c, d, e, nSolve);

	// Factor the LHS
	pentFactor(dsSingle, dlSingle, diagSingle, duSingle, dwSingle, nSolve);

	// Set f vectors and invert
	setInv1(inv1, a, d, e, nSolve);
	setInv2(inv2, a, b, e, nSolve);

	// Solve the inverses
	pentSolve(dsSingle, dlSingle, diagSingle, duSingle, dwSingle, inv1, nSolve);
	pentSolve(dsSingle, dlSingle, diagSingle, duSingle, dwSingle, inv2, nSolve);

	// Find z
	double z_11 = e * inv1[0] + a * inv1[nSolve - 2] + b * inv1[nSolve - 1]; 
	double z_12 = e * inv2[0] + a * inv2[nSolve - 2] + b * inv2[nSolve - 1]; 
	double z_21 = d * inv1[0] + e * inv1[1] + a * inv1[nSolve - 1];
	double z_22 = d * inv2[0] + e * inv2[1] + a * inv2[nSolve - 1];

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
// Find cBar as per numerical scheme
// ---------------------------------------------------------------------

static void findCBar(double* cOld, double* cCurr, double* cBar, int n)
{
	// Indexing
    int temp, index;

    for (int j = 0; j < n; j++)
    {
        temp = j * n;
        for (int i = 0; i < n; i++)
        {
        	index = temp + i;

			// Find cBar
			cBar[index] = 2.0 * cCurr[index] - cOld[index];
		}
	}
}

// ---------------------------------------------------------------------
// Find the full combined RHS
// ---------------------------------------------------------------------

static void findRHS(
	double* cOld, 
	double* cCurr, 
	double* cHalf, 
	double* cNonLinRHS, 

	int n
)
{

	// Indexing
    int temp, index;

    for (int j = 0; j < n; j++)
    {
        temp = j * n;
        for (int i = 0; i < n; i++)
        {
        	index = temp + i;	

			// Set the RHS for inversion
			cHalf[index] += - (2.0 / 3.0) * (cCurr[index] - cOld[index]) + cNonLinRHS[index];

			// Set cOld to cCurr
			cOld[index] = cCurr[index];
		}
	}
}

// ---------------------------------------------------------------------
// Compute the linear finite difference
// ---------------------------------------------------------------------

static void linearRHS(
	double* cBar, 
	double* cHalf, 
	double* weightsLinRHS, 

	int linVert, 
	int linHoriz,
	int linLeft, 
	int linTop, 
	int n
)
{

	// Indexing
    int temp, index;
    int tempOffset, indexOffset;
    int checkI, checkJ;
    int stenTemp, stenIndex;

    // Accumulator
    double sum;


    for (int j = 0; j < n; j++)
    {
        temp = j * n;
        for (int i = 0; i < n; i++)
        {
			index = temp + i;

			sum = 0.0;

			for (int offsetJ = 0; offsetJ < linVert; offsetJ++)
			{	
				stenTemp = offsetJ * linHoriz;

				checkJ = j - linTop + offsetJ;

				if (checkJ < 0)
				{
					tempOffset = (checkJ + n) * n;
				}
				else
				{
					tempOffset = (checkJ % n) * n;
				}

				for (int offsetI = 0; offsetI < linHoriz; offsetI++)
				{
					stenIndex = stenTemp + offsetI;

					checkI = i - linTop + offsetI;

					if (checkI < 0)
					{
						indexOffset = tempOffset + (checkI + n);
					}
					else
					{
						indexOffset = tempOffset + (checkI % n);
					}

					sum += weightsLinRHS[stenIndex] * cBar[indexOffset];
				}
			}

			cHalf[index] = sum;
		}
	}
}

// ---------------------------------------------------------------------
// Compute the linear finite difference
// ---------------------------------------------------------------------

static void nonlinearRHS(
	double* cCurr, 
	double* cNonLinRHS, 
	double* weightsNonLinRHS, 

	int nonlinVert, 
	int nonlinHoriz,
	int nonlinLeft, 
	int nonlinTop, 
	int n
)
{

	// Indexing
    int temp, index;
    int tempOffset, indexOffset;
    int checkI, checkJ;
    int stenTemp, stenIndex;

    // Accumulator
    double sum;


    for (int j = 0; j < n; j++)
    {
        temp = j * n;
        for (int i = 0; i < n; i++)
        {
			index = temp + i;

			sum = 0.0;

			for (int offsetJ = 0; offsetJ < nonlinVert; offsetJ++)
			{	
				stenTemp = offsetJ * nonlinHoriz;

				checkJ = j - nonlinTop + offsetJ;

				if (checkJ < 0)
				{
					tempOffset = (checkJ + n) * n;
				}
				else
				{
					tempOffset = (checkJ % n) * n;
				}

				for (int offsetI = 0; offsetI < nonlinHoriz; offsetI++)
				{
					stenIndex = stenTemp + offsetI;

					checkI = i - nonlinTop + offsetI;

					if (checkI < 0)
					{
						indexOffset = tempOffset + (checkI + n);
					}
					else
					{
						indexOffset = tempOffset + (checkI % n);
					}

					sum += weightsNonLinRHS[stenIndex] * ((cCurr[indexOffset] * cCurr[indexOffset] * cCurr[indexOffset]) - cCurr[indexOffset]);
				}
			}

			cNonLinRHS[index] = sum;
		}
	}
}

// ---------------------------------------------------------------------
// Solve for the final two points in the system
// ---------------------------------------------------------------------

static void solveEnd
(
	double* data,

	double a,
	double b,
	double d,
	double e,

	double omega_11,
	double omega_12,
	double omega_21,
	double omega_22,

	int n
)
{
	// Last two vectors
	double newNx2; 
	double newNx1;

	// Compute lambda = d^~ - transpose(g) * inverse(E) * d_hat
	newNx2 = data[n - 2] - (e * data[0] + a * data[n - 4] + b * data[n - 3]);
	newNx1 = data[n - 1] - (d * data[0] + e * data[1] + a * data[n - 3]);

	// Compute x^~ = omega * lambda
	data[n - 2] = omega_11 * newNx2 + omega_12 * newNx1;
	data[n - 1] = omega_21 * newNx2 + omega_22 * newNx1;
}

// ---------------------------------------------------------------------
// Reconstruct the full solution
// ---------------------------------------------------------------------

static void solveFull
(
	double* data,

	double* inv1,
	double* inv2,

	int n
)
{
	// Set values to last two entries in array
	double oldNx2 = data[n - 2]; // Two points from end
	double oldNx1 = data[n - 1]; // One point from end

	for (int i = 0; i < n - 2; i++)
	{
		data[i] = data[i] - (inv1[i] * oldNx2 + inv2[i] * oldNx1);
	}
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

	double* inv1,
	double* inv2,

	double* omega,

	double* data,

	double a,
	double b, 
	double d,
	double e,

	int nSolve,
	int n
)
{
	// Solve system
	pentSolve(ds, dl, diag, du, dw, data, nSolve);

	// Solve for the end two points
	solveEnd(data, a, b, d, e, omega[0], omega[1], omega[2], omega[3], n);

	// Reconstruct the full solution
	solveFull(data, inv1, inv2, n);
}

// ---------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------

void transpose(double* old, double* new, int n)
{
	for (int j = 0; j < n; j++)
	{
		for (int i = 0; i < n; i++)
		{
			new[i * n + j] = old[j * n + i];
		}
	}
}

// ---------------------------------------------------------------------
// Recover the updated timestep
// ---------------------------------------------------------------------

static void findNew(double* cCurr, double* cBar, double* cHalf, int n)
{

	// Indexing
    int temp, index;

    for (int j = 0; j < n; j++)
    {
        temp = j * n;
        for (int i = 0; i < n; i++)
        {
        	index = temp + i;

			cCurr[index] = cBar[index] + cHalf[index];
		}
	}
}

// ---------------------------------------------------------------------
// Main program
// ---------------------------------------------------------------------

int main(int argc, char *argv[])
{
    //----------------------------------------
    // Simulation paramters
    //----------------------------------------

    // Set coefficients
    double dCahn = 1.0;
    double gammaCahn = 0.01;

    // Set grid spacing -- Use a square grid -- thus all n = ny
    // Read from command line
    int n;
    n = atoi(argv[1]);

    // Set the size of the reduced matrix
    int nSolve = n - 2;

    // Set timing
    double T = 10.0;

    // Domain size
    double lx = 16.0 * M_PI;

    // Spacings
    double dx = lx / n;
    double dt = 0.1 * dx;

    //  Buffer used for error checking
    char msgStringBuffer[1024];

    // What device to compute on
    int computeDevice = 0;	

	//----------------------------------------
    // Memory allocation
    //----------------------------------------    
    
    // Old timestep
    double* cOld;
    cOld = (double*)malloc(n * n * sizeof(double));
	if (cOld == NULL)  
	{
		printf("Failed to allocate cOld \n");
		return 1;
	}

    // // Current timestep
    double* cCurr;
    cCurr = (double*)malloc(n * n * sizeof(double));
	if (cCurr == NULL)  
	{
		printf("Failed to allocate cCurr \n");
		return 1;
	}

    // // New timestep
    double* cNonLinRHS;
    cNonLinRHS = (double*)malloc(n * n * sizeof(double));
	if (cNonLinRHS == NULL)  
	{
		printf("Failed to allocate cNonLinRHS \n");
		return 1;
	} 

    // // Intermediate step
    double* cBar;
    cBar = (double*)malloc(n * n * sizeof(double));
	if (cBar == NULL)  
	{
		printf("Failed to allocate cBar \n");
		return 1;
	}
    
    // // Intermediate step
    double* cHalf;
    cHalf = (double*)malloc(n * n * sizeof(double));
	if (cHalf == NULL)  
	{
		printf("Failed to allocate cHalf \n");
		return 1;
	} 

    //----------------------------------------
    // Initial Condition
    //---------------------------------------- 

    // Indexing
    int temp, index;

    for (int j = 0; j < n; j++)
    {
        temp = j * n;
        for (int i = 0; i < n; i++)
        {
            index = temp + i;

            cOld[index] = double_rand(- 0.1, 0.1);
            cCurr[index] = cOld[index];
        }
    }

    //----------------------------------------
    // Allocate the memory for the LHS
    //----------------------------------------

    // Lowest diagonal
    double* ds;
    ds = (double*)malloc(nSolve * sizeof(double));
	if (ds == NULL)  
	{
		printf("Failed to allocate ds \n");
		return 1;
	} 

    // Lower diagonal
    double* dl;
    dl = (double*)malloc(nSolve * sizeof(double));
	if (ds == NULL)  
	{
		printf("Failed to allocate dl \n");
		return 1;
	} 

    // Main daigonal
    double* diag;
    diag = (double*)malloc(nSolve * sizeof(double));
	if (ds == NULL)  
	{
		printf("Failed to allocate diag \n");
		return 1;
	} 
	
    // Upper diagonal
    double* du;
    du = (double*)malloc(nSolve * sizeof(double));
	if (du == NULL)  
	{
		printf("Failed to allocate du \n");
		return 1;
	} 
	 
    // Highest diagonal
    double* dw;
    dw = (double*)malloc(nSolve * sizeof(double));
	if (ds == NULL)  
	{
		printf("Failed to allocate dw \n");
		return 1;
	} 

    //----------------------------------------
    // Set up LHS
    //---------------------------------------- 	

	// Linear coefficient
    double simgaLin = 2.0 * dt * dCahn * gammaCahn / (3.0 * (dx * dx * dx * dx));

    // Set the diagonal elements
    double a = simgaLin;
    double b = - 4 * simgaLin;
    double c = 1 + 6 * simgaLin;
    double d = - 4 * simgaLin;
    double e = simgaLin;

    // Set the LHS for inversion
    setLHS(ds, dl, diag, du, dw, a, b, c, d, e, nSolve);

    // Prefactor the LHS matrix
	pentFactor(ds, dl, diag, du, dw, nSolve);

    //----------------------------------------
    // Set up cyclic method
    //----------------------------------------

    double* omega = (double*)malloc(4 * sizeof(double));
	if (omega == NULL)
	{
		printf("%s \n", "Failed to malloc omega");
	}

	double* inv1 = (double*)malloc(nSolve * sizeof(double));
	if (inv1 == NULL)
	{
		printf("%s \n", "Failed to malloc inv1");
	}

	double* inv2 = (double*)malloc(nSolve * sizeof(double));
	if (inv2 == NULL)
	{
		printf("%s \n", "Failed to malloc inv2");
	}

	findOmega(omega, inv1, inv2, a, b, c, d, e, nSolve);

    //----------------------------------------
    // Set compute for linear RHS
    //----------------------------------------

    int linHoriz = 5;
	int linLeft = 2;
	int linRight = 2;

	int linVert = 5;
	int linTop = 2;
	int linBottom = 2;

	double* weightsLinRHS;
	weightsLinRHS = (double*)malloc(linHoriz * linVert * sizeof(double));

	weightsLinRHS[0] = 0.0; 						weightsLinRHS[1] = 0.0;								weightsLinRHS[2] = - 1.0 * simgaLin;					weightsLinRHS[3] = 0.0;								weightsLinRHS[4] = 0.0;					
	weightsLinRHS[5] = 0.0;							weightsLinRHS[6] = - 2.0 * simgaLin; 				weightsLinRHS[7] = 8.0 * simgaLin;						weightsLinRHS[8] = - 2.0 * simgaLin;				weightsLinRHS[9] = 0.0; 			
	weightsLinRHS[10] = - 1.0 * simgaLin; 			weightsLinRHS[11] = 8.0 * simgaLin;					weightsLinRHS[12] = - 20.0 * simgaLin;					weightsLinRHS[13] = 8.0 * simgaLin;					weightsLinRHS[14] = - 1.0 * simgaLin;					
	weightsLinRHS[15] = 0.0;						weightsLinRHS[16] = - 2.0 * simgaLin; 				weightsLinRHS[17] = 8.0 * simgaLin;						weightsLinRHS[18] = - 2.0 * simgaLin;				weightsLinRHS[19] = 0.0;
	weightsLinRHS[20] = 0.0; 						weightsLinRHS[21] = 0.0;							weightsLinRHS[22] = -1.0 * simgaLin;					weightsLinRHS[23] = 0.0;							weightsLinRHS[24] = 0.0;					


    //----------------------------------------
    // Set up computation of non-linear RHS 
    //----------------------------------------   

	// Set new non linear coefficient
	double sigmaNonLin = (dt / 3.0) * dCahn * (2.0 / (dx * dx));

	int nonlinHoriz = 3;
	int nonlinLeft = 1;
	int nonlinRight = 1;

	int nonlinVert = 3;
	int nonlinTop = 1;
	int nonlinBottom = 1;

	double* weightsNonLinRHS;
	weightsNonLinRHS = (double*)malloc(nonlinHoriz * nonlinVert * sizeof(double));

	weightsNonLinRHS[0] = 0.0;						weightsNonLinRHS[1] = 1.0 * sigmaNonLin;				weightsNonLinRHS[2] = 0.0;
	weightsNonLinRHS[3] = 1.0 * sigmaNonLin;		weightsNonLinRHS[4] = - 4.0 * sigmaNonLin;				weightsNonLinRHS[5] = 1.0 * sigmaNonLin;
	weightsNonLinRHS[6] = 0.0;						weightsNonLinRHS[7] = 1.0 * sigmaNonLin;				weightsNonLinRHS[8] = 0.0;

    //----------------------------------------
	// Timestepping
    //----------------------------------------

    // Track current time-step
    double t = 0.0;

    // Timing
	clock_t begin = clock();

    while (t < T)
    {
    	// Set cBar
        findCBar(cOld, cCurr, cBar, n);

        // Compute the linear RHS
		linearRHS(cBar, cHalf, weightsLinRHS, linVert, linHoriz, linLeft, linTop, n);

		// Compute the non linear RHS
		nonlinearRHS(cCurr, cNonLinRHS, weightsNonLinRHS, nonlinVert, nonlinHoriz, nonlinLeft, nonlinTop, n);

    	// Find the full RHS and then set cOld to cCurrent
		findRHS(cOld, cCurr, cHalf, cNonLinRHS, n);

		// Loop over x direction solve
		for (int i = 0; i < n; i++)
		{
			cyclicInv(ds, dl, diag, du, dw, inv1, inv2, omega, &cHalf[i * n], a, b, d, e, nSolve, n);
		}

		// Transpose to sweep in y
		transpose(cHalf, cCurr, n);
		
		// Loop over y direction solve
		for (int i = 0; i < n; i++)
		{
			cyclicInv(ds, dl, diag, du, dw, inv1, inv2, omega, &cCurr[i * n], a, b, d, e, nSolve, n);
		}

		// Transform back to x
		transpose(cCurr, cHalf, n);

		// Find updated time step
		findNew(cCurr, cBar, cHalf, n);

        // Add on the next time
        t += dt;
    }

    // Print out timing
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("%lf \n", time_spent);

    //----------------------------------------
    // Free memory
    //---------------------------------------- 

    free(cOld);
    free(cCurr);
    free(cNonLinRHS);
    free(cBar);
    free(cHalf);

    free(ds);
    free(dl);
    free(diag);
    free(du);
    free(dw);

    free(inv1);
    free(inv2);
    free(omega);

    free(weightsLinRHS);
}	
