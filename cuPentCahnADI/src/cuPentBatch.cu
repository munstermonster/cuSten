// Andrew Gloster
// June 2018
// Function declarations for cuPentBatch routine to solve batches of pentadiagonal systems

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
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  User Libraries and Headers
// ---------------------------------------------------------------------

#include "cuPentBatch.h"

// ---------------------------------------------------------------------
// Function to factorise the LHS matrix
// ---------------------------------------------------------------------

__global__ void pentFactorBatch
(
	double* ds,  	// Array containing the lower diagonal, 2 away from the main diagonal. First two elements are 0. Stored in interleaved format.
	double* dl,  	// Array containing the lower diagonal, 1 away from the main diagonal. First elements is 0. Stored in interleaved format.
	double* d, 	 	// Array containing the main diagonal. Stored in interleaved format.
	double* du,	 	// Array containing the upper diagonal, 1 away from the main diagonal. Last element is 0. Stored in interleaved format.
	double* dw,  	// Array containing the upper diagonal, 2 awy from the main diagonal. Last 2 elements are 0. Stored in interleaved format.

	const int m,  		// Size of the linear systems, number of unknowns
	const int batchCount	// Number of linear systems
)
{

	// Indices used to store relative indexes
	int rowCurrent;
	int rowPrevious;
	int rowSecondPrevious;

	// Starting index
    rowCurrent = blockDim.x * blockIdx.x + threadIdx.x;

    // Only want to solve equations that exist
    if (rowCurrent < batchCount)
    {
		// First Row
		d[rowCurrent] = d[rowCurrent];
		du[rowCurrent] = du[rowCurrent] / d[rowCurrent];
		dw[rowCurrent] = dw[rowCurrent] / d[rowCurrent];

		// Second row index
		rowPrevious = rowCurrent;
		rowCurrent += batchCount;

		// Second row
		dl[rowCurrent] = dl[rowCurrent];

		d[rowCurrent] = d[rowCurrent] - dl[rowCurrent] * du[rowPrevious];

		du[rowCurrent] = (du[rowCurrent] - dl[rowCurrent] * dw[rowPrevious]) / d[rowCurrent];

		dw[rowCurrent] = dw[rowCurrent] / d[rowCurrent];

		// Interior rows - Note 0 indexing
		#pragma unroll
		for (int i = 2; i < m - 2; i++)
		{
			rowSecondPrevious = rowCurrent - batchCount; 
			rowPrevious = rowCurrent;
			rowCurrent += batchCount;

			dl[rowCurrent] = dl[rowCurrent] - ds[rowCurrent] * du[rowSecondPrevious];
			
			d[rowCurrent] = d[rowCurrent] - ds[rowCurrent] * dw[rowSecondPrevious] - dl[rowCurrent] * du[rowPrevious];

			dw[rowCurrent] = dw[rowCurrent] / d[rowCurrent];

			du[rowCurrent] = (du[rowCurrent] - dl[rowCurrent] * dw[rowPrevious]) / d[rowCurrent];
		}

		// Second last row indexes
		rowSecondPrevious = rowCurrent - batchCount; 
		rowPrevious = rowCurrent;
		rowCurrent += batchCount;

		// Second last row
		dl[rowCurrent] = dl[rowCurrent] - ds[rowCurrent] * du[rowSecondPrevious];
		d[rowCurrent] = d[rowCurrent] - ds[rowCurrent] * dw[rowSecondPrevious] - dl[rowCurrent] * du[rowPrevious];
		du[rowCurrent] = (du[rowCurrent] - dl[rowCurrent] * dw[rowPrevious]) / d[rowCurrent];

		// Last row indexes
		rowSecondPrevious = rowCurrent - batchCount; 
		rowPrevious = rowCurrent;
		rowCurrent += batchCount;

		// Last row
		dl[rowCurrent] = dl[rowCurrent] - ds[rowCurrent] * du[rowSecondPrevious];
		d[rowCurrent] = d[rowCurrent] - ds[rowCurrent] * dw[rowSecondPrevious] - dl[rowCurrent] * du[rowPrevious];
	}
}

// ---------------------------------------------------------------------
// Function to solve the Ax = b system of pentadiagonal matrices
// ---------------------------------------------------------------------

__global__ void pentSolveBatch
(
	double* ds, 	// Array containing updated ds after using pentFactorBatch
	double* dl,		// Array containing updated ds after using pentFactorBatch
	double* d,		// Array containing updated ds after using pentFactorBatch
	double* du,		// Array containing updated ds after using pentFactorBatch
	double* dw,		// Array containing updated ds after using pentFactorBatch
	
	double* b,		// Dense array of RHS stored in interleaved format

	const int m,  		// Size of the linear systems, number of unknowns
	const int batchCount	// Number of linear systems
)
{

	// Indices used to store relative indexes
	int rowCurrent;
	int rowPrevious;
	int rowSecondPrevious;

	int rowAhead;
	int rowSecondAhead;

	// Starting index
    rowCurrent = blockDim.x * blockIdx.x + threadIdx.x;

    // Only want to solve equations that exist
    if (rowCurrent < batchCount)
    {
    	// --------------------------
		// Forward Substitution
		// --------------------------

		// First Row
		b[rowCurrent] = b[rowCurrent] / d[rowCurrent];

		// Second row index
		rowPrevious = rowCurrent;
		rowCurrent += batchCount;

		// Second row
		b[rowCurrent] = (b[rowCurrent] - dl[rowCurrent] * b[rowPrevious]) / d[rowCurrent];

		// Interior rows - Note 0 indexing
		#pragma unroll
		for (int i = 2; i < m; i++)
		{
			rowSecondPrevious = rowCurrent - batchCount; 
			rowPrevious = rowCurrent;
			rowCurrent += batchCount;

			b[rowCurrent] = (b[rowCurrent] - ds[rowCurrent] * b[rowSecondPrevious] - dl[rowCurrent] * b[rowPrevious]) / d[rowCurrent];	
		}

    	// --------------------------
		// Backward Substitution
		// --------------------------

		// Last row
		b[rowCurrent] = b[rowCurrent];

		// Second last row index
		rowAhead = rowCurrent;
		rowCurrent -= batchCount;

		// Second last row
		b[rowCurrent] = b[rowCurrent] - du[rowCurrent] * b[rowAhead];

		// Interior points - Note row indexing
		#pragma unroll
		for (int i = m - 3; i >= 0; i -= 1)
		{
			rowSecondAhead = rowCurrent + batchCount;
			rowAhead = rowCurrent;
			rowCurrent -= batchCount;

			b[rowCurrent] = b[rowCurrent] - du[rowCurrent] * b[rowAhead] - dw[rowCurrent] * b[rowSecondAhead];
		}
	}
}

// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------
