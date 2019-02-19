// Andrew Gloster
// February 2019
// Program to solve the 2D Cahn-Hilliard equation on a periodic domain using the ADI method

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
#include "cuda.h"
#include <cublas_v2.h>
#include "hdf5.h"
#include <time.h>

// ---------------------------------------------------------------------
//  Programmer Libraries and Headers
// ---------------------------------------------------------------------

#include "../../cuSten/cuSten.h"
#include "cuPentBatch.h"
#include "BatchHyper.h"

// ---------------------------------------------------------------------
// MACROS
// ---------------------------------------------------------------------

// Block sizes for finding RHS
#define BLOCK_X_FUN 8
#define BLOCK_Y_FUN 8

#define BLOCK_X 32
#define BLOCK_Y 32

// Block size for inverting
#define BLOCK_INV 64

//---------------------------------------------------------------------
// Static functions for use in main program
//---------------------------------------------------------------------

// Find cBar for differencing
__global__ static void findCBar(double* cOld, double* cCurr, double* cBar, int nx)
{
	// Matrix index
	int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

	// Set index being computed
	int index = globalIdy * nx + globalIdx;

	// Find cBar
	cBar[index] = 2.0 * cCurr[index] - cOld[index];
}

// Find the full combined RHS
__global__ static void findRHS(double* cOld, double* cCurr, double* cHalf, double* cNonLinRHS, int nx)
{
	// Matrix index
	int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

	// Set index being computed
	int index = globalIdy * nx + globalIdx;

	// Set the RHS for inversion
	cHalf[index] += - (2.0 / 3.0) * (cCurr[index] - cOld[index]) + cNonLinRHS[index];

	// Set cOld to cCurr
	cOld[index] = cCurr[index];
}

// Recover the updated timestep
__global__ static void findNew(double* cCurr, double* cBar, double* cHalf, int nx)
{
	// Matrix index
	int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

	// Set index being computed
	int index = globalIdy * nx + globalIdx;

	// Recover the new data
	cCurr[index] = cBar[index] + cHalf[index];
}

// Print out to hdf5
static herr_t Print_Out(double* data, double time, int nx, int ny){
    
    char str_num[1024];
    char str_file[2048];

    snprintf(str_num, 20, "%0.10lf", time);

    snprintf(str_file, 100, "%s%s%s", "output/cahn_hilliard_", str_num, ".nc");

    // Print out matrix
    hsize_t dims[2];

    hid_t file_id, dataset_id, dataspace_id;
    herr_t status;

    file_id = H5Fcreate(str_file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    dims[0] = nx;
    dims[1] = ny;

    dataspace_id = H5Screate_simple(2, dims, NULL);

    dataset_id = H5Dcreate2(file_id, "/c", H5T_NATIVE_DOUBLE, dataspace_id, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    /* End access to the dataset and release resources used by it. */
    status = H5Dclose(dataset_id);

    /* Terminate access to the data space. */ 
    status = H5Sclose(dataspace_id);

    /* Close the file. */
    status = H5Fclose(file_id);

    return status;
}

static double double_rand(double min, double max)
{
    double scale = (double) rand() / (double) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

//---------------------------------------------------------------------
// Function to calculate the non linear RHS
//---------------------------------------------------------------------

/*! \var typedef double (*devArg1X)(double*, double*, int);
    \brief The function pointer containing the user defined function to be applied <br>
    Input 1: The pointer to input data to the function <br>
    Input 2: The pointer to the coefficients provided by the user <br>
    Input 3: The current index position (centre of the stencil to be applied) <br>
    Input 4: Value to be used to jump between rows. (j + 1, j - 1 etc.) <br>
    Input 5: Size of stencil in x direction <br>
    Input 6: Size of stencil in y direction
*/

typedef double (*devArg1XY)(double*, double*, int, int, int, int);

__inline__ __device__ double nonLinRHS(double* data, double* coe, int loc, int jump, int nx, int ny)
{	
	double result = 0.0;
	double current;
	int temp;
	int count = 0;

	#pragma unroll
	for (int j = 0; j < ny; j++)
	{
		temp = loc + j * jump;

		#pragma unroll
		for (int i = 0; i < nx; i++)
		{
			current = data[temp + i];

			result += coe[count] * ((current * current * current) - current);

			count ++;
		}
	}

	return result;
}

__device__ devArg1XY devFunc = nonLinRHS;


// ---------------------------------------------------------------------
//  Begin main program
// ---------------------------------------------------------------------

int main()
{
    //----------------------------------------
    // Simulation paramters
    //----------------------------------------

    // Set coefficients
    double D = 0.6;
    double gamma = 0.01;

    // Set grid spacing -- Use a square grid -- thus all nx = ny
    int nx = 1 << 6;

    // Set the size of the reduced matrix
    int size = nx - 2;

    // Set timing
    double T = 10.0;

    // Domain size
    double lx = 2.0 * M_PI;

    // Spacings
    double dx = lx / nx;
    double dt = dx * dx / 4.0;

    //  Buffer used for error checking
    char msgStringBuffer[1024];

    // How often to output
    int print = 1000;

    // What device to compute on
    int computeDevice = 0;

    //----------------------------------------
    // Set up GPU grids
    //----------------------------------------

    // Set for inversion
    int gridInv = (nx % BLOCK_INV == 0) ? (nx / BLOCK_INV) : (nx / BLOCK_INV + 1);

    dim3 blockDimInv(BLOCK_INV);
    dim3 gridDimInv(gridInv);

    // Set for any standard grid
    int xGrid = (nx % BLOCK_X == 0) ? (nx / BLOCK_X) : (nx / BLOCK_X + 1);
    int yGrid = (nx % BLOCK_Y == 0) ? (nx / BLOCK_Y) : (nx / BLOCK_Y + 1);

    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim(xGrid, yGrid);

    //----------------------------------------
    // Memory allocation
    //----------------------------------------    
    
    // Old timestep
    double* cOld;
    cudaMallocManaged(&cOld, nx * nx * sizeof(double));
   
    sprintf(msgStringBuffer, "Failed to allocate memory for cOld");
    checkError(msgStringBuffer);    

    // Current timestep
    double* cCurr;
    cudaMallocManaged(&cCurr, nx * nx * sizeof(double));
   
    sprintf(msgStringBuffer, "Failed to allocate memory for cCurr");
    checkError(msgStringBuffer);

    // New timestep
    double* cNonLinRHS;
    cudaMallocManaged(&cNonLinRHS, nx * nx * sizeof(double));
   
    sprintf(msgStringBuffer, "Failed to allocate memory for cNonLinRHS");
    checkError(msgStringBuffer);   

    // Intermediate step
    double* cBar;
    cudaMallocManaged(&cBar, nx * nx * sizeof(double));
   
    sprintf(msgStringBuffer, "Failed to allocate memory for cBar");
    checkError(msgStringBuffer);  
    
    // Intermediate step
    double* cHalf;
    cudaMallocManaged(&cHalf, nx * nx * sizeof(double));
    
    sprintf(msgStringBuffer, "Failed to allocate memory for cBar");
    checkError(msgStringBuffer); 

    //----------------------------------------
    // Initial Condition
    //---------------------------------------- 

    // Indexing
    int temp, index;

    for (int j = 0; j < nx; j++)
    {
        temp = j * nx;
        for (int i = 0; i < nx; i++)
        {
            index = temp + i;

            cOld[index] = double_rand(- 0.1, 0.1);
        }
    }

    //----------------------------------------
    // Allocate the memory for the LHS
    //----------------------------------------

    // Lowest diagonal
    double* ds;
    cudaMallocManaged(&ds, size * nx * sizeof(double));

    sprintf(msgStringBuffer, "Failed to allocate memory for ds");
    checkError(msgStringBuffer);   

    // Lower diagonal
    double* dl;
    cudaMallocManaged(&dl, size * nx * sizeof(double));

    sprintf(msgStringBuffer, "Failed to allocate memory for dl");
    checkError(msgStringBuffer);  

    // Main daigonal
    double* diag;
    cudaMallocManaged(&diag, size * nx * sizeof(double));

    sprintf(msgStringBuffer, "Failed to allocate memory for diag");
    checkError(msgStringBuffer); 

    // Upper diagonal
    double* du;
    cudaMallocManaged(&du, size * nx * sizeof(double));

    sprintf(msgStringBuffer, "Failed to allocate memory for du");
    checkError(msgStringBuffer);   

    // Highest diagonal
    double* dw;
    cudaMallocManaged(&dw, size * nx * sizeof(double));

    sprintf(msgStringBuffer, "Failed to allocate memory for dw");
    checkError(msgStringBuffer);   

    //----------------------------------------
    // Coefficients for initial timestep
    //----------------------------------------

    // These will be replaced later for the double step version
    double simgaLin = dt / (2.0 * (pow(dx, 4.0))) * D * gamma;

    // Set the diagonal elements
    double a = simgaLin;
    double b = - 4 * simgaLin;
    double c = 1 + 6 * simgaLin;
    double d = - 4 * simgaLin;
    double e = simgaLin;

    //----------------------------------------
    // Set the matrix for the initial timestep
    //----------------------------------------

    // Set the LHS for inversion
    setMultiLHS<<<gridDim, blockDim>>>(ds, dl, diag, du, dw, a, b, c, d, e, size, nx);

    sprintf(msgStringBuffer, "Failed to set LHS matrix for initial timestep");
    checkError(msgStringBuffer);

	// Ensure matrix is set
	cudaDeviceSynchronize();

    // Pre-factor the LHS
    pentFactorBatch<<<gridDimInv, blockDimInv>>>(ds, dl, diag, du, dw, size, nx);

    sprintf(msgStringBuffer, "Failed to pre factor LHS matrix for initial timestep");
    checkError(msgStringBuffer);

	// Ensure matrix is factorised
	cudaDeviceSynchronize();

    //----------------------------------------
    // Find Omega
    //----------------------------------------

    double* omega = (double*)malloc(4 * sizeof(double));
	if (omega == NULL)
	{
		printf("%s \n", "Failed to malloc omega");
	}

	double* inv1Single = (double*)malloc(size * sizeof(double));
	if (inv1Single == NULL)
	{
		printf("%s \n", "Failed to malloc inv1Single");
	}

	double* inv2Single = (double*)malloc(size * sizeof(double));
	if (inv2Single == NULL)
	{
		printf("%s \n", "Failed to malloc inv2Single");
	}

    findOmega(omega, inv1Single, inv2Single, a, b, c, d, e, nx);

    //----------------------------------------
    // Copy the inverses to full matrix
    //----------------------------------------    

    double* inv1Multi;
    cudaMallocManaged(&inv1Multi, nx * size * sizeof(double));

    sprintf(msgStringBuffer, "Failed to allocate memory for inv1Multi");
    checkError(msgStringBuffer); 

    double* inv2Multi;
    cudaMallocManaged(&inv2Multi, nx * size * sizeof(double));

    sprintf(msgStringBuffer, "Failed to allocate memory for inv2Multi");
    checkError(msgStringBuffer); 

    for (int j = 0; j < size; j++)
    {	
    	temp = j * nx;

    	for (int i = 0; i < nx; i++)
    	{
    		index = temp + i;

    		inv1Multi[index] = inv1Single[j]; 
    		inv2Multi[index] = inv2Single[j];
    	}
    }

    //----------------------------------------
    // Set up computation of non-linear RHS - x
    //----------------------------------------   

	int numStenHoriz = 3;
	int numStenLeft = 1;
	int numStenRight = 1;

	int numStenVert = 3;
	int numStenTop = 1;
	int numStenBottom = 1;

	double* coe;
	cudaMallocManaged(&coe, numStenHoriz * numStenVert * sizeof(double));

	double sigmaNonLin = (dt / 2.0) * D * (1.0 / pow(dx, 2.0));

	coe[0] = 0.0;						coe[1] = 1.0 * sigmaNonLin;				coe[2] = 0.0;
	coe[3] = 1.0 * sigmaNonLin;			coe[4] = - 4.0 * sigmaNonLin;			coe[5] = 1.0 * sigmaNonLin;
	coe[6] = 0.0;						coe[7] = 1.0 * sigmaNonLin;				coe[8] = 0.0;

	// Set up the compute device structs
	cuSten_t nonLinInitComputeX;

	// Copy the function to device memory
	double* func;
	cudaMemcpyFromSymbol(&func, devFunc, sizeof(devArg1XY));	

	// Set the number of tiles
	int nonLinTiles = 1;

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// Initialise the instance of the stencil
	custenCreate2DXYpFun(&nonLinInitComputeX, computeDevice, nonLinTiles, nx, nx, BLOCK_X_FUN, BLOCK_Y_FUN, cNonLinRHS, cOld, coe, numStenHoriz, numStenLeft, numStenRight, numStenVert, numStenTop, numStenBottom, func);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();	

    //----------------------------------------
    // Set up computation of linear RHS - x
    //---------------------------------------- 

    int xInitHoriz = 3;
	int xInitLeft = 1;
	int xInitRight = 1;

	int xInitVert = 5;
	int xInitTop = 2;
	int xInitBottom = 2;

	double* weightsInitRHSx;
	cudaMallocManaged(&weightsInitRHSx, xInitHoriz * xInitVert * sizeof(double));

	weightsInitRHSx[0] = 0.0; 						weightsInitRHSx[1] = - 1.0 * simgaLin;					weightsInitRHSx[2] = 0.0;
	weightsInitRHSx[3] = - 2.0 * simgaLin;			weightsInitRHSx[4] = 8.0 * simgaLin;					weightsInitRHSx[5] = - 2.0 * simgaLin;
	weightsInitRHSx[6] = 4.0 * simgaLin; 			weightsInitRHSx[7] = 1 - 14.0 * simgaLin;				weightsInitRHSx[8] = 4.0 * simgaLin;
	weightsInitRHSx[9] = - 2.0 * simgaLin; 			weightsInitRHSx[10] = 8.0 * simgaLin;					weightsInitRHSx[11] = - 2.0 * simgaLin;
	weightsInitRHSx[12] = 0.0;						weightsInitRHSx[13] = - 1.0 * simgaLin;					weightsInitRHSx[14] = 0.0;

	// Set up the compute device structs
	cuSten_t xInitRHS;

	// Set the number of tiles
	int xInitTiles = 1;

	// Initialise the instance of the stencil
	custenCreate2DXYp(&xInitRHS, computeDevice, xInitTiles, nx, nx, BLOCK_X, BLOCK_Y, cHalf, cOld, weightsInitRHSx, xInitHoriz, xInitLeft, xInitRight, xInitVert, xInitTop, xInitBottom);
	
	// Ensure compute type created
	cudaDeviceSynchronize();

    //----------------------------------------
    // Set up cuBLAS
    //---------------------------------------- 

	// Set a handle
	cublasHandle_t handleBLAS;

	// Set a status
	cublasStatus_t statusBLAS;

	// Create the handle
	statusBLAS = cublasCreate(&handleBLAS);

	// Set constants
	const double alpha = 1.0;
	const double beta = 0.0;

	// Set addition constants
	const double constant = 1.0;

    //----------------------------------------
    // Compute the initial half step - transpose - batch invert - transpose back
    //---------------------------------------- 

	// Run the computation - non-linear RHS
	custenCompute2DXYpFun(&nonLinInitComputeX, 0);

	// Run the computation - linear RHS
	custenCompute2DXYp(&xInitRHS, 0);

	// Ensure computation completed
	cudaDeviceSynchronize();

	// Add two vectors together
	statusBLAS  = cublasDaxpy(handleBLAS, nx * nx, &constant, cNonLinRHS, 1, cHalf, 1);
	
	if (statusBLAS != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to add vectors \n");
		return EXIT_FAILURE;
	}

	// Ensure computation completed
	cudaDeviceSynchronize();	

	// Transpose the result
	statusBLAS = cublasDgeam(handleBLAS, CUBLAS_OP_T, CUBLAS_OP_T, nx, nx, &alpha, cHalf, nx, &beta, NULL, nx, cCurr, nx);

	if (statusBLAS != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to compute transpose \n");
		return EXIT_FAILURE;
	}

	// Ensure transpose completed
	cudaDeviceSynchronize();

    // Invert the matrix
    cyclicInv(ds, dl, diag, du, dw, inv1Multi, inv2Multi, omega, cCurr, a, b, d, e, BLOCK_INV, BLOCK_X, BLOCK_Y, size, nx);

	// Transpose the result
	statusBLAS = cublasDgeam(handleBLAS, CUBLAS_OP_T, CUBLAS_OP_T, nx, nx, &alpha, cCurr, nx, &beta, NULL, nx, cHalf, nx);

	if (statusBLAS != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to compute transpose \n");
		return EXIT_FAILURE;
	}

	// Ensure transpose completed
	cudaDeviceSynchronize();

    // ----------------------------------------
    // Set up computation of linear RHS - y
    // ---------------------------------------- 

    int yInitHoriz = 5;
	int yInitLeft = 2;
	int yInitRight = 2;

	int yInitVert = 3;
	int yInitTop = 1;
	int yInitBottom = 1;

	double* weightsInitRHSy;
	cudaMallocManaged(&weightsInitRHSy, yInitHoriz * yInitVert * sizeof(double));

	weightsInitRHSy[0] = 0.0; 							weightsInitRHSy[1] = - 2.0 * simgaLin;				weightsInitRHSy[2] = 4.0 * simgaLin;				weightsInitRHSy[3] = - 2.0 * simgaLin;				weightsInitRHSy[4] = 0.0;
	weightsInitRHSy[5] = - simgaLin;					weightsInitRHSy[6] = 8.0 * simgaLin; 				weightsInitRHSy[7] = 1 - 14.0 * simgaLin;			weightsInitRHSy[8] = 8.0 * simgaLin;				weightsInitRHSy[9] = - simgaLin; 			
	weightsInitRHSy[10] = 0.0;							weightsInitRHSy[11] = - 2.0 * simgaLin;				weightsInitRHSy[12] = 4.0 * simgaLin;				weightsInitRHSy[13] = - 2.0 * simgaLin;				weightsInitRHSy[14] = 0.0;

	// Set up the compute device structs
	cuSten_t yInitRHS;

	// Set the number of tiles
	int yInitTiles = 1;

	// Initialise the instance of the stencil
	custenCreate2DXYp(&yInitRHS, computeDevice, yInitTiles, nx, nx, BLOCK_X, BLOCK_Y, cCurr, cHalf, weightsInitRHSy, yInitHoriz, yInitLeft, yInitRight, yInitVert, yInitTop, yInitBottom);

	// Ensure compute type created
	cudaDeviceSynchronize();

    //----------------------------------------
    // Set up computation of non-linear RHS - y
    //----------------------------------------   

	// Set up the compute device structs
	cuSten_t nonLinInitComputeY;

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// Initialise the instance of the stencil
	custenCreate2DXYpFun(&nonLinInitComputeY, computeDevice, nonLinTiles, nx, nx, BLOCK_X_FUN, BLOCK_Y_FUN, cNonLinRHS, cCurr, coe, numStenHoriz, numStenLeft, numStenRight, numStenVert, numStenTop, numStenBottom, func);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

    //----------------------------------------
    // Finish computing the initial step
    //---------------------------------------- 

	// Run the computation - non-linear RHS
	custenCompute2DXYpFun(&nonLinInitComputeX, 0);

	// Run the computation - linear RHS
	custenCompute2DXYp(&yInitRHS, 0);

	// Ensure computation completed
	cudaDeviceSynchronize();

	// Add two vectors together
	statusBLAS  = cublasDaxpy(handleBLAS, nx * nx, &constant, cNonLinRHS, 1, cCurr, 1);
	
	if (statusBLAS != CUBLAS_STATUS_SUCCESS) {
		printf("Unable to add vectors \n");
		return EXIT_FAILURE;
	}

	// Ensure computation completed
	cudaDeviceSynchronize();

    // Invert the matrix
    cyclicInv(ds, dl, diag, du, dw, inv1Multi, inv2Multi, omega, cCurr, a, b, d, e, BLOCK_INV, BLOCK_X, BLOCK_Y, size, nx);

	// Ensure computation completed
	cudaDeviceSynchronize();

    // ----------------------------------------
    // Initial Step Completed
    // ---------------------------------------- 

	// cOld contains step n - 1
	// cCurr contains step n

    // ----------------------------------------
    // Now set up new timestepping
    // ---------------------------------------- 	

    //----------------------------------------
    // Set new coefficients
    //---------------------------------------- 	

	// These will be replaced later for the double step version
    simgaLin = 2.0 * dt * D * gamma / (3.0 * (pow(dx, 4.0)));

    // Set the diagonal elements
    a = simgaLin;
    b = - 4 * simgaLin;
    c = 1 + 6 * simgaLin;
    d = - 4 * simgaLin;
    e = simgaLin;

    //----------------------------------------
    // Set the matrix for the full time steppings
    //----------------------------------------

    // Set the LHS for inversion
    setMultiLHS<<<gridDim, blockDim>>>(ds, dl, diag, du, dw, a, b, c, d, e, size, nx);

    sprintf(msgStringBuffer, "Failed to set LHS matrix for initial timestep");
    checkError(msgStringBuffer);

	// Ensure matrix is set
	cudaDeviceSynchronize();

    // Pre-factor the LHS
    pentFactorBatch<<<gridDimInv, blockDimInv>>>(ds, dl, diag, du, dw, size, nx);

    sprintf(msgStringBuffer, "Failed to pre factor LHS matrix for initial timestep");
    checkError(msgStringBuffer);

	// Ensure matrix is factorised
	cudaDeviceSynchronize();

    //----------------------------------------
    // Find omega and set new inverses
    //----------------------------------------

    findOmega(omega, inv1Single, inv2Single, a, b, c, d, e, nx);

    for (int j = 0; j < size; j++)
    {	
    	temp = j * nx;

    	for (int i = 0; i < nx; i++)
    	{
    		index = temp + i;

    		inv1Multi[index] = inv1Single[j]; 
    		inv2Multi[index] = inv2Single[j];
    	}
    }

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
	cudaMallocManaged(&weightsLinRHS, linHoriz * linVert * sizeof(double));

	weightsLinRHS[0] = 0.0; 						weightsLinRHS[1] = 0.0;								weightsLinRHS[2] = - 1.0 * simgaLin;					weightsLinRHS[3] = 0.0;								weightsLinRHS[4] = 0.0;					
	weightsLinRHS[5] = 0.0;							weightsLinRHS[6] = - 2.0 * simgaLin; 				weightsLinRHS[7] = 8.0 * simgaLin;						weightsLinRHS[8] = - 2.0 * simgaLin;				weightsLinRHS[9] = 0.0; 			
	weightsLinRHS[10] = - 1.0 * simgaLin; 			weightsLinRHS[11] = 8.0 * simgaLin;					weightsLinRHS[12] = - 20.0 * simgaLin;					weightsLinRHS[13] = 8.0 * simgaLin;					weightsLinRHS[14] = - 1.0 * simgaLin;					
	weightsLinRHS[15] = 0.0;						weightsLinRHS[16] = - 2.0 * simgaLin; 				weightsLinRHS[17] = 8.0 * simgaLin;						weightsLinRHS[18] = - 2.0 * simgaLin;				weightsLinRHS[19] = 0.0;
	weightsLinRHS[20] = 0.0; 						weightsLinRHS[21] = 0.0;							weightsLinRHS[22] = -1.0 * simgaLin;					weightsLinRHS[23] = 0.0;							weightsLinRHS[24] = 0.0;					

	// Set up the compute device structs
	cuSten_t linRHS;

	// Set the number of tiles
	int linInitTiles = 1;

	// Initialise the instance of the stencil
	custenCreate2DXYp(&linRHS, computeDevice, linInitTiles, nx, nx, BLOCK_X, BLOCK_Y, cHalf, cBar, weightsLinRHS, linHoriz, linLeft, linRight, linVert, linTop, linBottom);
	
	// Ensure compute type created
	cudaDeviceSynchronize();

    //----------------------------------------
    // Set up computation of non-linear RHS 
    //----------------------------------------   

	// Set up the compute device structs
	cuSten_t nonLinCompute;

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

	// Set new non linear coefficient
	sigmaNonLin = (dt / 3.0) * D * (2.0 / pow(dx, 2.0));

	coe[0] = 0.0;						coe[1] = 1.0 * sigmaNonLin;				coe[2] = 0.0;
	coe[3] = 1.0 * sigmaNonLin;			coe[4] = - 4.0 * sigmaNonLin;			coe[5] = 1.0 * sigmaNonLin;
	coe[6] = 0.0;						coe[7] = 1.0 * sigmaNonLin;				coe[8] = 0.0;

	// Initialise the instance of the stencil
	custenCreate2DXYpFun(&nonLinCompute, computeDevice, nonLinTiles, nx, nx, BLOCK_X_FUN, BLOCK_Y_FUN, cNonLinRHS, cCurr, coe, numStenHoriz, numStenLeft, numStenRight, numStenVert, numStenTop, numStenBottom, func);

	// Synchronise to ensure everything initialised
	cudaDeviceSynchronize();

    //----------------------------------------
    // Begin timestepping
    //----------------------------------------

    double time = dt;
    int timeCount = 1;

    while (time < T)
    {
    	// Set cBar
        findCBar<<<gridDim, blockDim>>>(cOld, cCurr, cBar, nx);

     	// Ensure compute type created
    	cudaDeviceSynchronize();

        // Compute the non-linear RHS
        custenCompute2DXYpFun(&nonLinCompute, 0);

        // Compute the linear RHS
        custenCompute2DXYp(&linRHS, 0);

        // Ensure compute type created
        cudaDeviceSynchronize();

    	// Find the full RHS and then set cOld to cCurrent
    	findRHS<<<gridDim, blockDim>>>(cOld, cCurr, cHalf, cNonLinRHS, nx);

     	// Ensure compute type created
    	cudaDeviceSynchronize();

    	// Transpose the result
    	statusBLAS = cublasDgeam(handleBLAS, CUBLAS_OP_T, CUBLAS_OP_T, nx, nx, &alpha, cHalf, nx, &beta, NULL, nx, cCurr, nx);

    	if (statusBLAS != CUBLAS_STATUS_SUCCESS) {
    		printf("Unable to compute transpose \n");
    		return EXIT_FAILURE;
    	}

    	// Ensure transpose completed
    	cudaDeviceSynchronize();

        // Invert the matrix
        cyclicInv(ds, dl, diag, du, dw, inv1Multi, inv2Multi, omega, cCurr, a, b, d, e, BLOCK_INV, BLOCK_X, BLOCK_Y, size, nx);

    	// Transpose the result
    	statusBLAS = cublasDgeam(handleBLAS, CUBLAS_OP_T, CUBLAS_OP_T, nx, nx, &alpha, cCurr, nx, &beta, NULL, nx, cHalf, nx);

    	if (statusBLAS != CUBLAS_STATUS_SUCCESS) {
    		printf("Unable to compute transpose \n");
    		return EXIT_FAILURE;
    	}

    	// Ensure transpose completed
    	cudaDeviceSynchronize();

        // Invert the matrix
        cyclicInv(ds, dl, diag, du, dw, inv1Multi, inv2Multi, omega, cHalf, a, b, d, e, BLOCK_INV, BLOCK_X, BLOCK_Y, size, nx);

    	// Ensure computation completed
    	cudaDeviceSynchronize();

    	findNew<<<gridDim, blockDim>>>(cCurr, cBar, cHalf, nx);

    	// Ensure computation completed
    	cudaDeviceSynchronize();



        // Add on the next time
        time += dt;
        timeCount += 1;
        printf("%lf \n", time);

        if (timeCount % print == 0)
        {
        	Print_Out(cCurr, time, nx, nx);
        }
    }

   	// Ensure computation completed
	cudaDeviceSynchronize(); 

	Print_Out(cCurr, time, nx, nx);

    //----------------------------------------
    // Free memory at the end
    //----------------------------------------

    free(omega);
    free(inv1Single);
    free(inv2Single);

    custenDestroy2DXYp(&xInitRHS);

    cudaFree(inv1Multi);
    cudaFree(inv2Multi);

    cudaFree(cOld);
    cudaFree(cNonLinRHS);
    cudaFree(cBar);
    cudaFree(cHalf);

    cudaFree(ds);
    cudaFree(dl);
    cudaFree(diag);
    cudaFree(du);
    cudaFree(dw);
}

// ---------------------------------------------------------------------
//  End main program
// ---------------------------------------------------------------------


// ---------------------------------------------------------------------
//  End of File
// ---------------------------------------------------------------------
