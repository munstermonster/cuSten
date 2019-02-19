// Andrew Gloster
// May 2018

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


/**
 * @file cuSten_struct_functions.h
 * Header file with function declarations for Create, Destroy and Swap
 */

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



// ---------------------------------------------------------------------
//  Header file functions
// ---------------------------------------------------------------------

// ----------------------------------------
// 2D x direction non periodic
// ----------------------------------------

/*! \fun __global__ void custenCreate2DXnp
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param weights Pointer to input weights for each point in the stencil
	\param numSten Total number of points in the stencil
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
*/

void custenCreate2DXnp(
	cuSten_t* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	double* dataOutput,
	double* dataInput,
	double* weights,
	int numSten,
	int numStenLeft,
	int numStenRight
);

/*! \fun __global__ void custenSwap2DXnp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DXnp(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DXnp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DXnp(
	cuSten_t* pt_cuSten
);

// ----------------------------------------
// 2D x direction non periodic - user function
// ----------------------------------------

/*! \fun __global__ void custenCreate2DXnpFun
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param numSten Total number of points in the stencil
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
	\param Number of coefficients used by the user in their function
	\param Pointer to user function
*/

void custenCreate2DXnpFun(
	cuSten_t* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	double* dateOutput,
	double* dateInput,
	double* coe,
	int numSten,
	int numStenLeft,
	int numStenRight,
	int numCoe,
	double* func
);

/*! \fun __global__ void custenSwap2DXnpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DXnpFun(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DXnpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DXnpFun(
	cuSten_t* pt_cuSten
);

// ----------------------------------------
// 2D x direction periodic
// ----------------------------------------

/*! \fun __global__ void custenCreate2DXp
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param weights Pointer to input weights for each point in the stencil
	\param numSten Total number of points in the stencil
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
*/

void custenCreate2DXp(
	cuSten_t* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	double* dataNew,
	double* dataOld,
	double* weights,
	int numSten,
	int numStenLeft,
	int numStenRight
);

/*! \fun __global__ void custenSwap2DXp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DXp(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DXnp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DXp(
	cuSten_t* pt_cuSten
);

// ----------------------------------------
// 2D x direction periodic - user function
// ----------------------------------------

/*! \fun __global__ void custenCreate2DXpFun
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param numSten Total number of points in the stencil
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
	\param Number of coefficients used by the user in their function
	\param Pointer to user function
*/

void custenCreate2DXpFun(
	cuSten_t* pt_cuSten,

	int deviceNum,

	int numTiles,

	int nx,
	int ny,

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

/*! \fun __global__ void custenSwap2DXpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DXpFun(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DXpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DXpFun(
	cuSten_t* pt_cuSten
);

// ----------------------------------------
// 2D xy direction periodic WENO
// ----------------------------------------

/*! \fun __global__ void custenCreate2DXYWENOADVp
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
	\param dx Spacing of grid in x direction
	\param dy Spacing of grid in y direction
	\param u Pointer to u velocity data
	\param v Pointer to v velocity data
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
*/

void custenCreate2DXYWENOADVp
(
	cuSten_t* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	double dx,
	double dy,
	double* u,
	double* v,
	double* dataOutput,
	double* dataInput
);

/*! \fun __global__ void custenSwap2DXYWENOADVp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DXYWENOADVp(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DXYWENOADVp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DXYWENOADVp
(
	cuSten_t* pt_cuSten
);

// ----------------------------------------
// 2D xy direction non periodic 
// ----------------------------------------

/*! \fun __global__ void custenCreate2DXYnp
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param weights Pointer to input weights for each point in the stencil
	\param numStenHoriz Total number of points in the stencil in the x direction
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
	\param numStenHoriz Total number of points in the stencil in the y direction
	\param numStenTop Number of points on the top of the stencil
	\param numStenBottom Number of points on the bottom of the stencil
*/

void custenCreate2DXYnp(
	cuSten_t* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	double* dataOutput,
	double* dataInput,
	double* weights,
	int numStenHoriz,
	int numStenLeft,
	int numStenRight,
	int numStenVert,
	int numStenTop,
	int numStenBottom
);

/*! \fun __global__ void custenSwap2DXYnp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DXYnp(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DXYnp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DXYnp(
	cuSten_t* pt_cuSten
);

// ----------------------------------------
// 2D xy direction non periodic - user function
// ----------------------------------------

/*! \fun __global__ void custenCreate2DXYnpFun
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param coe Pointer to input coefficients for the user function
	\param numStenHoriz Total number of points in the stencil in the x direction
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
	\param numStenHoriz Total number of points in the stencil in the y direction
	\param numStenTop Number of points on the top of the stencil
	\param numStenBottom Number of points on the bottom of the stencil
	\param Pointer to user function
*/

void custenCreate2DXYnpFun(
	cuSten_t* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	double* dataNew,
	double* dataOld,
	double* coe,
	int numStenHoriz,
	int numStenLeft,
	int numStenRight,
	int numStenVert,
	int numStenTop,
	int numStenBottom,
	double* func
);

/*! \fun __global__ void custenSwap2DXYnpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DXYnpFun(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DXYnpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DXYnpFun(
	cuSten_t* pt_cuSten
);


// ----------------------------------------
// 2D xy direction periodic 
// ----------------------------------------

/*! \fun __global__ void custenCreate2DXYp
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param weights Pointer to input weights for each point in the stencil
	\param numStenHoriz Total number of points in the stencil in the x direction
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
	\param numStenHoriz Total number of points in the stencil in the y direction
	\param numStenTop Number of points on the top of the stencil
	\param numStenBottom Number of points on the bottom of the stencil
*/

void custenCreate2DXYp(
	cuSten_t* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	double* dataNew,
	double* dataOld,
	double* weights,
	int numStenHoriz,
	int numStenLeft,
	int numStenRight,
	int numStenVert,
	int numStenTop,
	int numStenBottom
);

/*! \fun __global__ void custenSwap2DYp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DXYp(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DXYp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DXYp(
	cuSten_t* pt_cuSten
);

// ----------------------------------------
// 2D xy direction periodic - user function 
// ----------------------------------------

/*! \fun __global__ void custenCreate2DXYpFun
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param coe Pointer to input coefficients for the user function
	\param numStenHoriz Total number of points in the stencil in the x direction
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
	\param numStenHoriz Total number of points in the stencil in the y direction
	\param numStenTop Number of points on the top of the stencil
	\param numStenBottom Number of points on the bottom of the stencil
	\param Pointer to user function
*/

void custenCreate2DXYpFun(
	cuSten_t* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	double* dataOutput,
	double* dataInput,
	double* coe,
	int numStenHoriz,
	int numStenLeft,
	int numStenRight,
	int numStenVert,
	int numStenTop,
	int numStenBottom,
	double* func
);

/*! \fun __global__ void custenSwap2DYpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DXYpFun(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DXYpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DXYpFun(
	cuSten_t* pt_cuSten
);


// ----------------------------------------
// 2D y direction non periodic
// ----------------------------------------

/*! \fun __global__ void custenCreate2DYnp
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param weights Pointer to the weights for the stencil
	\param numStenSten Total number of points in the stencil in the y direction
	\param numStenTop Number of points on the top of the stencil
	\param numStenBottom Number of points on the bottom of the stencil
*/

void custenCreate2DYnp(
	cuSten_t* pt_cuSten,		
	int deviceNum,				
	int numTiles,				
	int nx,				
	int ny,				
	int BLOCK_X,			
	int BLOCK_Y,				
	double* dataOutput,			
	double* dataInput,			
	double* weights,			
	int numSten,				
	int numStenTop,				
	int numStenBottom			
);

/*! \fun __global__ void custenSwap2DYnp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DYnp(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DYnp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DYnp(
	cuSten_t* pt_cuSten
);

// ----------------------------------------
// 2D y direction non periodic - user function
// ----------------------------------------

/*! \fun __global__ void custenCreate2DYnpFun
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param coe Pointer to input coefficients for the user function
	\param numSten Total number of points in the stencil
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
	\param Pointer to user function
*/

void custenCreate2DYnpFun(
	cuSten_t* pt_cuSten,		
	int deviceNum,				
	int numTiles,				
	int nx,				
	int ny,				
	int BLOCK_X,				
	int BLOCK_Y,				
	double* dataOutput,			
	double* dataInput,			
	double* coe,				
	int numSten,				
	int numStenTop,				
	int numStenBottom,			
	double* func 			
);

/*! \fun __global__ void custenSwap2DYnpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DYnpFun(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DYnpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DYnpFun(
	cuSten_t* pt_cuSten
);

// ----------------------------------------
// 2D y direction periodic
// ----------------------------------------

/*! \fun __global__ void custenCreate2DYp
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param weights Pointer to the weights for the stencil
	\param numStenSten Total number of points in the stencil in the y direction
	\param numStenTop Number of points on the top of the stencil
	\param numStenBottom Number of points on the bottom of the stencil
*/

void custenCreate2DYp(
	cuSten_t* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	double* dataOutput,
	double* dataInput,
	double* weights,
	int numSten,
	int numStenTop,
	int numStenBottom
);

/*! \fun __global__ void custenSwap2DYp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DYp(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DYp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

void custenDestroy2DYp(
	cuSten_t* pt_cuSten
);

// ----------------------------------------
// 2D y direction periodic - user function
// ----------------------------------------

/*! \fun __global__ void custenCreate2DYnpFun
    \brief Function to set up cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
	\param numTiles Number of tiles to divide the data into
	\param nx Total number of points in the x direction 
	\param ny Total number of points in the y direction 
	\param BLOCK_X Size of thread block in the x direction
	\param BLOCK_Y Size of thread block in the y direction
    \param dataOutput Pointer to data output by the function
	\param dataInput Pointer to data input to the function    
	\param coe Pointer to input coefficients for the user function
	\param numSten Total number of points in the stencil
	\param numStenLeft Number of points on the left side of the stencil
	\param numStenRight Number of points on the right side of the stencil
	\param Number of coefficients user in user function
	\param Pointer to user function
*/

void custenCreate2DYpFun(
	cuSten_t* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	double* dataOutput,
	double* dataInput,
	double* coe,
	int numSten,
	int numStenTop,
	int numStenBottom,
	int numCoe,
	double* func	
);

/*! \fun __global__ void custenSwap2DYpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

void custenSwap2DYpFun(
	cuSten_t* pt_cuSten,

	double* dataInput
);

/*! \fun __global__ void custenDestroy2DYnpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

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
