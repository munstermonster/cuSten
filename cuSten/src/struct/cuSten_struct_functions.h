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

/*! \fun void cuStenCreate2DXnp
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

template <typename elemType>
void cuStenCreate2DXnp(
	cuSten_t<elemType>* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	elemType* dataOutput,
	elemType* dataInput,
	elemType* weights,
	int numSten,
	int numStenLeft,
	int numStenRight
);

/*! \fun void cuStenSwap2DXnp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DXnp(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DXnp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DXnp(
	cuSten_t<elemType>* pt_cuSten
);

// ----------------------------------------
// 2D x direction non periodic - user function
// ----------------------------------------

/*! \fun void cuStenCreate2DXnpFun
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

template <typename elemType>
void cuStenCreate2DXnpFun(
	cuSten_t<elemType>* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	elemType* dateOutput,
	elemType* dateInput,
	elemType* coe,
	int numSten,
	int numStenLeft,
	int numStenRight,
	int numCoe,
	elemType* func
);

/*! \fun void cuStenSwap2DXnpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DXnpFun(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DXnpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DXnpFun(
	cuSten_t<elemType>* pt_cuSten
);

// ----------------------------------------
// 2D x direction periodic
// ----------------------------------------

/*! \fun void cuStenCreate2DXp
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

template <typename elemType>
void cuStenCreate2DXp(
	cuSten_t<elemType>* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	elemType* dataNew,
	elemType* dataOld,
	elemType* weights,
	int numSten,
	int numStenLeft,
	int numStenRight
);

/*! \fun void cuStenSwap2DXp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DXp(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DXnp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DXp(
	cuSten_t<elemType>* pt_cuSten
);

// ----------------------------------------
// 2D x direction periodic - user function
// ----------------------------------------

/*! \fun void cuStenCreate2DXpFun
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

template <typename elemType>
void cuStenCreate2DXpFun(
	cuSten_t<elemType>* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	elemType* dataNew,
	elemType* dataOld,
	elemType* coe,
	int numSten,
	int numStenLeft,
	int numStenRight,
	int numCoe,
	elemType* func
);

/*! \fun void cuStenSwap2DXpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DXpFun(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DXpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DXpFun(
	cuSten_t<elemType>* pt_cuSten
);

// ----------------------------------------
// 2D xy direction periodic WENO
// ----------------------------------------

/*! \fun void cuStenCreate2DXYWENOADVp
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

template <typename elemType>
void cuStenCreate2DXYWENOADVp(
	cuSten_t<elemType>* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	elemType dx,
	elemType dy,
	elemType* u,
	elemType* v,
	elemType* dataOutput,
	elemType* dataInput
);

/*! \fun void cuStenSwap2DXYWENOADVp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DXYWENOADVp(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DXYWENOADVp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DXYWENOADVp
(
	cuSten_t<elemType>* pt_cuSten
);

// ----------------------------------------
// 2D xy direction non periodic 
// ----------------------------------------

/*! \fun void cuStenCreate2DXYnp
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

template <typename elemType>
void cuStenCreate2DXYnp(
	cuSten_t<elemType>* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	elemType* dataOutput,
	elemType* dataInput,
	elemType* weights,
	int numStenHoriz,
	int numStenLeft,
	int numStenRight,
	int numStenVert,
	int numStenTop,
	int numStenBottom
);

/*! \fun void cuStenSwap2DXYnp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DXYnp(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DXYnp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DXYnp(
	cuSten_t<elemType>* pt_cuSten
);

// ----------------------------------------
// 2D xy direction non periodic - user function
// ----------------------------------------

/*! \fun void cuStenCreate2DXYnpFun
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

template <typename elemType>
void cuStenCreate2DXYnpFun(
	cuSten_t<elemType>* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	elemType* dataNew,
	elemType* dataOld,
	elemType* coe,
	int numStenHoriz,
	int numStenLeft,
	int numStenRight,
	int numStenVert,
	int numStenTop,
	int numStenBottom,
	elemType* func
);

/*! \fun void cuStenSwap2DXYnpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DXYnpFun(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DXYnpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DXYnpFun(
	cuSten_t<elemType>* pt_cuSten
);


// ----------------------------------------
// 2D xy direction periodic 
// ----------------------------------------

/*! \fun void cuStenCreate2DXYp
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

template <typename elemType>
void cuStenCreate2DXYp(
	cuSten_t<elemType>* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	elemType* dataNew,
	elemType* dataOld,
	elemType* weights,
	int numStenHoriz,
	int numStenLeft,
	int numStenRight,
	int numStenVert,
	int numStenTop,
	int numStenBottom
);

/*! \fun void cuStenSwap2DYp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DXYp(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DXYp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DXYp(
	cuSten_t<elemType>* pt_cuSten
);

// ----------------------------------------
// 2D xy direction periodic - user function 
// ----------------------------------------

/*! \fun void cuStenCreate2DXYpFun
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

template <typename elemType>
void cuStenCreate2DXYpFun(
	cuSten_t<elemType>* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	elemType* dataOutput,
	elemType* dataInput,
	elemType* coe,
	int numStenHoriz,
	int numStenLeft,
	int numStenRight,
	int numStenVert,
	int numStenTop,
	int numStenBottom,
	elemType* func
);

/*! \fun void cuStenSwap2DYpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DXYpFun(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DXYpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DXYpFun(
	cuSten_t<elemType>* pt_cuSten
);


// ----------------------------------------
// 2D y direction non periodic
// ----------------------------------------

/*! \fun void cuStenCreate2DYnp
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

template <typename elemType>
void cuStenCreate2DYnp(
	cuSten_t<elemType>* pt_cuSten,		
	int deviceNum,				
	int numTiles,				
	int nx,				
	int ny,				
	int BLOCK_X,			
	int BLOCK_Y,				
	elemType* dataOutput,			
	elemType* dataInput,			
	elemType* weights,			
	int numSten,				
	int numStenTop,				
	int numStenBottom			
);

/*! \fun void cuStenSwap2DYnp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DYnp(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DYnp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DYnp(
	cuSten_t<elemType>* pt_cuSten
);

// ----------------------------------------
// 2D y direction non periodic - user function
// ----------------------------------------

/*! \fun void cuStenCreate2DYnpFun
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

template <typename elemType>
void cuStenCreate2DYnpFun(
	cuSten_t<elemType>* pt_cuSten,		
	int deviceNum,				
	int numTiles,				
	int nx,				
	int ny,				
	int BLOCK_X,				
	int BLOCK_Y,				
	elemType* dataOutput,			
	elemType* dataInput,			
	elemType* coe,				
	int numSten,				
	int numStenTop,				
	int numStenBottom,			
	elemType* func 			
);

/*! \fun void cuStenSwap2DYnpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DYnpFun(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DYnpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DYnpFun(
	cuSten_t<elemType>* pt_cuSten
);

// ----------------------------------------
// 2D y direction periodic
// ----------------------------------------

/*! \fun void cuStenCreate2DYp
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

template <typename elemType>
void cuStenCreate2DYp(
	cuSten_t<elemType>* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	elemType* dataOutput,
	elemType* dataInput,
	elemType* weights,
	int numSten,
	int numStenTop,
	int numStenBottom
);

/*! \fun void cuStenSwap2DYp
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DYp(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DYp
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DYp(
	cuSten_t<elemType>* pt_cuSten
);

// ----------------------------------------
// 2D y direction periodic - user function
// ----------------------------------------

/*! \fun void cuStenCreate2DYnpFun
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

template <typename elemType>
void cuStenCreate2DYpFun(
	cuSten_t<elemType>* pt_cuSten,
	int deviceNum,
	int numTiles,
	int nx,
	int ny,
	int BLOCK_X,
	int BLOCK_Y,
	elemType* dataOutput,
	elemType* dataInput,
	elemType* coe,
	int numSten,
	int numStenTop,
	int numStenBottom,
	int numCoe,
	elemType* func	
);

/*! \fun void cuStenSwap2DYpFun
    \brief Function to swap pointers necessary for timestepping
    \param pt_cuSten Pointer to cuSten type provided by user
	\param dataInput Pointer to data input to the on the next compute
*/

template <typename elemType>
void cuStenSwap2DYpFun(
	cuSten_t<elemType>* pt_cuSten,
	elemType* dataInput
);

/*! \fun void cuStenDestroy2DYnpFun
    \brief Function to destroy data associated with cuSten_t
    \param pt_cuSten Pointer to cuSten type provided by user
*/

template <typename elemType>
void cuStenDestroy2DYpFun(
	cuSten_t<elemType>* pt_cuSten
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
