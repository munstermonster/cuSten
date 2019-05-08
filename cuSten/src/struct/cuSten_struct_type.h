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
 * @file cuSten_struct_type.h
 * Detailing the stuct used in the cuSten library.
 */

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

 /** @struct cuSten_t
*  Struct used in the cuSten library
*  @var deviceNum The device to run the computation on.
*  @var numStreams Number of streams to be used for loading and unloading etc.
*  @var numTiles Number of tiles that the domain is to be divided into.
*  @var nx Total number of points in x.
*  @var ny Total number of points in y.
*  @var nyTile Number of point in y direction on a tile.
*  @var numSten  Number of points in the stencil (or total points when cross derivative)
*  @var deviceNum The device to run the computation on.
*  @var numStenLeft Number of points on the left side of the stencil
*  @var numStenRight Number of points on the right side of the stencil
*  @var numStenTop Number of points to the top in the stencil
*  @var numStenBottom Number of points to the bottom in the stencil
*  @var numStenHoriz Number of points in a horizontal stencil
*  @var numStenVert Number of points in a vertical stencil
*  @var BLOCK_X Number of threads in x
*  @var BLOCK_Y Number of threads in y
*  @var xGrid Size of grid in x
*  @var yGrid Size of grid in y
*  @var mem_shared Amount of shared memory required
*  @var dataInput Pointers for the input data
*  @var dataOutput Pointers for the output data
*  @var uVel Pointers for the u velocity data
*  @var vVel Pointers for the v velocity data
*  @var weights Pointer for the device weights data
*  @var coe  Pointer for the device coefficient data
*  @var coeDx  Coefficient for WENO - x 
*  @var coeDy  Coefficient for WENO - y 
*  @var numCoe  Number of coefficient in function pointer
*  @var nxLocal Number of points in shared memory in x
*  @var nyLocal Number of points in shared memory in y
*  @var boundaryTop Pointers to data for top of tile boundaries
*  @var boundaryBottom Pointers to data for bottom of tile boundaries
*  @var numBoundaryTop Number of points in a top boundary
*  @var numBoundaryBottom Number of points in a bottom boundary
*  @var streams Pointers to streams used for computing
*  @var events Pointers to events used for computing
*  @var devFunc Pointer to user defined function pointer
*/

template <typename elemType>
struct cuSten_t
{
    int deviceNum;
    int numStreams;
    int numTiles;
	int nx;
 	int ny;
 	int nyTile;
	int numSten;
	int numStenLeft;
	int numStenRight;
	int numStenTop;
	int numStenBottom;
	int numStenHoriz;
	int numStenVert;
	int BLOCK_X;
	int BLOCK_Y; 
	int xGrid;
	int yGrid;
	int mem_shared; 
	elemType** dataInput;
	elemType** dataOutput;
	elemType** uVel;
	elemType** vVel;
	elemType* weights;
	elemType* coe;
	elemType coeDx;
	elemType coeDy;
	int numCoe;
	int nxLocal;
	int nyLocal;	
	elemType** boundaryTop;
	elemType** boundaryBottom;
	int numBoundaryTop;
	int numBoundaryBottom;
	cudaStream_t* streams;
	cudaEvent_t* events;
	elemType* devFunc;
};

#endif
