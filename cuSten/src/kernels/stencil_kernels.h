// Andrew Gloster
// February 2019
// Header file for cuSten kernels

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
 * @file stencil_kernels.h
 * Header file for cuSten kernels
 */

// ---------------------------------------------------------------------
// Define Header
// ---------------------------------------------------------------------

#ifndef FINITE_H
#define FINITE_H

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

template <typename elemType>
void cuStenCompute2DXnp
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D x direction periodic
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DXp
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D x direction non periodic - user function
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DXnpFun
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D x direction periodic - user function
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DXpFun
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D y direction periodic
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DYp
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D y direction periodic - user function
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DYpFun
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D y direction non periodic
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DYnp
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D y direction non periodic - user function
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DYnpFun
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D xy direction periodic
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DXYp
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D xy direction periodic - user function
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DXYpFun
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D xy direction non periodic
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DXYnp
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D xy direction non periodic - user function
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DXYnpFun 
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
);

// ----------------------------------------
// 2D xy WENO periodic
// ----------------------------------------

template <typename elemType>
void cuStenCompute2DXYWENOADVp
(
	cuSten_t<elemType>* pt_cuSten,
	bool offload
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
