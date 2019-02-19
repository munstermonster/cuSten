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
 * @file cuSten.h
 * Main header for cuSten library
 */

// ---------------------------------------------------------------------
// Define Header
// ---------------------------------------------------------------------

#ifndef CUSTEN_H
#define CUSTEN_H

// ---------------------------------------------------------------------
// Include headers from cuSten library
// ---------------------------------------------------------------------

#include "src/struct/cuSten_struct_type.h"
#include "src/struct/cuSten_struct_functions.h"
#include "src/kernels/stencil_kernels.h"
#include "src/util/util.h"

#endif

/*! \mainpage cuSten - CUDA Finite Difference Library
 *
 * \section intro_sec Introduction
 *
 * This is the documentation for the cuSten library. For compiling please see the README in the top level directory.
 *
 * All details of the library API can be found in src/struct/cuSten_struct_functions.h and src/kernels/stencil_kernels.h
 *
 * For usage examples please see either the examples folder or the cuCahnPentADI, both of which are supplied when downloading this repository from https://github.com/munstermonster/cuSten
 */