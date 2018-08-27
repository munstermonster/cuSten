// Andrew Gloster
// May 2018
// Header file for user stencil functions

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
// Define Header
// ---------------------------------------------------------------------

#ifndef DEVFUN_H
#define DEVFUN_H

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------



// ---------------------------------------------------------------------
//  Programmer Libraries and Headers
// ---------------------------------------------------------------------




// ---------------------------------------------------------------------
//  Header file functions
// ---------------------------------------------------------------------

// Data -- Coefficients -- Current node index
typedef double (*devArg1X)(double*, double*, int);

// Data -- Coefficients -- Current node index -- Jump
typedef double (*devArg1Y)(double*, double*, int, int);

// Data -- Coefficients -- Current node index -- Jump -- Points in x -- Points in y
typedef double (*devArg1XY)(double*, double*, int, int, int, int);

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
