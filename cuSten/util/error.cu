// Andrew Gloster
// May 2018
// Functions to catch errors in the cuSten library

// ---------------------------------------------------------------------
//  Standard Libraries and Headers
// ---------------------------------------------------------------------

#include <iostream>

// ---------------------------------------------------------------------
// Custom libraries and headers
// ---------------------------------------------------------------------

#include "util.h"

// ---------------------------------------------------------------------
//  Error checking function
// ---------------------------------------------------------------------

void checkError (const char* action) 
{
  
  cudaError_t error;
  error = cudaGetLastError(); 

  if (error != cudaSuccess) {
    printf ("\nError while '%s': %s\nprogram terminated ...\n\n", action, cudaGetErrorString(error));
    exit (EXIT_FAILURE);
  }
}

// ---------------------------------------------------------------------
// End of file
// ---------------------------------------------------------------------