# cuSten - CUDA Finite Difference Library

** Welcome to cuSten **
cuSten is a CUDA library under development by Andrew Gloster at University College Dublin. The idea behind the library is to simplify the development of code where we wish to apply stencil operations, for example a finite difference stencil, to a block of data in parallel. The user of this library simply has to provide basic details such as the stencil weights and stencil size to the library and then cuSten will handle the rest. Including calling the kernel with the correct block/thread structure, assign local memory and loading the data on/off the device asynchronously.

# Getting started
Examples of how to use the library are found in examples/src. See section below for how to compile.

The naming convention is as follows

dimension_direction_periodicity.cu

dimension = dimension of the data 2d
direction = List of directions in which the stencil will be applied, for example an x direction in 2D will apply the stencil in only the x direction
periodicity = Apply the stencil periodically to the data or not, p for periodic, np for non-periodic

The extra 'fun' seen on some files are examples of how the user can use a function pointer to implement a broader range of operations on their data, such as raising to a power, source terms etc.

# Compiling
Use the Makefile in the cuSten/ directory followed by the examples/ directory. In order to use the library in your own code simply include the cuSten.h header and link to the static library stored in cuSten/lib. The compiled examples are found in /examples/bin after compiling.

In addition to this a sample Cahn-Hilliard solver can be compiled by running the Makefile in cuPentCahnADI. This requires the HDF5 library to be installed and linked to the correct path in the Makefile. Several python scripts are also included to analyise the output.

The Makefiles used in this project assume the CUDA compiler nvcc is installed and the parameter --gpu-architecture=sm_61 is set, this may need to be changed appropiately for other generations of NVIDIA GPUs, this is user dependant so we reccomend checking the correct setting of this parameter for your system.

# Documentation
Run the command make doc in the cuSten directory.

# Where is cuSten used
cuSten is currently used in the following academic papers:

- cuPentBatch -- A batched pentadiagonal solver for NVIDIA GPUs (preprint: https://arxiv.org/abs/1807.07382, published article: https://www.sciencedirect.com/science/article/pii/S0010465519300979)
- cuSten -- CUDA Finite Difference and Stencil Library (preprint: https://arxiv.org/abs/1902.09931)
- Efficient Interleaved Batch Matrix Solvers for CUDA (preprint: https://arxiv.org/abs/1909.04539)



