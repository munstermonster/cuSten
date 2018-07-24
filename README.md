# cuSten - CUDA Finite Difference Library

** Welcome to cuSten **
cuSten is a CUDA library under development by Andrew Gloster at University College Dublin. The idea behind the library is to simplify the development of code where we wish to repeatedly apply a stencil operation, for example a finite difference stencil, to a block of data. The user of this library simply has to provide basic details such as the stencil weights and stencil size to the library and then cuSten will handle the rest. Including calling the kernel with the correct block/thread structure, assign local memory, load the data on/off the device asynchronously.

# Getting started
This library is still under development but examples of how to use it are included in the top level directory. The included make file will compile these examples.

The naming convention is as follows

dimension_direction_periodicity.cu

dimension = dimension of the data 2d or 3d (3d under developent)
direction = List of directions in which the stencil will be applied, for example an x direction in 2D will apply the stencil in only the x direction
periodicity = Apply the stencil periodically to the data or not, p for periodic, np for non-periodic

The extra 'fun' seen on some files are examples of how the user can use a function pointer to implement a broader range of operations on their data, such as raising to a power, source terms etc.

# Where is cuSten used
cuSten is currently used in the following academic papers:

- cuPentBatch -- A batched pentadiagonal solver for NVIDIA GPUs (preprint: https://arxiv.org/abs/1807.07382)




