CHANGELOG
=========

<details>
<summary>Note: This is in reverse chronological order, so newer entries are added to the top.</summary>

| Contents               |
| :--------------------- |
| [cuSten 2.1](#cuSten-21) |
| [cuSten 2.0](#cuSten-20) |
| [cuSten 1.5](#cuSten-15) |
| [cuSten 1.4](#cuSten-14) |
| [cuSten 1.3](#cuSten-13) |
| [cuSten 1.2.1](#cuSten-121) |
| [cuSten 1.2](#cuSten-12) |
| [cuSten 1.1](#cuSten-11) |
| [cuSten 1.0](#cuSten-10) |


</details>

cuSten 2.1
---------

* Updated naming convention to consistent cuSten for all functions and types
* Replaced the 0 and 1 for loading/unloading data with MACROS HOST and DEVICE to specify where the programmer desires the data to be stored
* Restructured cuPentCahnADI to use a better guess for the initial n - 1 time step, scheme now more stable.
* Added serial example of the same for GPU versus CPU timing benchmarks.
* doxygen updated to reflect above changes

cuSten 2.0
---------

* Swap functions added
* cuPentCahnADI added as an example solver
* Documentation completed with doxygen

cuSten 1.5
---------

* Add 2D_xy_np and 2D_xy_np_fun
* 2D cases are now completed, next release will have formalised documentation and improved code commenting.

cuSten 1.4
---------

* Add 2D_y_np_fun
* Fixed some bugs in other code
* Fixed makefile for examples

cuSten 1.3
---------

* Restructured into static library with its own Makefile and moved examples into separate directory with their own Makefile that calls library.
* Updated readme.

cuSten 1.2.1
---------

* Fixed a bug in 2D xy periodic derivatives, corners of shared memory blocks copied incorrectly

cuSten 1.2
---------

* Added 2D XY periodic derivatives with and without user defined functions.

cuSten 1.1
---------

* Added the ability to do non periodic y direction derivatives in 2D.
* Removed user set stream quantity as it was pointless.

cuSten 1.0
---------

* Initial release of cuSten.
