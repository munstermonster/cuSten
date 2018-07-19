# Andrew Gloster
# May 2018
# Makefile for cuSten

NVCC = nvcc
NVFLAGS = -O3 -lineinfo --cudart=static -arch=compute_61 -code=compute_61 -std=c++11 -dc
FNVFLAGS = -O3 -lineinfo --cudart=static -arch=compute_61 -code=compute_61 -std=c++11

MAIN = 2d_x_p 2d_x_np 2d_x_p_fun 2d_x_np_fun 2d_y_p 2d_y_p_fun
MAINOBJ = 2d_x_p.o 2d_x_np.o 2d_x_p_fun.o 2d_x_np_fun.o 2d_y_p.o 2d_y_p_fun.o

STRUCTDIR = cuSten/structs/
STRUCTOBJ = custenCreateDestroy2DXnp.o custenCreateDestroy2DXp.o custenCreateDestroy2DXpFun.o custenCreateDestroy2DXnpFun.o custenCreateDestroy2DYp.o custenCreateDestroy2DYpFun.o
STRUCTTAR = $(addprefix $(STRUCTDIR), $(STRUCTOBJ))

UTILDIR = cuSten/util/
UTILOBJ = error.o
UTILTAR = $(addprefix $(UTILDIR), $(UTILOBJ))

KERNELDIR = cuSten/kernels/
KERNELOBJ = 2d_x_np_kernel.o 2d_x_p_kernel.o 2d_x_p_fun_kernel.o 2d_x_np_fun_kernel.o 2d_y_p_kernel.o 2d_y_p_fun_kernel.o
KERNELTAR = $(addprefix $(KERNELDIR), $(KERNELOBJ))

# ----------------------
# Possible functions
# ---------------------

# Make everything
all: $(MAIN)

# 2D x periodic
2d_x_p: 2d_x_p.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
	$(NVCC) -o 2d_x_p $^ $(FNVFLAGS)

2d_x_p.o: 2d_x_p.cu
	$(NVCC) $(NVFLAGS) -c 2d_x_p.cu

# 2D x non periodic
2d_x_np: 2d_x_np.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
	$(NVCC) -o 2d_x_np $^ $(FNVFLAGS)

2d_x_np.o: 2d_x_np.cu
	$(NVCC) $(NVFLAGS) -c 2d_x_np.cu

# 2D x periodic function
2d_x_p_fun: 2d_x_p_fun.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
	$(NVCC) -o 2d_x_p_fun $^ $(FNVFLAGS)

2d_x_p_fun.o: 2d_x_p_fun.cu
	$(NVCC) $(NVFLAGS) -c 2d_x_p_fun.cu

# 2D x non periodic function
2d_x_np_fun: 2d_x_np_fun.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
	$(NVCC) -o 2d_x_np_fun $^ $(FNVFLAGS)

2d_x_np_fun.o: 2d_x_np_fun.cu
	$(NVCC) $(NVFLAGS) -c 2d_x_np_fun.cu

# 2D y periodic 
2d_y_p: 2d_y_p.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
	$(NVCC) -o 2d_y_p $^ $(FNVFLAGS)

2d_y_p.o: 2d_y_p.cu
	$(NVCC) $(NVFLAGS) -c 2d_y_p.cu

# 2D y periodic function
2d_y_p_fun: 2d_y_p_fun.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
	$(NVCC) -o 2d_y_p_fun $^ $(FNVFLAGS)

2d_y_p_fun.o: 2d_y_p_fun.cu
	$(NVCC) $(NVFLAGS) -c 2d_y_p_fun.cu

# ----------------------
# Library Functions
# ---------------------

$(STRUCTTAR): %.o: %.cu
	$(NVCC) -c $(NVFLAGS) $< -o $@

$(KERNELTAR): %.o: %.cu
	$(NVCC) -c $(NVFLAGS) $< -o $@

$(UTILTAR): %.o: %.cu
	$(NVCC) -c $(NVFLAGS) $< -o $@

# ----------------------
# Remove everything
# ---------------------

clean:
	rm -f $(STRUCTTAR) $(UTILTAR) $(KERNELTAR) $(MAINOBJ) $(MAIN)


