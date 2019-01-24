# Andrew Gloster
# August 2018
# Makefile for cuSten

#   Copyright 2018 Andrew Gloster

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


NVCC = nvcc
NVFLAGS = -O3 -lineinfo --cudart=static -arch=compute_61 -code=compute_61 -std=c++11 -dc
FNVFLAGS = -O3 -lineinfo --cudart=static -arch=compute_61 -code=compute_61 -std=c++11

MAIN = 2d_x_p 2d_x_np 2d_x_p_fun 2d_x_np_fun 2d_y_p 2d_y_p_fun 2d_y_np 2d_xy_p 2d_xy_p_fun 2d_xyWENOADV_p
MAINOBJ = 2d_x_p.o 2d_x_np.o 2d_x_p_fun.o 2d_x_np_fun.o 2d_y_p.o 2d_y_p_fun.o 2d_y_np.o 2d_xy_p.o 2d_xy_p_fun.o 2d_xyWENOADV_p.o

STRUCTDIR = cuSten/structs/
STRUCTOBJ = custenCreateDestroy2DXnp.o custenCreateDestroy2DXp.o custenCreateDestroy2DXpFun.o custenCreateDestroy2DXnpFun.o custenCreateDestroy2DYp.o custenCreateDestroy2DYpFun.o custenCreateDestroy2DYnp.o custenCreateDestroy2DXYp.o custenCreateDestroy2DXYpFun.o custenCreateDestroy2DXYADVWENOp.o
STRUCTTAR = $(addprefix $(STRUCTDIR), $(STRUCTOBJ))

UTILDIR = cuSten/util/
UTILOBJ = error.o
UTILTAR = $(addprefix $(UTILDIR), $(UTILOBJ))

KERNELDIR = cuSten/kernels/
KERNELOBJ = 2d_x_np_kernel.o 2d_x_p_kernel.o 2d_x_p_fun_kernel.o 2d_x_np_fun_kernel.o 2d_y_p_kernel.o 2d_y_p_fun_kernel.o 2d_y_np_kernel.o 2d_xy_p_kernel.o 2d_xy_p_fun_kernel.o 2d_xyADVWENO_p_kernel.o
KERNELTAR = $(addprefix $(KERNELDIR), $(KERNELOBJ))

OBJDIR = obj

# ----------------------
# Possible functions
# ---------------------

# Make everything
# all: $(OBJDIR)/%.o

# bin/$(MAIN):$(OBJDIR)/%.o 
# 	$(NVCC) -o bin/$(MAIN) $^ $(FNVFLAGS)

# $(OBJDIR)/%.o: %.cu $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
# 	$(NVCC) $(NVFLAGS) -c -o $@ $<
# # 2D x periodic
# 2d_x_p: 2d_x_p.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
# 	$(NVCC) -o 2d_x_p $^ $(FNVFLAGS)

# 2d_x_p.o: 2d_x_p.cu
# 	$(NVCC) $(NVFLAGS) -c 2d_x_p.cu

# # 2D x non periodic
# 2d_x_np: 2d_x_np.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
# 	$(NVCC) -o 2d_x_np $^ $(FNVFLAGS)

# 2d_x_np.o: 2d_x_np.cu
# 	$(NVCC) $(NVFLAGS) -c 2d_x_np.cu

# # 2D x periodic function
# 2d_x_p_fun: 2d_x_p_fun.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
# 	$(NVCC) -o 2d_x_p_fun $^ $(FNVFLAGS)

# 2d_x_p_fun.o: 2d_x_p_fun.cu
# 	$(NVCC) $(NVFLAGS) -c 2d_x_p_fun.cu

# # 2D x non periodic function
# 2d_x_np_fun: 2d_x_np_fun.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
# 	$(NVCC) -o 2d_x_np_fun $^ $(FNVFLAGS)

# 2d_x_np_fun.o: 2d_x_np_fun.cu
# 	$(NVCC) $(NVFLAGS) -c 2d_x_np_fun.cu

# # 2D y periodic 
# 2d_y_p: 2d_y_p.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
# 	$(NVCC) -o 2d_y_p $^ $(FNVFLAGS)

# 2d_y_p.o: 2d_y_p.cu
# 	$(NVCC) $(NVFLAGS) -c 2d_y_p.cu

# # 2D y periodic function
# 2d_y_p_fun: 2d_y_p_fun.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
# 	$(NVCC) -o 2d_y_p_fun $^ $(FNVFLAGS)

# 2d_y_p_fun.o: 2d_y_p_fun.cu
# 	$(NVCC) $(NVFLAGS) -c 2d_y_p_fun.cu

# # 2D y non periodic
# 2d_y_np: 2d_y_np.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
# 	$(NVCC) -o 2d_y_np $^ $(FNVFLAGS)

# 2d_y_np.o: 2d_y_np.cu
# 	$(NVCC) $(NVFLAGS) -c 2d_y_np.cu

# # 2D xy periodic
# 2d_xy_p: 2d_xy_p.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
# 	$(NVCC) -o 2d_xy_p $^ $(FNVFLAGS)

# 2d_xy_p.o: 2d_xy_p.cu
# 	$(NVCC) $(NVFLAGS) -c 2d_xy_p.cu

# # 2D xy periodic
# 2d_xy_p_fun: 2d_xy_p_fun.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
# 	$(NVCC) -o 2d_xy_p_fun $^ $(FNVFLAGS)

# 2d_xy_p_fun.o: 2d_xy_p_fun.cu
# 	$(NVCC) $(NVFLAGS) -c 2d_xy_p_fun.cu

# # 2D xy periodic
# 2d_xyWENOADV_p: 2d_xyWENOADV_p.o $(STRUCTTAR) $(KERNELTAR) $(UTILTAR) $(DEVFUN)
# 	$(NVCC) -o 2d_xyWENOADV_p $^ $(FNVFLAGS)

# 2d_xyWENOADV_p.o: 2d_xyWENOADV_p.cu
# 	$(NVCC) $(NVFLAGS) -c 2d_xyWENOADV_p.cu

# ----------------------
# Library Functions
# ---------------------

# $(STRUCTTAR): %.o: %.cu
# 	$(NVCC) -c $(NVFLAGS) $< -o $@

# all: $(KERNELTAR)

# $(KERNELTAR):$(KERNELDIR):%.cu
	# $(NVCC) -c $(NVFLAGS) $< -o $@

$(KERNELDIR)/%.o: $(KERNELDIR)/%.cu
   $(NVCC) -c $(NVFLAGS) -o $@ $<

# $(UTILTAR): %.o: %.cu
# 	$(NVCC) -c $(NVFLAGS) $< -o $@

# ----------------------
# Remove everything
# ---------------------

# clean:
# 	rm -f $(STRUCTTAR) $(UTILTAR) $(KERNELTAR) $(MAINOBJ) $(MAIN)


