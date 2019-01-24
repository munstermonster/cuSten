# Andrew Gloster
# January 2019
# Makefile for cuSten library

#   Copyright 2019 Andrew Gloster

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

NVCC := nvcc
NVFLAGS := --device-c --gpu-architecture=sm_61 -O3 -std=c++11

OBJBIN := obj
OBJDIR := $(shell mkdir -p $(OBJBIN))

LIBBIN := lib
LIBDIR := $(shell mkdir -p $(LIBBIN))

KERSRCDIR := src/kernels
KERSRCFILE := $(wildcard $(KERSRCDIR)/*.cu)
KEROBJ := $(patsubst $(KERSRCDIR)/%.cu,$(OBJBIN)/%.o,$(KERSRCFILE))

UTILSRCDIR := src/util
UTILSRCFILE := $(wildcard $(UTILSRCDIR)/*.cu)
UTILOBJ := $(patsubst $(UTILSRCDIR)/%.cu,$(OBJBIN)/%.o,$(UTILSRCFILE))

STRUCTSRCDIR := src/struct
STRUCTSRCFILE := $(wildcard $(STRUCTSRCDIR)/*.cu)
STRUCTOBJ := $(patsubst $(STRUCTSRCDIR)/%.cu,$(OBJBIN)/%.o,$(STRUCTSRCFILE))

# ----------------------
# Possible functions
# ---------------------

all: $(UTILOBJ) $(KEROBJ) $(STRUCTOBJ)
	nvcc --lib  $(UTILOBJ) $(KEROBJ) $(STRUCTOBJ) --output-file $(LIBBIN)/libcuSten.a

$(OBJBIN)/%.o: $(KERSRCDIR)/%.cu
	$(NVCC) -c $(NVFLAGS) -o $@ $<

$(OBJBIN)/%.o: $(UTILSRCDIR)/%.cu
	$(NVCC) -c $(NVFLAGS) -o $@ $<

$(OBJBIN)/%.o: $(STRUCTSRCDIR)/%.cu
	$(NVCC) -c $(NVFLAGS) -o $@ $<

# ----------------------
# Remove everything
# ---------------------

clean:
	rm -rf $(LIBBIN)
	rm -rf $(OBJBIN)

