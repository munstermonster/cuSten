# Andrew Gloster
# January 2019
# Makefile for cuSten library examples

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
NVFLAGS := --device-c -arch=sm_61 -O3 -std=c++11 -L /usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/include/hdf5/serial -lhdf5 -lhdf5_hl -lcublas
NVFLAGFIN := -arch=sm_61 -O3 -std=c++11 ../cuSten/lib/libcuSten.a -L /usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/include/hdf5/serial -lhdf5 -lhdf5_hl -lcublas

OBJBIN := obj
OBJDIR := $(shell mkdir -p $(OBJBIN))

BIN := bin
BINDIR := $(shell mkdir -p $(BIN))

OUT := output
OUTDIR := $(shell mkdir -p $(OUT))

SRCDIF := src
SRCFILE := $(wildcard $(SRCDIF)/*.cu)
SRCOBJ := $(patsubst $(SRCDIF)/%.cu,$(OBJBIN)/%.o,$(SRCFILE))

# ----------------------
# Possible functions
# ---------------------

all: cuPentCahnADI

cuPentCahnADI:$(SRCOBJ)
	$(NVCC) -o bin/cuPentCahnADI $^ $(NVFLAGFIN)

$(OBJBIN)/%.o: $(SRCDIF)/%.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

# ----------------------
# Remove everything
# ---------------------

clean:
	rm -rf $(OBJBIN)
	rm -rf $(BIN)
	rm -rf ${OUT}

