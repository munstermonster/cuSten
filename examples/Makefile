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
NVFLAGS := --device-c -arch=sm_61 -O3 -std=c++11
NVFLAGFIN := -arch=sm_61 -O3 -std=c++11 ../cuSten/lib/libcuSten.a

OBJBIN := obj
OBJDIR := $(shell mkdir -p $(OBJBIN))

BIN := bin
BINDIR := $(shell mkdir -p $(BIN))

EXDIR := src
EXFILE := $(wildcard $(EXDIR)/*.cu)
EXOBJ := $(patsubst $(EXDIR)/%.cu,$(OBJBIN)/%.o,$(EXFILE))
EXEXE := $(patsubst $(EXDIR)/%.cu,$(BIN)/%,$(EXFILE))

# ----------------------
# Possible functions
# ---------------------

all: $(EXEXE) 

$(EXEXE):$(BIN)/%:$(OBJBIN)/%.o
	$(NVCC) $(NVFLAGFIN) -o $@ $<

$(OBJBIN)/%.o: $(EXDIR)/%.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

# ----------------------
# Remove everything
# ---------------------

clean:
	rm -rf $(OBJBIN)
	rm -rf $(BIN)

