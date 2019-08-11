# Andrew Gloster
# August 2019
# Script to compile serial Cahn-Hilliard

# //   Copyright 2018 Andrew Gloster

# //   Licensed under the Apache License, Version 2.0 (the "License");
# //   you may not use this file except in compliance with the License.
# //   You may obtain a copy of the License at

# //       http://www.apache.org/licenses/LICENSE-2.0

# //   Unless required by applicable law or agreed to in writing, software
# //   distributed under the License is distributed on an "AS IS" BASIS,
# //   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# //   See the License for the specific language governing permissions and
# //   limitations under the License.

gcc serialCahnADI.c -o serialCahnADI -O3 -lm -L /usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/include/hdf5/serial -lhdf5 -lhdf5_hl