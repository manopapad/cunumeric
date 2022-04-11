# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# import tempfile

import numpy as np

import cunumeric as cn


def test():
    # fname = tempfile.mkstemp()[1] -- causes internal error (5030)
    fname = "test.bin"
    x = np.arange(5).astype(np.float64)
    x.tofile(fname)
    y = cn.fromfile(fname, np.float64, 5)
    assert np.array_equal(x, y)


if __name__ == "__main__":
    test()
