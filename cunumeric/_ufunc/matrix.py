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

from cunumeric.module import matmul as matmul_modfun

from .ufunc import float_dtypes, ufunc

_MATMUL_DOCSTRING = """
Matrix product of two arrays.

Refer to :func:`cunumeric.matmul` for full documentation.

See Also
--------
:func:`cunumeric.matmul` : equivalent function

Availability
--------
Multiple GPUs, Multiple CPUs
"""

matmul = ufunc(
    "matmul", _MATMUL_DOCSTRING, {(ty, ty): ty for ty in float_dtypes}
)
matmul.__call__ = matmul_modfun
