/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "cunumeric/io/load.h"
#include "cunumeric/cuda_help.h"

namespace cunumeric {

/*static*/ void LoadTask::gpu_variant(TaskContext& context)
{
  // TODO: assuming denseness, same storage order on disk and in memory
  // TODO: proper error reporting
  const char* fname = getenv("CUNUMERIC_FNAME");
  assert(fname != NULL);
  int fd = open(fname, O_RDONLY | O_DIRECT);
  if (fd < 0) {
    std::cerr << "Error " << errno << " while opening " << fname << stdb::endl;
    exit(-1);
  }
  CUfileDescr_t cf_descr;
  memset((void*)&cf_descr, 0, sizeof(CUfileDescr_t));
  cf_descr.handle.fd = fd;
  cf_descr.type      = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  CUfileHandle_t cf_handle;
  CHECK_CUFILE(cuFileHandleRegister(&cf_handle, &cf_descr));
  const Array& out = context.outputs()[0];
  // TODO: type and #dims should match actual array
  Rect<1> rect              = out.shape<1>();
  size_t size               = (rect.hi[0] - rect.lo[0] + 1) * sizeof(double);
  AccessorWO<double, 1> out = out.write_accessor<double, 1>(rect);
  void* outptr              = out.ptr(rect);
  size_t bytes_read         = cuFileRead(cf_handle, outptr, size, 0, 0);
  assert(bytes_read == size);
  CHECK_CUFILE(cuFileHandleDeregister(cf_handle));
  close(fd);
}

}  // namespace cunumeric
