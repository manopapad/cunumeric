/* Copyright 2021 NVIDIA Corporation
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

#ifndef __NUMPY_ITEM_H__
#define __NUMPY_ITEM_H__

#include "numpy.h"

namespace legate {
namespace numpy {

template <typename T>
class ReadItemTask : public NumPyTask<ReadItemTask<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 1;

 public:
  static T cpu_variant(const Legion::Task* task,
                       const std::vector<Legion::PhysicalRegion>& regions,
                       Legion::Context ctx,
                       Legion::Runtime* runtime);
#ifdef LEGATE_USE_CUDA
  static Legion::DeferredValue<T> gpu_variant(const Legion::Task* task,
                                              const std::vector<Legion::PhysicalRegion>& regions,
                                              Legion::Context ctx,
                                              Legion::Runtime* runtime);
#endif
};

template <typename T>
class WriteItemTask : public NumPyTask<WriteItemTask<T>> {
 public:
  static const int TASK_ID;
  static const int REGIONS = 1;

 public:
  static void cpu_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context ctx,
                          Legion::Runtime* runtime);
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context ctx,
                          Legion::Runtime* runtime);
#endif
};
}  // namespace numpy
}  // namespace legate

#endif  // __NUMPY_ITEM_H__
