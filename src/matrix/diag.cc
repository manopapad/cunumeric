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

#include "matrix/diag.h"
#include "matrix/diag_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <LegateTypeCode CODE>
struct DiagImplBody<VariantKind::CPU, CODE> {
  using VAL = legate_type_of<CODE>;

  void operator()(const AccessorWO<VAL, 2> &out,
                  const AccessorRO<VAL, 2> &in,
                  const Point<2> &start,
                  size_t distance) const
  {
    for (coord_t idx = 0; idx < distance; ++idx) {
      Point<2> p(start[0] + idx, start[1] + idx);
      out[p] = in[p];
    }
  }

  void operator()(const AccessorRD<SumReduction<VAL>, true, 2> &out,
                  const AccessorRO<VAL, 2> &in,
                  const Point<2> &start,
                  size_t distance) const
  {
    for (coord_t idx = 0; idx < distance; ++idx) {
      Point<2> p(start[0] + idx, start[1] + idx);
      auto v = in[p];
      out.reduce(p, v);
    }
  }
};

void deserialize(Deserializer &ctx, DiagArgs &args)
{
  deserialize(ctx, args.matrix);
  deserialize(ctx, args.diag);
  args.extract = args.diag.is_reducible();
  if (!args.extract) deserialize(ctx, args.matrix);
}

/*static*/ void DiagTask::cpu_variant(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context context,
                                      Runtime *runtime)
{
  diag_template<VariantKind::CPU>(task, regions, context, runtime);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { DiagTask::register_variants(); }
}  // namespace

}  // namespace numpy
}  // namespace legate
