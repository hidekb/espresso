//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \\file InitialPDFsSetterDoublePrecisionCUDA.h
//! \\author pystencils
//======================================================================================================================

// kernel generated with pystencils v1.3.7, lbmpy v1.3.7, sympy v1.12.1,
// lbmpy_walberla/pystencils_walberla from waLBerla commit
// f36fa0a68bae59f0b516f6587ea8fa7c24a41141

#pragma once
#include "core/DataTypes.h"
#include "core/logging/Logging.h"

#include "gpu/GPUField.h"
#include "gpu/GPUWrapper.h"

#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/StructuredBlockStorage.h"
#include "field/SwapableCompare.h"

#include <functional>
#include <unordered_map>

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#if (defined WALBERLA_CXX_COMPILER_IS_GNU) ||                                  \
    (defined WALBERLA_CXX_COMPILER_IS_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wreorder"
#endif

namespace walberla {
namespace pystencils {

class InitialPDFsSetterDoublePrecisionCUDA {
public:
  InitialPDFsSetterDoublePrecisionCUDA(BlockDataID forceID_,
                                       BlockDataID pdfsID_,
                                       BlockDataID velocityID_, double rho_0)
      : forceID(forceID_), pdfsID(pdfsID_), velocityID(velocityID_),
        rho_0_(rho_0) {}

  void run(IBlock *block, gpuStream_t stream = nullptr);

  void runOnCellInterval(const shared_ptr<StructuredBlockStorage> &blocks,
                         const CellInterval &globalCellInterval,
                         cell_idx_t ghostLayers, IBlock *block,
                         gpuStream_t stream = nullptr);

  void operator()(IBlock *block, gpuStream_t stream = nullptr) {
    run(block, stream);
  }

  static std::function<void(IBlock *)>
  getSweep(const shared_ptr<InitialPDFsSetterDoublePrecisionCUDA> &kernel) {
    return [kernel](IBlock *b) { kernel->run(b); };
  }

  static std::function<void(IBlock *, gpuStream_t)> getSweepOnCellInterval(
      const shared_ptr<InitialPDFsSetterDoublePrecisionCUDA> &kernel,
      const shared_ptr<StructuredBlockStorage> &blocks,
      const CellInterval &globalCellInterval, cell_idx_t ghostLayers = 1) {
    return [kernel, blocks, globalCellInterval,
            ghostLayers](IBlock *b, gpuStream_t stream = nullptr) {
      kernel->runOnCellInterval(blocks, globalCellInterval, ghostLayers, b,
                                stream);
    };
  }

  std::function<void(IBlock *)> getSweep(gpuStream_t stream = nullptr) {
    return [this, stream](IBlock *b) { this->run(b, stream); };
  }

  std::function<void(IBlock *)>
  getSweepOnCellInterval(const shared_ptr<StructuredBlockStorage> &blocks,
                         const CellInterval &globalCellInterval,
                         cell_idx_t ghostLayers = 1,
                         gpuStream_t stream = nullptr) {
    return [this, blocks, globalCellInterval, ghostLayers, stream](IBlock *b) {
      this->runOnCellInterval(blocks, globalCellInterval, ghostLayers, b,
                              stream);
    };
  }

  void configure(const shared_ptr<StructuredBlockStorage> & /*blocks*/,
                 IBlock * /*block*/) {}

  inline double getRho_0() const { return rho_0_; }
  inline void setRho_0(const double value) { rho_0_ = value; }

private:
  BlockDataID forceID;
  BlockDataID pdfsID;
  BlockDataID velocityID;
  double rho_0_;
};

} // namespace pystencils
} // namespace walberla

#if (defined WALBERLA_CXX_COMPILER_IS_GNU) ||                                  \
    (defined WALBERLA_CXX_COMPILER_IS_CLANG)
#pragma GCC diagnostic pop
#endif
