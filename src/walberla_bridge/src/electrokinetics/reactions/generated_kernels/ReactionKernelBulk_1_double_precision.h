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
//! \\file ReactionKernelBulk_1_double_precision.h
//! \\author pystencils
//======================================================================================================================

// kernel generated with pystencils v1.3.7, lbmpy v1.3.7, sympy v1.12.1,
// lbmpy_walberla/pystencils_walberla from waLBerla commit
// f36fa0a68bae59f0b516f6587ea8fa7c24a41141

#pragma once
#include "core/DataTypes.h"
#include "core/logging/Logging.h"

#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "domain_decomposition/StructuredBlockStorage.h"
#include "field/GhostLayerField.h"
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

class ReactionKernelBulk_1_double_precision {
public:
  ReactionKernelBulk_1_double_precision(BlockDataID rho_0ID_, double order_0,
                                        double rate_coefficient,
                                        double stoech_0)
      : rho_0ID(rho_0ID_), order_0_(order_0),
        rate_coefficient_(rate_coefficient), stoech_0_(stoech_0) {}

  void run(IBlock *block);

  void runOnCellInterval(const shared_ptr<StructuredBlockStorage> &blocks,
                         const CellInterval &globalCellInterval,
                         cell_idx_t ghostLayers, IBlock *block);

  void operator()(IBlock *block) { run(block); }

  static std::function<void(IBlock *)>
  getSweep(const shared_ptr<ReactionKernelBulk_1_double_precision> &kernel) {
    return [kernel](IBlock *b) { kernel->run(b); };
  }

  static std::function<void(IBlock *)> getSweepOnCellInterval(
      const shared_ptr<ReactionKernelBulk_1_double_precision> &kernel,
      const shared_ptr<StructuredBlockStorage> &blocks,
      const CellInterval &globalCellInterval, cell_idx_t ghostLayers = 1) {
    return [kernel, blocks, globalCellInterval, ghostLayers](IBlock *b) {
      kernel->runOnCellInterval(blocks, globalCellInterval, ghostLayers, b);
    };
  }

  std::function<void(IBlock *)> getSweep() {
    return [this](IBlock *b) { this->run(b); };
  }

  std::function<void(IBlock *)>
  getSweepOnCellInterval(const shared_ptr<StructuredBlockStorage> &blocks,
                         const CellInterval &globalCellInterval,
                         cell_idx_t ghostLayers = 1) {
    return [this, blocks, globalCellInterval, ghostLayers](IBlock *b) {
      this->runOnCellInterval(blocks, globalCellInterval, ghostLayers, b);
    };
  }

  void configure(const shared_ptr<StructuredBlockStorage> & /*blocks*/,
                 IBlock * /*block*/) {}

  inline double getOrder_0() const { return order_0_; }
  inline double getRate_coefficient() const { return rate_coefficient_; }
  inline double getStoech_0() const { return stoech_0_; }
  inline void setOrder_0(const double value) { order_0_ = value; }
  inline void setRate_coefficient(const double value) {
    rate_coefficient_ = value;
  }
  inline void setStoech_0(const double value) { stoech_0_ = value; }

private:
  BlockDataID rho_0ID;
  double order_0_;
  double rate_coefficient_;
  double stoech_0_;
};

} // namespace pystencils
} // namespace walberla

#if (defined WALBERLA_CXX_COMPILER_IS_GNU) ||                                  \
    (defined WALBERLA_CXX_COMPILER_IS_CLANG)
#pragma GCC diagnostic pop
#endif
