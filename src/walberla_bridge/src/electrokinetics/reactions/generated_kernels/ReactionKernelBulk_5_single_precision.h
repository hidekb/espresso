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
//! \\file ReactionKernelBulk_5_single_precision.h
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

class ReactionKernelBulk_5_single_precision {
public:
  ReactionKernelBulk_5_single_precision(
      BlockDataID rho_0ID_, BlockDataID rho_1ID_, BlockDataID rho_2ID_,
      BlockDataID rho_3ID_, BlockDataID rho_4ID_, float order_0, float order_1,
      float order_2, float order_3, float order_4, float rate_coefficient,
      float stoech_0, float stoech_1, float stoech_2, float stoech_3,
      float stoech_4)
      : rho_0ID(rho_0ID_), rho_1ID(rho_1ID_), rho_2ID(rho_2ID_),
        rho_3ID(rho_3ID_), rho_4ID(rho_4ID_), order_0_(order_0),
        order_1_(order_1), order_2_(order_2), order_3_(order_3),
        order_4_(order_4), rate_coefficient_(rate_coefficient),
        stoech_0_(stoech_0), stoech_1_(stoech_1), stoech_2_(stoech_2),
        stoech_3_(stoech_3), stoech_4_(stoech_4) {}

  void run(IBlock *block);

  void runOnCellInterval(const shared_ptr<StructuredBlockStorage> &blocks,
                         const CellInterval &globalCellInterval,
                         cell_idx_t ghostLayers, IBlock *block);

  void operator()(IBlock *block) { run(block); }

  static std::function<void(IBlock *)>
  getSweep(const shared_ptr<ReactionKernelBulk_5_single_precision> &kernel) {
    return [kernel](IBlock *b) { kernel->run(b); };
  }

  static std::function<void(IBlock *)> getSweepOnCellInterval(
      const shared_ptr<ReactionKernelBulk_5_single_precision> &kernel,
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

  inline float getOrder_0() const { return order_0_; }
  inline float getOrder_1() const { return order_1_; }
  inline float getOrder_2() const { return order_2_; }
  inline float getOrder_3() const { return order_3_; }
  inline float getOrder_4() const { return order_4_; }
  inline float getRate_coefficient() const { return rate_coefficient_; }
  inline float getStoech_0() const { return stoech_0_; }
  inline float getStoech_1() const { return stoech_1_; }
  inline float getStoech_2() const { return stoech_2_; }
  inline float getStoech_3() const { return stoech_3_; }
  inline float getStoech_4() const { return stoech_4_; }
  inline void setOrder_0(const float value) { order_0_ = value; }
  inline void setOrder_1(const float value) { order_1_ = value; }
  inline void setOrder_2(const float value) { order_2_ = value; }
  inline void setOrder_3(const float value) { order_3_ = value; }
  inline void setOrder_4(const float value) { order_4_ = value; }
  inline void setRate_coefficient(const float value) {
    rate_coefficient_ = value;
  }
  inline void setStoech_0(const float value) { stoech_0_ = value; }
  inline void setStoech_1(const float value) { stoech_1_ = value; }
  inline void setStoech_2(const float value) { stoech_2_ = value; }
  inline void setStoech_3(const float value) { stoech_3_ = value; }
  inline void setStoech_4(const float value) { stoech_4_ = value; }

private:
  BlockDataID rho_0ID;
  BlockDataID rho_1ID;
  BlockDataID rho_2ID;
  BlockDataID rho_3ID;
  BlockDataID rho_4ID;
  float order_0_;
  float order_1_;
  float order_2_;
  float order_3_;
  float order_4_;
  float rate_coefficient_;
  float stoech_0_;
  float stoech_1_;
  float stoech_2_;
  float stoech_3_;
  float stoech_4_;
};

} // namespace pystencils
} // namespace walberla

#if (defined WALBERLA_CXX_COMPILER_IS_GNU) ||                                  \
    (defined WALBERLA_CXX_COMPILER_IS_CLANG)
#pragma GCC diagnostic pop
#endif
