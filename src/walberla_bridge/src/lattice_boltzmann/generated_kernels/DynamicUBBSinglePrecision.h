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
//! \\file DynamicUBBSinglePrecision.h
//! \\author pystencils
//======================================================================================================================

// kernel generated with pystencils v1.3.7, lbmpy v1.3.7, sympy v1.12.1,
// lbmpy_walberla/pystencils_walberla from waLBerla commit
// f36fa0a68bae59f0b516f6587ea8fa7c24a41141

#pragma once
#include "core/DataTypes.h"
#include "core/logging/Logging.h"

#include "blockforest/StructuredBlockForest.h"
#include "core/debug/Debug.h"
#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"
#include "field/FlagField.h"
#include "field/GhostLayerField.h"

#include <set>
#include <vector>

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#ifdef WALBERLA_BUILD_WITH_HALF_PRECISION_SUPPORT
using walberla::half;
#endif

namespace walberla {
namespace lbm {

class DynamicUBBSinglePrecision {
public:
  struct IndexInfo {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t dir;
    float vel_0;
    float vel_1;
    float vel_2;
    IndexInfo(int32_t x_, int32_t y_, int32_t z_, int32_t dir_)
        : x(x_), y(y_), z(z_), dir(dir_), vel_0(), vel_1(), vel_2() {}
    bool operator==(const IndexInfo &o) const {
      return x == o.x && y == o.y && z == o.z && dir == o.dir &&
             floatIsEqual(vel_0, o.vel_0) && floatIsEqual(vel_1, o.vel_1) &&
             floatIsEqual(vel_2, o.vel_2);
    }
  };

  class IndexVectors {
  public:
    using CpuIndexVector = std::vector<IndexInfo>;

    enum Type { ALL = 0, INNER = 1, OUTER = 2, NUM_TYPES = 3 };

    IndexVectors() = default;
    bool operator==(IndexVectors const &other) const {
      return other.cpuVectors_ == cpuVectors_;
    }

    CpuIndexVector &indexVector(Type t) { return cpuVectors_[t]; }
    IndexInfo *pointerCpu(Type t) { return cpuVectors_[t].data(); }

    void syncGPU() {}

  private:
    std::vector<CpuIndexVector> cpuVectors_{NUM_TYPES};
  };

  DynamicUBBSinglePrecision(
      const shared_ptr<StructuredBlockForest> &blocks, BlockDataID pdfsID_,
      std::function<Vector3<float32>(const Cell &,
                                     const shared_ptr<StructuredBlockForest> &,
                                     IBlock &)> &velocityCallback)
      : elementInitialiser(velocityCallback), pdfsID(pdfsID_) {
    auto createIdxVector = [](IBlock *const, StructuredBlockStorage *const) {
      return new IndexVectors();
    };
    indexVectorID = blocks->addStructuredBlockData<IndexVectors>(
        createIdxVector, "IndexField_DynamicUBBSinglePrecision");
  }

  void run(IBlock *block);

  void operator()(IBlock *block) { run(block); }

  void inner(IBlock *block);

  void outer(IBlock *block);

  Vector3<double> getForce(IBlock * /*block*/) {

    WALBERLA_ABORT(
        "Boundary condition was not generated including force calculation.")
    return Vector3<double>(double_c(0.0));
  }

  std::function<void(IBlock *)> getSweep() {
    return [this](IBlock *b) { this->run(b); };
  }

  std::function<void(IBlock *)> getInnerSweep() {
    return [this](IBlock *b) { this->inner(b); };
  }

  std::function<void(IBlock *)> getOuterSweep() {
    return [this](IBlock *b) { this->outer(b); };
  }

  template <typename FlagField_T>
  void fillFromFlagField(const shared_ptr<StructuredBlockForest> &blocks,
                         ConstBlockDataID flagFieldID, FlagUID boundaryFlagUID,
                         FlagUID domainFlagUID) {
    for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
      fillFromFlagField<FlagField_T>(blocks, &*blockIt, flagFieldID,
                                     boundaryFlagUID, domainFlagUID);
  }

  template <typename FlagField_T>
  void fillFromFlagField(const shared_ptr<StructuredBlockForest> &blocks,
                         IBlock *block, ConstBlockDataID flagFieldID,
                         FlagUID boundaryFlagUID, FlagUID domainFlagUID) {
    auto *indexVectors = block->getData<IndexVectors>(indexVectorID);
    auto &indexVectorAll = indexVectors->indexVector(IndexVectors::ALL);
    auto &indexVectorInner = indexVectors->indexVector(IndexVectors::INNER);
    auto &indexVectorOuter = indexVectors->indexVector(IndexVectors::OUTER);

    auto *flagField = block->getData<FlagField_T>(flagFieldID);

    if (!(flagField->flagExists(boundaryFlagUID) &&
          flagField->flagExists(domainFlagUID)))
      return;

    auto boundaryFlag = flagField->getFlag(boundaryFlagUID);
    auto domainFlag = flagField->getFlag(domainFlagUID);

    auto inner = flagField->xyzSize();
    inner.expand(cell_idx_t(-1));

    indexVectorAll.clear();
    indexVectorInner.clear();
    indexVectorOuter.clear();

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(0, 0, 0, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 0);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 0, it.y() + 0, it.z() + 0), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(0, 1, 0, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 1);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 0, it.y() + 1, it.z() + 0), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(0, -1, 0, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 2);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 0, it.y() - 1, it.z() + 0), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(-1, 0, 0, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 3);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() - 1, it.y() + 0, it.z() + 0), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(1, 0, 0, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 4);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 1, it.y() + 0, it.z() + 0), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(0, 0, 1, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 5);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 0, it.y() + 0, it.z() + 1), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(0, 0, -1, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 6);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 0, it.y() + 0, it.z() - 1), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(-1, 1, 0, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 7);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() - 1, it.y() + 1, it.z() + 0), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(1, 1, 0, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 8);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 1, it.y() + 1, it.z() + 0), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(-1, -1, 0, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 9);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() - 1, it.y() - 1, it.z() + 0), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(1, -1, 0, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 10);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 1, it.y() - 1, it.z() + 0), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(0, 1, 1, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 11);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 0, it.y() + 1, it.z() + 1), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(0, -1, 1, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 12);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 0, it.y() - 1, it.z() + 1), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(-1, 0, 1, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 13);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() - 1, it.y() + 0, it.z() + 1), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(1, 0, 1, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 14);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 1, it.y() + 0, it.z() + 1), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(0, 1, -1, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 15);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 0, it.y() + 1, it.z() - 1), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(0, -1, -1, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 16);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 0, it.y() - 1, it.z() - 1), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(-1, 0, -1, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 17);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() - 1, it.y() + 0, it.z() - 1), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    for (auto it = flagField->beginWithGhostLayerXYZ(
             cell_idx_c(flagField->nrOfGhostLayers() - 1));
         it != flagField->end(); ++it) {
      if (!isFlagSet(it, domainFlag))
        continue;

      if (isFlagSet(it.neighbor(1, 0, -1, 0), boundaryFlag)) {
        auto element = IndexInfo(it.x(), it.y(), it.z(), 18);
        auto const InitialisationAdditionalData = elementInitialiser(
            Cell(it.x() + 1, it.y() + 0, it.z() - 1), blocks, *block);
        element.vel_0 = InitialisationAdditionalData[0];
        element.vel_1 = InitialisationAdditionalData[1];
        element.vel_2 = InitialisationAdditionalData[2];
        indexVectorAll.push_back(element);
        if (inner.contains(it.x(), it.y(), it.z()))
          indexVectorInner.push_back(element);
        else
          indexVectorOuter.push_back(element);
      }
    }

    indexVectors->syncGPU();
  }

private:
  void run_impl(IBlock *block, IndexVectors::Type type);

  BlockDataID indexVectorID;

  std::function<Vector3<float32>(
      const Cell &, const shared_ptr<StructuredBlockForest> &, IBlock &)>
      elementInitialiser;

public:
  BlockDataID pdfsID;
};

} // namespace lbm
} // namespace walberla
