/*
 * Copyright (C) 2010-2022 The ESPResSo project
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008,2009,2010
 *   Max-Planck-Institute for Polymer Research, Theory Group
 *
 * This file is part of ESPResSo.
 *
 * ESPResSo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ESPResSo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

/** \file
 *  Routines to calculate the angle energy or/and and force
 *  for a particle triple using the potential described in
 *  @ref bondedIA_angle_cossquare.
 */

#include "angle_common.hpp"

#include <utils/Vector.hpp>
#include <utils/math/sqr.hpp>

#include <cmath>
#include <tuple>

/** Parameters for three-body angular potential (cossquare). */
struct AngleCossquareBond {
  /** bending constant */
  double bend;
  /** equilibrium angle (default is 180 degrees) */
  double phi0;
  /** cosine of @p phi0 (internal parameter) */
  double cos_phi0;

  double cutoff() const { return 0.; }

  static constexpr int num = 2;

  AngleCossquareBond(double bend, double phi0) {
    this->bend = bend;
    this->phi0 = phi0;
    this->cos_phi0 = cos(phi0);
  }

  std::tuple<Utils::Vector3d, Utils::Vector3d, Utils::Vector3d>
  forces(Utils::Vector3d const &vec1, Utils::Vector3d const &vec2) const;
  double energy(Utils::Vector3d const &vec1, Utils::Vector3d const &vec2) const;
};

/** Compute the three-body angle interaction force.
 *  @param[in]  vec1  Vector from central particle to left particle.
 *  @param[in]  vec2  Vector from central particle to right particle.
 *  @return Forces on the second, first and third particles, in that order.
 */
inline std::tuple<Utils::Vector3d, Utils::Vector3d, Utils::Vector3d>
AngleCossquareBond::forces(Utils::Vector3d const &vec1,
                           Utils::Vector3d const &vec2) const {

  auto forceFactor = [this](double const cos_phi) {
    return bend * (cos_phi - cos_phi0);
  };

  return angle_generic_force(vec1, vec2, forceFactor, false);
}

/** Computes the three-body angle interaction energy.
 *  @param[in]  vec1  Vector from central particle to left particle.
 *  @param[in]  vec2  Vector from central particle to right particle.
 */
inline double AngleCossquareBond::energy(Utils::Vector3d const &vec1,
                                         Utils::Vector3d const &vec2) const {
  auto const cos_phi = calc_cosine(vec1, vec2, true);
  return 0.5 * bend * Utils::sqr(cos_phi - cos_phi0);
}
