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
 *  Statistical tools to analyze simulations.
 *
 *  Implementation in statistics.cpp.
 */

#include "system/System.hpp"

#include <utils/Vector.hpp>

#include <vector>

/** Calculate the minimal distance of two particles with types in @p set1 and
 *  @p set2, respectively.
 *  @param system particle system
 *  @param set1 types of particles
 *  @param set2 types of particles
 *  @return the minimal distance of two particles
 */
double mindist(System::System const &system, std::vector<int> const &set1,
               std::vector<int> const &set2);

/** Find all particles within a given radius @p dist around a position @p pos.
 *  @param system particle system
 *  @param pos        position of sphere center
 *  @param dist       the sphere radius
 *
 *  @return List of ids close to @p pos.
 */
std::vector<int> nbhood(System::System const &system,
                        Utils::Vector3d const &pos, double dist);

/** Calculate the distribution of particles around others.
 *
 *  Calculates the distance distribution of particles with types given
 *  in the @p p1_types list around particles with types given in the
 *  @p p2_types list. The distances range from @p r_min to @p r_max, binned
 *  into @p r_bins bins which are either equidistant (@p log_flag==false) or
 *  logarithmically equidistant (@p log_flag==true). The result is stored
 *  in the @p array dist.
 *  @param system   particle system
 *  @param p1_types list with types of particles to find the distribution for.
 *  @param p2_types list with types of particles the others are distributed
 *                  around.
 *  @param r_min    Minimal distance for the distribution.
 *  @param r_max    Maximal distance for the distribution.
 *  @param r_bins   Number of bins.
 *  @param log_flag Whether the bins are (logarithmically) equidistant.
 *  @param int_flag Whether the distribution should be cumulative.
 *  @return Radii and distance distribution.
 */
std::vector<std::vector<double>>
calc_part_distribution(System::System const &system,
                       std::vector<int> const &p1_types,
                       std::vector<int> const &p2_types, double r_min,
                       double r_max, int r_bins, bool log_flag, bool int_flag);

/** Calculate the spherically averaged structure factor.
 *
 *  Calculates the spherically averaged structure factor of particles of a
 *  given type. The possible wave vectors are given by q = 2PI/L sqrt(nx^2 +
 *  ny^2 + nz^2).
 *  The S(q) is calculated up to a given length measured in 2PI/L (the
 *  recommended order of the wave vector is less than 20).
 *  The data is stored starting with q=1, and contains alternatingly S(q-1) and
 *  the number of wave vectors l with l^2=q. Only if the second number is
 *  nonzero, the first is meaningful. This means the q=1 entries are sf[0]=S(1)
 *  and sf[1]=1. For q=7, there are no possible wave vectors, so
 *  sf[2*(7-1)]=sf[2*(7-1)+1]=0.
 *
 *  @param  system    particle system
 *  @param  p_types   list with types of particles to be analyzed
 *  @param  order     the maximum wave vector length in units of 2PI/L
 *  @return The scattering vectors q and structure factors S(q).
 */
std::vector<std::vector<double>>
structure_factor(System::System const &system, std::vector<int> const &p_types,
                 int order);

/** @brief Calculate the center of mass of particles of a certain type.
 *  @param system      particle system
 *  @param p_type      type of the particle
 */
Utils::Vector3d center_of_mass(System::System const &system, int p_type);

/** @brief Calculate the angular momentum of particles of a certain type.
 *  @param system      particle system
 *  @param p_type      type of the particle
 */
Utils::Vector3d angular_momentum(System::System const &system, int p_type);

/** @brief Calculate the gyration tensor of particles of certain types.
 *  @param system      particle system
 *  @param p_types     types of the particle
 */
Utils::Vector9d gyration_tensor(System::System const &system,
                                std::vector<int> const &p_types);

/** @brief Calculate the moment of inertia of particles of a certain type.
 *  @param system      particle system
 *  @param p_type      type of the particle
 */
Utils::Vector9d moment_of_inertia_matrix(System::System const &system,
                                         int p_type);

/** Calculate total momentum of the system (particles & LB fluid).
 *  @param system              particle system
 *  @param include_particles   Add particles momentum
 *  @param include_lbfluid     Add LB fluid momentum
 */
Utils::Vector3d calc_linear_momentum(System::System const &system,
                                     bool include_particles,
                                     bool include_lbfluid);
#ifdef NPT
/** Calculate instantaneous pressure during NPT integration.
 *  @param system              particle system
 */
double get_instantaneous_pressure(System::System const &system);

/** Calculate instantaneous virial pressure during NPT integration.
 *  @param system              particle system
 */
double get_instantaneous_pressure_virial(System::System const &system);
#endif
