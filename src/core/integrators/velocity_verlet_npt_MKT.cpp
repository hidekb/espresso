/*
 * Copyright (C) 2010-2022 The ESPResSo project
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

#include "config/config.hpp"

#ifdef NPT
#include "velocity_verlet_npt.hpp"

#include "BoxGeometry.hpp"
#include "Particle.hpp"
#include "ParticleRange.hpp"
#include "cell_system/CellStructure.hpp"
#include "communication.hpp"
#include "errorhandling.hpp"
#include "npt.hpp"
#include "particle_node.hpp"
#include "system/System.hpp"
#include "thermostat.hpp"
#include "thermostats/npt_inline.hpp"

#include <utils/Vector.hpp>
#include <utils/math/sqr.hpp>

#include <boost/mpi/collectives.hpp>

#include <cmath>
#include <functional>

static constexpr Utils::Vector3i nptgeom_dir{{1, 2, 4}};

/** Scale and communicate instantaneous NpT pressure */
static void
velocity_verlet_npt_propagate_p_eps(IsotropicNptThermostat const &npt_iso,
                                    double time_step) {
  /* finalize derivation of p_inst */
  auto &nptiso = *System::get_system().nptiso;
  nptiso.p_inst = 0.0;
  //nptiso.p_inst_tpdt = 0.0;
  nptiso.p_inst_vir = 0.0;
  for (unsigned int i = 0; i < 3; i++) {
    if (nptiso.geometry & ::nptgeom_dir[i]) {
      nptiso.p_inst += nptiso.p_vir[i] + nptiso.p_vel[i];
      //nptiso.p_inst_tpdt += nptiso.p_vir[i] + nptiso.p_vel_tpdt[i];
      nptiso.p_inst_vir  += nptiso.p_vir[i];
    }
  }

  double p_sum = 0.0;
  boost::mpi::reduce(comm_cart, nptiso.p_inst, p_sum, std::plus<double>(), 0);
  //double p_sum_tpdt = 0.0;
  //boost::mpi::reduce(comm_cart, nptiso.p_inst_tpdt, p_sum_tpdt, std::plus<double>(), 0);
  double p_sum_vir = 0.0;
  boost::mpi::reduce(comm_cart, nptiso.p_inst_vir, p_sum_vir, std::plus<double>(), 0);
  double p_epsilon = 0.0;
  if (this_node == 0) {
    nptiso.p_inst = p_sum / (nptiso.dimension * nptiso.volume);
    //nptiso.p_inst_tpdt = p_sum_tpdt / (nptiso.dimension * nptiso.volume);
    nptiso.p_inst_vir  = p_sum_vir / (nptiso.dimension * nptiso.volume);
    p_epsilon = nptiso.p_epsilon;
    p_epsilon += (nptiso.p_inst - nptiso.p_ext) * 1.5 * time_step * nptiso.volume;
    p_epsilon += (1.0/(get_n_part() - 1)) * (p_sum - p_sum_vir) * 0.5 * time_step;
  }
  boost::mpi::broadcast(comm_cart, p_epsilon, 0);
  nptiso.p_epsilon = p_epsilon;
}

static void
velocity_verlet_npt_propagate_pos(ParticleRangeNPT const &particles,
				  IsotropicNptThermostat const &npt_iso,
				  double time_step) {
  for (auto &p : particles) {
    for (unsigned int j = 0; j < 3; j++) {
      if (!p.is_fixed_along(j)) {
          p.pos()[j] = p.pos()[j] + p.v()[j] * 0.5 * time_step;
      }
    }
  }
}

static void
velocity_verlet_npt_propagate_pos_MTK(ParticleRangeNPT const &particles,
                                      IsotropicNptThermostat const &npt_iso,
                                      double time_step) {
  auto &nptiso = *System::get_system().nptiso;
  auto const propagator =
	  std::exp(nptiso.inv_piston * nptiso.p_epsilon * 0.5 * time_step);

  for (auto &p : particles) {
    for (unsigned int j = 0; j < 3; j++) {
      if (!p.is_fixed_along(j)) {
        p.pos()[j] *= propagator;
      }
    }
  }
}

static void
velocity_verlet_npt_propagate_pos_V(ParticleRangeNPT const &particles,
                                    IsotropicNptThermostat const &npt_iso,
                                    double time_step, System::System &system) {

  auto &box_geo = *system.box_geo;
  auto &cell_structure = *system.cell_structure;
  auto &nptiso = *system.nptiso;
  Utils::Vector3d scal{};
  double L_new = 0.0;

  /* 1st propagation pos_MTK and pos*/
  velocity_verlet_npt_propagate_pos_MTK(particles, npt_iso, time_step);
  velocity_verlet_npt_propagate_pos(particles, npt_iso, time_step);

  /* stochastic reserviors for conjugate momentum for particles*/
  for (auto &p : particles) {
    auto const v_therm = propagate_therm0_nptiso<1>(npt_iso, p.v(), p.mass(), p.id());
    for (unsigned int j = 0; j < 3; j++) {
      if (!p.is_fixed_along(j)) {
        if (nptiso.geometry & ::nptgeom_dir[j]) {
	  p.v()[j] = v_therm[j];
        }
      }
    }
  }

  /* stochastic reserviors for conjugate momentum for V
   * adjust \ref NptIsoParameters::nptiso.volume with dt/2 */
  nptiso.p_epsilon = propagate_thermV_nptiso(npt_iso, nptiso.p_epsilon, nptiso.piston);

  /* propagate Volume */
  if (this_node == 0) {
    nptiso.volume *= std::exp(3.0 * nptiso.inv_piston * nptiso.p_epsilon * time_step);
    if (nptiso.volume < 0.0) {
      runtimeErrorMsg()
	  << "your choice of piston= " << nptiso.piston << ", dt= " << time_step
	  << ", p_epsilon= " << nptiso.p_epsilon
	  << " just caused the volume to become negative, decrease dt";
    }
    L_new = pow(nptiso.volume, 1.0 / nptiso.dimension);
  }

  /* stochastic reserviors for conjugate momentum for V
   * readjust \ref NptIsoParameters::nptiso.volume with dt/2 */
  nptiso.p_epsilon = propagate_thermV_nptiso(npt_iso, nptiso.p_epsilon, nptiso.piston);

  /* 2nd propagation pos and pos_MTK*/
  velocity_verlet_npt_propagate_pos(particles, npt_iso, time_step);
  velocity_verlet_npt_propagate_pos_MTK(particles, npt_iso, time_step);

  cell_structure.set_resort_particles(Cells::RESORT_LOCAL);

  /* Apply new volume to the box-length, communicate it, and account for
   * necessary adjustments to the cell geometry */
  Utils::Vector3d new_box;

  if (this_node == 0) {
    new_box = box_geo.length();

    for (unsigned int i = 0; i < 3; i++) {
      if (nptiso.cubic_box || nptiso.geometry & ::nptgeom_dir[i]) {
        new_box[i] = L_new;
      }
    }
  }

  boost::mpi::broadcast(comm_cart, new_box, 0);

  box_geo.set_length(new_box);
  // fast box length update
  system.on_boxl_change(true);
}

static void
velocity_verlet_npt_propagate_vel_MTK(ParticleRangeNPT const &particles,
                                      IsotropicNptThermostat const &npt_iso,
                                      double time_step) {
  auto &nptiso = *System::get_system().nptiso;
  nptiso.p_vel = {};
  auto const propagater =
	  std::exp(-nptiso.inv_piston * nptiso.p_epsilon * 0.5 * time_step * 
	       (1. + 1./(get_n_part() - 1)));

  for (auto &p : particles) {
    for (unsigned int j = 0; j < 3; j++) {
      if (!p.is_fixed_along(j)) {
        if (nptiso.geometry & ::nptgeom_dir[j]) {
          p.v()[j] *= propagater;
          nptiso.p_vel[j] += Utils::sqr(p.v()[j]) * p.mass();
        }
      }
    }
  }
}

static void
velocity_verlet_npt_propagate_vel(ParticleRangeNPT const &particles,
                                  IsotropicNptThermostat const &npt_iso,
                                  double time_step) {
  auto &nptiso = *System::get_system().nptiso;
  nptiso.p_vel = {};

  for (auto &p : particles) {
    for (unsigned int j = 0; j < 3; j++) {
      if (!p.is_fixed_along(j)) {
        p.v()[j] += p.force()[j] * 0.5 * time_step / p.mass();
        if (nptiso.geometry & ::nptgeom_dir[j]) {
          nptiso.p_vel[j] += Utils::sqr(p.v()[j]) * p.mass();
        }
      }
    }
  }
}

void velocity_verlet_npt_step_1(ParticleRangeNPT const &particles,
                                IsotropicNptThermostat const &npt_iso,
                                double time_step, System::System &system) {
  if (get_n_part() == 1.0) {
    runtimeErrorMsg()
        << "your choice of n = 1 is not allowed with MKT NPT.";
  }
  velocity_verlet_npt_propagate_vel_MTK(particles, npt_iso, time_step);
  velocity_verlet_npt_propagate_p_eps(npt_iso, time_step);
  velocity_verlet_npt_propagate_vel(particles, npt_iso, time_step);
  velocity_verlet_npt_propagate_pos_V(particles, npt_iso, time_step, system);
}

void velocity_verlet_npt_step_2(ParticleRangeNPT const &particles,
                                IsotropicNptThermostat const &npt_iso,
                                double time_step) {
  velocity_verlet_npt_propagate_vel(particles, npt_iso, time_step);
  velocity_verlet_npt_propagate_p_eps(npt_iso, time_step);
  velocity_verlet_npt_propagate_vel_MTK(particles, npt_iso, time_step);
}

#endif // NPT
