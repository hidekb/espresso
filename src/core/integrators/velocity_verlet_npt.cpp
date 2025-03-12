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
#include "system/System.hpp"
#include "thermostat.hpp"
#include "thermostats/npt_inline.hpp"

#include <utils/Vector.hpp>
#include <utils/math/sqr.hpp>

#include <boost/mpi/collectives.hpp>

#include <cmath>
#include <functional>

static constexpr Utils::Vector3i nptgeom_dir{{1, 2, 4}};

static void
velocity_verlet_npt_propagate_vel_final(ParticleRangeNPT const &particles,
                                        IsotropicNptThermostat const &npt_iso,
                                        double time_step) {
  auto &nptiso = *System::get_system().nptiso;
  nptiso.p_vel = {};

  for (auto &p : particles) {
    for (unsigned int j = 0; j < 3; j++) {
      if (!p.is_fixed_along(j)) {
        if (nptiso.geometry & ::nptgeom_dir[j]) {
          nptiso.p_vel[j] += Utils::sqr(p.v()[j]) * p.mass();
          p.v()[j] += p.force()[j] * time_step / 2.0 / p.mass();
        } else {
          // Propagate velocity: v(t+dt) = v(t+0.5*dt) + 0.5*dt * a(t+dt)
          p.v()[j] += p.force()[j] * time_step / 2.0 / p.mass();
        }
      }
    }
  }
}

/** Scale and communicate instantaneous NpT pressure */
static void
velocity_verlet_npt_finalize_p_inst(IsotropicNptThermostat const &npt_iso,
                                    double time_step) {
  /* finalize derivation of p_inst */
  auto &nptiso = *System::get_system().nptiso;
  nptiso.p_inst = 0.0;
  nptiso.p_inst_vir = 0.0;
  for (unsigned int i = 0; i < 3; i++) {
    if (nptiso.geometry & ::nptgeom_dir[i]) {
      nptiso.p_inst += nptiso.p_vir[i] + nptiso.p_vel[i];
      nptiso.p_inst_vir += nptiso.p_vir[i];
    }
  }

  double p_sum = 0.0;
  boost::mpi::reduce(comm_cart, nptiso.p_inst, p_sum, std::plus<double>(), 0);
  double p_sum_vir = 0.0;
  boost::mpi::reduce(comm_cart, nptiso.p_inst_vir, p_sum_vir,
                     std::plus<double>(), 0);
  double p_epsilon = 0.0;
  if (this_node == 0) {
    nptiso.p_inst = p_sum / (nptiso.dimension * nptiso.volume);
    nptiso.p_inst_vir = p_sum_vir / (nptiso.dimension * nptiso.volume);
    p_epsilon = nptiso.p_epsilon;
    p_epsilon += (nptiso.p_inst - nptiso.p_ext) * 0.5 * time_step;
  }
  boost::mpi::broadcast(comm_cart, p_epsilon, 0);
  nptiso.p_epsilon = p_epsilon;
}

static void
velocity_verlet_npt_propagate_pos(ParticleRangeNPT const &particles,
                                  IsotropicNptThermostat const &npt_iso,
                                  double time_step, System::System &system) {

  auto &box_geo = *system.box_geo;
  auto &cell_structure = *system.cell_structure;
  auto &nptiso = *system.nptiso;
  Utils::Vector3d scal{};
  double L_halfdt = 0.0;
  double L_dt = 0.0;

  /* finalize derivation of p_inst */
  velocity_verlet_npt_finalize_p_inst(npt_iso, time_step);

  /* 1st adjust \ref NptIsoParameters::nptiso.volume with dt/2; prepare
   * pos-rescaling
   */
  if (this_node == 0) {
    nptiso.volume += nptiso.inv_piston * nptiso.p_epsilon * 0.5 * time_step;
    scal[2] =
        Utils::sqr(box_geo.length()[nptiso.non_const_dim]) /
        pow(nptiso.volume, 2.0 / nptiso.dimension); // L_0**2 / L_halfdt**2
    if (nptiso.volume < 0.0) {
      runtimeErrorMsg()
          << "your choice of piston= " << nptiso.piston << ", dt= " << time_step
          << ", p_epsilon= " << nptiso.p_epsilon
          << " just caused the volume to become negative, decrease dt";
      nptiso.volume = box_geo.volume();
      scal[2] = 1;
    }

    L_halfdt = pow(nptiso.volume, 1.0 / nptiso.dimension);

    scal[1] = L_halfdt * box_geo.length_inv()[nptiso.non_const_dim];
    scal[0] = 1. / scal[1];
  }
  boost::mpi::broadcast(comm_cart, scal, 0);

  /* 1st propagate positions with dt/2 and rescaling pos*/
  for (auto &p : particles) {
    for (unsigned int j = 0; j < 3; j++) {
      if (!p.is_fixed_along(j)) {
        if (nptiso.geometry & ::nptgeom_dir[j]) {
          p.pos()[j] =
              scal[1] * (p.pos()[j] + scal[2] * p.v()[j] * 0.5 * time_step);
          p.pos_at_last_verlet_update()[j] *= scal[1];
          p.v()[j] *= scal[0];
        } else {
          p.pos()[j] += p.v()[j] * 0.5 * time_step;
        }
      }
    }
  }

  /* stochastic reserviors for conjugate momentum for V
   * 2nd adjust \ref NptIsoParameters::nptiso.volume with dt/2; prepare pos- and
   * vel-rescaling
   */
  nptiso.p_epsilon =
      propagate_thermV_nptiso(npt_iso, nptiso.p_epsilon, nptiso.piston);
  if (this_node == 0) {
    nptiso.volume += nptiso.inv_piston * nptiso.p_epsilon * 0.5 * time_step;
    L_dt = pow(nptiso.volume, 1.0 / nptiso.dimension);

    scal[2] = 1.0;
    scal[1] = L_dt / L_halfdt;
    scal[0] = 1. / scal[1];
  }
  boost::mpi::broadcast(comm_cart, scal, 0);

  /* 2nd propagate positions with dt/2 while rescaling positions and velocities
   */
  for (auto &p : particles) {
    auto const v_therm =
        propagate_therm0_nptiso(npt_iso, p.v(), p.mass(), p.id());
    for (unsigned int j = 0; j < 3; j++) {
      if (!p.is_fixed_along(j)) {
        if (nptiso.geometry & ::nptgeom_dir[j]) {
          p.v()[j] = v_therm[j];
          p.pos()[j] =
              scal[1] * (p.pos()[j] + scal[2] * p.v()[j] * 0.5 * time_step);
          p.pos_at_last_verlet_update()[j] *= scal[1];
          p.v()[j] *= scal[0];
        } else {
          p.pos()[j] += p.v()[j] * 0.5 * time_step;
        }
      }
    }
  }

  cell_structure.set_resort_particles(Cells::RESORT_LOCAL);

  /* Apply new volume to the box-length, communicate it, and account for
   * necessary adjustments to the cell geometry */
  Utils::Vector3d new_box;

  if (this_node == 0) {
    new_box = box_geo.length();

    for (unsigned int i = 0; i < 3; i++) {
      if (nptiso.cubic_box || nptiso.geometry & ::nptgeom_dir[i]) {
        new_box[i] = L_dt;
      }
    }
  }

  boost::mpi::broadcast(comm_cart, new_box, 0);

  box_geo.set_length(new_box);
  // fast box length update
  system.on_boxl_change(true);
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
        p.v()[j] += p.force()[j] * time_step / 2.0 / p.mass();
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
  velocity_verlet_npt_propagate_vel(particles, npt_iso, time_step);
  velocity_verlet_npt_propagate_pos(particles, npt_iso, time_step, system);
}

void velocity_verlet_npt_step_2(ParticleRangeNPT const &particles,
                                IsotropicNptThermostat const &npt_iso,
                                double time_step) {
  velocity_verlet_npt_propagate_vel_final(particles, npt_iso, time_step);
  velocity_verlet_npt_finalize_p_inst(npt_iso, time_step);
}

#endif // NPT
