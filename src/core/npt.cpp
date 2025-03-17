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
#include "npt.hpp"

#ifdef NPT

#include "PropagationMode.hpp"
#include "communication.hpp"
#include "config/config.hpp"
#include "electrostatics/coulomb.hpp"
#include "errorhandling.hpp"
#include "integrators/Propagation.hpp"
#include "magnetostatics/dipoles.hpp"
#include "system/System.hpp"

#include <utils/Vector.hpp>

#include <boost/mpi/collectives.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

static constexpr Utils::Vector3i nptgeom_dir{{1, 2, 4}};

void synchronize_npt_state() {
  auto &nptiso = *System::get_system().nptiso;
  boost::mpi::broadcast(comm_cart, nptiso.p_epsilon, 0);
  boost::mpi::broadcast(comm_cart, nptiso.volume, 0);
  auto &npt_inst_pressure = *System::get_system().npt_inst_pressure;
  boost::mpi::broadcast(comm_cart, npt_inst_pressure.p_inst, 0);
}

void NptIsoParameters::coulomb_dipole_sanity_checks() const {
#if defined(ELECTROSTATICS) or defined(DIPOLES)
  auto &system = System::get_system();
#ifdef ELECTROSTATICS
  if (dimension < 3 and not cubic_box and system.coulomb.impl->solver) {
    throw std::runtime_error("If electrostatics is being used you must "
                             "use the cubic box NpT.");
  }
#endif

#ifdef DIPOLES
  if (dimension < 3 and not cubic_box and system.dipoles.impl->solver) {
    throw std::runtime_error("If magnetostatics is being used you must "
                             "use the cubic box NpT.");
  }
#endif
#endif
}

NptIsoParameters::NptIsoParameters(double ext_pressure, double piston,
                                   Utils::Vector<bool, 3> const &rescale,
                                   bool cubic_box)
    : piston{piston}, p_ext{ext_pressure}, cubic_box{cubic_box} {

  if (ext_pressure < 0.0) {
    throw std::runtime_error("The external pressure must be positive");
  }
  if (piston <= 0.0) {
    throw std::runtime_error("The piston mass must be positive");
  }
  auto const &nptiso = *System::get_system().nptiso;

  inv_piston = nptiso.inv_piston;
  volume = nptiso.volume;
  //p_inst = nptiso.p_inst;
  p_epsilon = nptiso.p_epsilon;
  //p_vir = nptiso.p_vir;
  //p_vel = nptiso.p_vel;

  /* set the NpT geometry */
  for (auto const i : {0u, 1u, 2u}) {
    if (rescale[i]) {
      geometry |= ::nptgeom_dir[i];
      dimension += 1;
      non_const_dim = static_cast<int>(i);
    }
  }

  if (dimension == 0 or non_const_dim == -1) {
    throw std::runtime_error(
        "You must enable at least one of the x y z components "
        "as fluctuating dimension(s) for box length motion");
  }

  coulomb_dipole_sanity_checks();
}

Utils::Vector<bool, 3> NptIsoParameters::get_direction() const {
  return {static_cast<bool>(geometry & ::nptgeom_dir[0]),
          static_cast<bool>(geometry & ::nptgeom_dir[1]),
          static_cast<bool>(geometry & ::nptgeom_dir[2])};
}

void npt_ensemble_init(Utils::Vector3d const &box_l, bool recalc_forces) {
  auto &nptiso = *System::get_system().nptiso;
  nptiso.inv_piston = 1. / nptiso.piston;
  nptiso.volume = std::pow(box_l[nptiso.non_const_dim], nptiso.dimension);
  auto &npt_inst_pressure = *System::get_system().npt_inst_pressure;
  if (recalc_forces) {
    npt_inst_pressure.p_inst = 0.0;
    npt_inst_pressure.p_vir = Utils::Vector3d{};
    npt_inst_pressure.p_vel = Utils::Vector3d{};
  }

  std::vector<double> local_mass;
  for (auto &p : System::get_system().cell_structure->local_particles()) {
    local_mass.push_back(p.mass());
  }
  std::sort(local_mass.begin(), local_mass.end());
  local_mass.erase(std::unique(local_mass.begin(), local_mass.end()),
                   local_mass.end());

  std::vector<std::vector<double>> gathered_mass;
  boost::mpi::gather(comm_cart, local_mass, gathered_mass, 0);
  // Merge mass_list
  std::vector<double> merged_mass;
  if (this_node == 0) {
    for (const auto &vec : gathered_mass) {
      merged_mass.insert(merged_mass.end(), vec.begin(), vec.end());
    }
    std::sort(merged_mass.begin(), merged_mass.end());
    merged_mass.erase(std::unique(merged_mass.begin(), merged_mass.end()),
                      merged_mass.end());
  }
  boost::mpi::broadcast(comm_cart, merged_mass, 0);
  nptiso.mass_list = merged_mass;
}

void integrator_npt_sanity_checks() {
  if (::System::get_system().propagation->used_propagations &
      PropagationMode::TRANS_LANGEVIN_NPT) {
    try {
      auto const &nptiso = *System::get_system().nptiso;
      nptiso.coulomb_dipole_sanity_checks();
    } catch (std::runtime_error const &err) {
      runtimeErrorMsg() << err.what();
    }
  }
}

/** reset virial part of instantaneous pressure */
void npt_reset_instantaneous_virials() {
  if (::System::get_system().propagation->used_propagations &
      PropagationMode::TRANS_LANGEVIN_NPT) {
    auto &npt_inst_pressure = *System::get_system().npt_inst_pressure;
    npt_inst_pressure.p_vir = Utils::Vector3d{};
  }
}

void npt_add_virial_contribution(double energy) {
  if (::System::get_system().propagation->used_propagations &
      PropagationMode::TRANS_LANGEVIN_NPT) {
    auto &npt_inst_pressure = *System::get_system().npt_inst_pressure;
    npt_inst_pressure.p_vir[0] += energy;
  }
}

void npt_add_virial_contribution(const Utils::Vector3d &force,
                                 const Utils::Vector3d &d) {
  if (::System::get_system().propagation->used_propagations &
      PropagationMode::TRANS_LANGEVIN_NPT) {
    auto &npt_inst_pressure = *System::get_system().npt_inst_pressure;
    npt_inst_pressure.p_vir += hadamard_product(force, d);
  }
}
#endif // NPT
