/*
 * Copyright (C) 2017-2022 The ESPResSo project
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

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_MODULE scatter_buffer test
#define BOOST_TEST_ALTERNATIVE_INIT_API
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <utils/mpi/scatter_buffer.hpp>

#include <boost/mpi.hpp>

#include <algorithm>
#include <vector>

void check_pointer(boost::mpi::communicator comm, int root) {
  std::vector<int> buf;
  if (comm.rank() == root) {
    auto const n = comm.size();
    auto const total_size = n * (n + 1) / 2;

    for (int i = 1; i <= comm.size(); i++) {
      for (int j = 0; j < i; j++) {
        buf.emplace_back(i);
      }
    }

    BOOST_CHECK_EQUAL(static_cast<int>(buf.size()), total_size);

    Utils::Mpi::scatter_buffer(buf.data(), comm.rank(), comm, root);
  } else {
    for (int i = 0; i < comm.rank(); i++)
      buf.emplace_back(-1);

    Utils::Mpi::scatter_buffer(buf.data(), comm.rank(), comm, root);

    BOOST_CHECK(
        std::ranges::all_of(buf, [&comm](int i) { return i == comm.rank(); }));
  }
}

BOOST_AUTO_TEST_CASE(pointer) {
  boost::mpi::communicator world;
  check_pointer(world, 0);
}

BOOST_AUTO_TEST_CASE(pointer_root) {
  boost::mpi::communicator world;

  auto root = (world.size() >= 3) ? world.size() - 2 : world.size() - 1;
  check_pointer(world, root);
}

int main(int argc, char **argv) {
  boost::mpi::environment mpi_env(argc, argv);

  return boost::unit_test::unit_test_main(init_unit_test, argc, argv);
}
