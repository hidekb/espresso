/*
 * Copyright (C) 2020 The ESPResSo project
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
#ifndef SCRIPT_INTERFACE_PACKED_VARIANT_HPP
#define SCRIPT_INTERFACE_PACKED_VARIANT_HPP

#include "Variant.hpp"

#include <functional>
#include <unordered_map>
#include <utility>

namespace ScriptInterface {
using ObjectId = std::size_t;

/**
 * @brief Id for object.
 *
 * This assigns every ObjectHandle a unique id.
 */
inline ObjectId object_id(const ObjectHandle *p) {
  static_assert(sizeof(const ObjectHandle *) <= sizeof(ObjectId), "");
  return reinterpret_cast<ObjectId>(p);
}

/**
 * @brief Variant value with references replaced by ids.
 */
using PackedVariant = boost::make_recursive_variant<
    None, bool, int, double, std::string, std::vector<int>, std::vector<double>,
    ObjectId, std::vector<boost::recursive_variant_>, Utils::Vector2d,
    Utils::Vector3d, Utils::Vector4d>::type;

using PackedMap = std::vector<std::pair<std::string, PackedVariant>>;

/**
 * @brief Visitor that converts a Variant to a PackedVariant.
 *
 * While packing keeps track of all the ObjectRef values that
 * were encountered, and stores them. This also keeps the
 * referees alive if there are no other owners.
 */
struct PackVisitor : boost::static_visitor<PackedVariant> {
private:
  mutable std::unordered_map<ObjectId, ObjectRef> m_objects;

public:
  /** @brief Map of objects whose references were replaced by ids. */
  auto const &objects() const { return m_objects; }

  /* For the vector, we recurse into each element. */
  auto operator()(const std::vector<Variant> &vec) const {
    std::vector<PackedVariant> ret(vec.size());

    boost::transform(vec, ret.begin(), [this](const Variant &v) {
      return boost::apply_visitor(*this, v);
    });

    return ret;
  }

  /* For object references we store the object reference, and
   * replace it by just an id. */
  PackedVariant operator()(const ObjectRef &so_ptr) const {
    auto const oid = object_id(so_ptr.get());
    m_objects[oid] = so_ptr;

    return oid;
  }

  /* Regular value are just verbatim copied into the result. */
  template <class T> PackedVariant operator()(T &&val) const {
    return std::forward<T>(val);
  }
};

/**
 * @brief Visitor that converts a PackedVariant to a Variant.
 *
 * Object Id are replaced according to the provided object map.
 */
struct UnpackVisitor : boost::static_visitor<Variant> {
  std::unordered_map<ObjectId, ObjectRef> const &objects;

  explicit UnpackVisitor(std::unordered_map<ObjectId, ObjectRef> const &objects)
      : objects(objects) {}

  /* For the vector, we recurse into each element. */
  auto operator()(const std::vector<PackedVariant> &vec) const {
    std::vector<Variant> ret(vec.size());

    boost::transform(vec, ret.begin(), [this](const PackedVariant &v) {
      return boost::apply_visitor(*this, v);
    });

    return ret;
  }

  /* Regular value are just verbatim copied into the result. */
  template <class T> Variant operator()(T &&val) const {
    return std::forward<T>(val);
  }

  /* For object id's they are replaced by references accoding to the map. */
  Variant operator()(const ObjectId &id) const { return objects.at(id); }
};

/**
 * @brief Transform a Variant to a PackedVariant
 *
 * Does apply @ref PackVisitor to a @ref Variant.
 *
 * @param v Input Variant
 * @return Packed variant.
 */
inline PackedVariant pack(const Variant &v) {
  return boost::apply_visitor(PackVisitor{}, v);
}

/**
 * @brief Unpack a PackedVariant.
 *
 * Does apply @ref UnpackVisitor to a @ref Variant.
 *
 * @param v Packed Variant.
 * @param objects Map of ids to reference.
 * @return Transformed variant.
 */
inline Variant unpack(const PackedVariant &v,
                      std::unordered_map<ObjectId, ObjectRef> const &objects) {
  return boost::apply_visitor(UnpackVisitor{objects}, v);
}

inline PackedMap pack(const VariantMap &v) {
  PackedMap ret(v.size());

  boost::transform(v, ret.begin(), [](auto const &kv) {
    return std::pair<std::string, PackedVariant>{kv.first, pack(kv.second)};
  });

  return ret;
}

/**
 * @brief Unpack a PackedMap.
 *
 * Applies @ref unpack to every value in the
 * input map.
 */
inline VariantMap
unpack(const PackedMap &v,
       std::unordered_map<ObjectId, ObjectRef> const &objects) {
  VariantMap ret;

  boost::transform(
      v, std::inserter(ret, ret.end()),
      [&objects](auto const &kv) -> std::pair<std::string, Variant> {
        return {kv.first, unpack(kv.second, objects)};
      });

  return ret;
}
} // namespace ScriptInterface

#endif