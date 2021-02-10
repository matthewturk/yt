#ifndef SPHERICAL_VOLUME_RENDERING_SPHERICALVOXELGRID_H
#define SPHERICAL_VOLUME_RENDERING_SPHERICALVOXELGRID_H

#include <vector>

#include "vec3.h"

namespace svr {

// Represents the boundary for the sphere. This is used to determine the minimum
// and maximum boundaries for a sectored traversal.
struct SphereBound {
  double radial;
  double lat;
  double lon;
};

// Represents a line segment that is used for the points of intersections
// between the lines corresponding to voxel boundaries and a given radial voxel.
struct LineSegment {
  double P1;
  double P2;
};

// The trigonometric values for a given radian.
struct TrigonometricValues {
  double cosine;
  double sine;
};

namespace {

constexpr double TAU = 2 * M_PI;

// Initializes the delta radii squared. These values are used for radial hit
// calculations in the main spherical volume algorithm. This calculates
// delta_radius^2 for num_radial_sections + 1 iterations. The delta radius value
// begins at max_radius, and subtracts delta_radius with each index.
// For example,
//
// Given: num_radial_voxels = 3, max_radius = 6, delta_radius = 2
// Returns: { 6*6, 4*4, 2*2, 0*0 }
std::vector<double> initializeDeltaRadiiSquared(
    const std::size_t num_radial_voxels, const double max_radius,
    const double delta_radius) {
  std::vector<double> delta_radii_squared(num_radial_voxels + 1);

  double current_delta_radius = max_radius;
  std::generate(delta_radii_squared.begin(), delta_radii_squared.end(),
                [&]() -> double {
                  const double old_delta_radius = current_delta_radius;
                  current_delta_radius -= delta_radius;
                  return old_delta_radius * old_delta_radius;
                });
  return delta_radii_squared;
}

// Returns a vector of TrigonometricValues for the given number of voxels.
// This begins with min_bound, and increments by a value of delta
// for num_voxels + 1 iterations. For example,
//
// Given: num_voxels = 2, min_bound = 0.0, delta = pi/2
// Returns: { {.cosine=1.0, .sine=0.0},
//            {.cosine=0.0, .sine=1.0},
//            {.cosine=1.0, .sine=0.0} }
std::vector<TrigonometricValues> initializeTrigonometricValues(
    const std::size_t num_voxels, const double min_bound, const double delta) {
  std::vector<TrigonometricValues> trig_values(num_voxels + 1);

  double radians = min_bound;
  std::generate(trig_values.begin(), trig_values.end(),
                [&]() -> TrigonometricValues {
                  const double cos = std::cos(radians);
                  const double sin = std::sin(radians);
                  radians += delta;
                  return {.cosine = cos, .sine = sin};
                });
  return trig_values;
}

// Returns a vector of maximum radius line segments for the given trigonometric
// values. The following predicate should be true:
// num_voxels + 1 == trig_values.size() + 1.
//
// The LineSegment points P1 and P2 are calculated with the following equations
// for the latitudinal case:
// .P1 = max_radius * trig_value.sine + center.x().
// .P2 = max_radius * trig_value.cosine + center.z().
// In the latitudinal case, we also add the symmetric sections to the vector.
std::vector<LineSegment> initializeMaxRadiusLineSegmentsLat(
    const std::size_t num_voxels, const BoundVec3 &center,
    const double max_radius,
    const std::vector<TrigonometricValues> &trig_values) {
  std::vector<LineSegment> line_segments(2 * (num_voxels + 1));
  std::transform(trig_values.cbegin(), trig_values.cend(),
                 line_segments.begin(),
                 [&](const TrigonometricValues &trig_value) -> LineSegment {
                   return {.P1 = max_radius * trig_value.sine + center.x(),
                           .P2 = max_radius * trig_value.cosine + center.z()};
                 });
  std::transform(trig_values.rbegin() , trig_values.rend(),
                  line_segments.begin() + num_voxels + 1,
                  [&](const TrigonometricValues &trig_value) -> LineSegment {
                    return {.P1 = max_radius * -1 * trig_value.sine   + center.x(),
                            .P2 = max_radius      * trig_value.cosine + center.z()};
                  });
  return line_segments;
}
// The LineSegment points P1 and P2 are calculated with the following equations
// for the longitudinal case:
// .P1 = max_radius * trig_value.cosine + center.x().
// .P2 = max_radius * trig_value.sine + center.y().
std::vector<LineSegment> initializeMaxRadiusLineSegmentsLon(
    const std::size_t num_voxels, const BoundVec3 &center,
    const double max_radius,
    const std::vector<TrigonometricValues> &trig_values) {
  std::vector<LineSegment> line_segments(num_voxels + 1);
  std::transform(trig_values.cbegin(), trig_values.cend(),
                 line_segments.begin(),
                 [&](const TrigonometricValues &trig_value) -> LineSegment {
                   return {.P1 = max_radius * trig_value.cosine + center.x(),
                           .P2 = max_radius * trig_value.sine + center.y()};
                 });
  return line_segments;
}

// Initializes the vectors determined by the following calculation:
// sphere center - {X, Y, Z}, WHERE X, Z = P1, P2 for lat voxels.
std::vector<BoundVec3> initializeCenterToLatPMaxVectors(
    const std::vector<LineSegment> &line_segments, const BoundVec3 &center) {
  std::vector<BoundVec3> center_to_pmax_vectors;
  center_to_pmax_vectors.reserve(line_segments.size());

  for (const auto &points : line_segments) {
    center_to_pmax_vectors.emplace_back(center -
                                        FreeVec3(points.P1, 0.0, points.P2));
  }
  return center_to_pmax_vectors;
}

// Similar to above, but uses:
// sphere center - {X, Y, Z}, WHERE X, Y = P1, P2 for lon voxels.
std::vector<BoundVec3> initializeCenterToLonPMaxVectors(
    const std::vector<LineSegment> &line_segments, const BoundVec3 &center) {
  std::vector<BoundVec3> center_to_pmax_vectors;
  center_to_pmax_vectors.reserve(line_segments.size());

  for (const auto &points : line_segments) {
    center_to_pmax_vectors.emplace_back(center -
                                        FreeVec3(points.P1, points.P2, 0.0));
  }
  return center_to_pmax_vectors;
}

}  // namespace

// Represents a spherical voxel grid used for ray casting. The bounds of the
// grid are determined by min_bound and max_bound. The deltas are then
// determined by (max_bound.X - min_bound.X) / num_X_sections. To minimize
// calculation duplication, many calculations are completed once here and used
// each time a ray traverses the spherical voxel grid.
//
// Note that the grid system currently does not align with one would expect
// from spherical coordinates. We represent both lat and lon within
// bounds [0, 2pi].
// TODO(cgyurgyik): Look into updating lat grid from [0, 2pi] -> [0, pi].
struct SphericalVoxelGrid {
 public:
  SphericalVoxelGrid(const SphereBound &min_bound, const SphereBound &max_bound,
                     std::size_t num_radial_sections,
                     std::size_t num_lat_sections,
                     std::size_t num_lon_sections,
                     const BoundVec3 &sphere_center)
      : num_radial_sections_(num_radial_sections),
        num_lat_sections_(num_lat_sections),
        num_lon_sections_(num_lon_sections),
        sphere_center_(sphere_center),
        sphere_max_bound_lat_(max_bound.lat),
        sphere_min_bound_lat_(min_bound.lat),
        sphere_max_bound_lon_(max_bound.lon),
        sphere_min_bound_lon_(min_bound.lon),
        // TODO(cgyurgyik): Verify we want the sphere_max_radius to simply be
        // max_bound.radial.
        sphere_max_radius_(max_bound.radial),
        sphere_max_diameter_(sphere_max_radius_ * 2.0),
        delta_radius_((max_bound.radial - min_bound.radial) /
                      num_radial_sections),
        delta_theta_((max_bound.lat - min_bound.lat) / num_lat_sections),
        delta_phi_((max_bound.lon - min_bound.lon) /
                   num_lon_sections),
        // TODO(cgyurgyik): Verify this is actually what we want for
        // 'max_radius'. The other option is simply using max_bound.radial
        delta_radii_sq_(initializeDeltaRadiiSquared(
            num_radial_sections,
            /*max_radius=*/max_bound.radial - min_bound.radial, delta_radius_)),
        lat_trig_values_(initializeTrigonometricValues(
            num_lat_sections, min_bound.lat, delta_theta_)),
        lon_trig_values_(initializeTrigonometricValues(
            num_lon_sections, min_bound.lon, delta_phi_)),
        P_max_lat_(initializeMaxRadiusLineSegmentsLat(
            num_lat_sections, sphere_center, sphere_max_radius_,
            lat_trig_values_)),
        P_max_lon_(initializeMaxRadiusLineSegmentsLon(
            num_lon_sections, sphere_center, sphere_max_radius_,
            lon_trig_values_)),
        center_to_lat_bound_vectors_(
            initializeCenterToLatPMaxVectors(P_max_lat_, sphere_center)),
        center_to_lon_bound_vectors_(
            initializeCenterToLonPMaxVectors(P_max_lon_,
                                                   sphere_center)) {}

  inline std::size_t numRadialSections() const noexcept {
    return this->num_radial_sections_;
  }

  inline std::size_t numLatSections() const noexcept {
    return this->num_lat_sections_;
  }

  inline std::size_t numLonSections() const noexcept {
    return this->num_lon_sections_;
  }

  inline double sphereMaxBoundLat() const noexcept {
    return this->sphere_max_bound_lat_;
  }

  inline double sphereMinBoundLat() const noexcept {
    return this->sphere_min_bound_lat_;
  }

  inline double sphereMaxBoundLon() const noexcept {
    return this->sphere_max_bound_lon_;
  }

  inline double sphereMinBoundLon() const noexcept {
    return this->sphere_min_bound_lon_;
  }

  inline double sphereMaxRadius() const noexcept {
    return this->sphere_max_radius_;
  }

  inline double sphereMaxDiameter() const noexcept {
    return this->sphere_max_diameter_;
  }

  inline const BoundVec3 &sphereCenter() const noexcept {
    return this->sphere_center_;
  }

  inline double deltaRadius() const noexcept { return delta_radius_; }

  inline double deltaPhi() const noexcept { return delta_phi_; }

  inline double deltaTheta() const noexcept { return delta_theta_; }

  inline double deltaRadiiSquared(std::size_t i) const noexcept {
    return this->delta_radii_sq_[i];
  }

  inline const LineSegment &pMaxLat(std::size_t i) const noexcept {
    return this->P_max_lat_[i];
  }

  inline const std::vector<LineSegment> &pMaxLat() const noexcept {
    return this->P_max_lat_;
  }

  inline const BoundVec3 &centerToLatBound(std::size_t i) const noexcept {
    return this->center_to_lat_bound_vectors_[i];
  }

  inline const LineSegment &pMaxLon(std::size_t i) const noexcept {
    return this->P_max_lon_[i];
  }

  inline const std::vector<LineSegment> &pMaxLon() const noexcept {
    return this->P_max_lon_;
  }

  inline const BoundVec3 &centerToLonBound(std::size_t i) const noexcept {
    return this->center_to_lon_bound_vectors_[i];
  }

  inline const std::vector<TrigonometricValues> &latTrigValues()
      const noexcept {
    return lat_trig_values_;
  }

  inline const std::vector<TrigonometricValues> &lonTrigValues()
      const noexcept {
    return lon_trig_values_;
  }

 private:
  // The number of radial, lat, and lon voxels.
  const std::size_t num_radial_sections_, num_lat_sections_,
      num_lon_sections_;

  // The center of the sphere.
  const BoundVec3 sphere_center_;

  // The maximum lat bound of the sphere.
  const double sphere_max_bound_lat_;

  // The minimum lat bound of the sphere.
  const double sphere_min_bound_lat_;

  // The maximum lon bound of the sphere.
  const double sphere_max_bound_lon_;

  // The minimum lon bound of the sphere.
  const double sphere_min_bound_lon_;

  // The maximum radius of the sphere.
  const double sphere_max_radius_;

  // The maximum diamater of the sphere.
  const double sphere_max_diameter_;

  // The maximum sphere radius divided by the number of radial sections.
  const double delta_radius_;

  // 2 * PI divided by X, where X is the number of lat and number of lon
  // sections respectively.
  const double delta_theta_, delta_phi_;

  // The delta radii squared calculated for use in radial hit calculations.
  const std::vector<double> delta_radii_sq_;

  // The trigonometric values calculated for the lon and lat voxels.
  const std::vector<TrigonometricValues> lon_trig_values_,
      lat_trig_values_;

  // The maximum radius line segments for lat and lon voxels.
  const std::vector<LineSegment> P_max_lat_, P_max_lon_;

  // The vectors represented by the vector sphere center - P_max[i].
  const std::vector<BoundVec3> center_to_lat_bound_vectors_,
      center_to_lon_bound_vectors_;
};

}  // namespace svr

#endif  // SPHERICAL_VOLUME_RENDERING_SPHERICALVOXELGRID_H
