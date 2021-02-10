#include "spherical_volume_rendering_util.h"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "floating_point_comparison_util.h"

#include <iostream>
using namespace std;

namespace svr {

namespace {
constexpr double DOUBLE_MAX = std::numeric_limits<double>::max();

// The type corresponding to the voxel(s) with the minimum tMax value for a
// given traversal.
enum VoxelIntersectionType {
  Radial = 1,
  Lat = 2,
  Lon = 3,
  RadialLat = 4,
  RadialLon = 5,
  LatLon = 6,
  RadialLatLon = 7
};

// The parameters returned by radialHit().
struct HitParameters {
  // The time at which a hit occurs for the ray at the next point of
  // intersection with a section.
  double tMax;

  // The voxel traversal value of a radial step: 0, +1, -1. This is added to the
  // current voxel.
  int tStep;
};

// Pre-calculated information for the generalized angular hit function, which
// generalizes lon and lat hits. Since the ray segment is dependent
// solely on time, this is unnecessary to calculate twice for each plane hit
// function. Here, ray_segment is the difference between P2 and P1.
struct RaySegment {
 public:
  inline RaySegment(double max_t, const Ray &ray)
      : P2_(ray.pointAtParameter(max_t)), NZDI_(ray.NonZeroDirectionIndex()) {}

  // Updates the point P1 with the new time traversal time t. Similarly, updates
  // the segment denoted by P2 - P1.
  inline void updateAtTime(double t, const Ray &ray) noexcept {
    P1_ = ray.pointAtParameter(t);
    ray_segment_ = P2_ - P1_;
  }

  // Calculates the updated ray segment intersection point given an intersect
  // parameter. More information on the use case can be found at:
  // http://geomalgorithms.com/a05-_intersect-1.html#intersect2D_2Segments()
  inline double intersectionTimeAt(double intersect_parameter,
                                   const Ray &ray) const noexcept {
    return (P1_[NZDI_] + ray_segment_[NZDI_] * intersect_parameter -
            ray.origin()[NZDI_]) *
           ray.invDirection()[NZDI_];
  }

  inline const BoundVec3 &P1() const noexcept { return P1_; }

  inline const BoundVec3 &P2() const noexcept { return P2_; }

  inline const FreeVec3 &vector() const noexcept { return ray_segment_; }

 private:
  // The end point of the ray segment.
  const BoundVec3 P2_;

  // The non-zero direction index of the ray.
  const DirectionIndex NZDI_;

  // The begin point of the ray segment.
  BoundVec3 P1_;

  // The free vector represented by P2 - P1.
  FreeVec3 ray_segment_;
};

inline int calculateLatVoxelIDFromPoints(
    const std::vector<LineSegment> &lat_max, const double p1,
    double p2) noexcept {
    const int bound_size = lat_max.size() / 2;

    // loop over the first set of array elements for the latitude boundaries
    // in the right (XZ) plane
  std::size_t i = 0;
  for (std::size_t j = 1; j < bound_size; ++i, ++j) {
    const double X_diff = lat_max[i].P1 - lat_max[j].P1;
    const double Y_diff = lat_max[i].P2 - lat_max[j].P2;
    const double X_p1_diff = lat_max[i].P1 - p1;
    const double X_p2_diff = lat_max[i].P2 - p2;
    const double Y_p1_diff = lat_max[j].P1 - p1;
    const double Y_p2_diff = lat_max[j].P2 - p2;
    const double d1d2 = (X_p1_diff * X_p1_diff) + (X_p2_diff * X_p2_diff) +
                        (Y_p1_diff * Y_p1_diff) + (Y_p2_diff * Y_p2_diff);
    const double d3 = (X_diff * X_diff) + (Y_diff * Y_diff);
    if (d1d2 < d3 || svr::isEqual(d1d2, d3)) {
      return i;
    }
  }

  // loop over the second set of array elements for the latitude boundaries
  // in the left half (XZ) plane
  std::size_t k = lat_max.size()-1;
  for (std::size_t j = lat_max.size() - 2;  bound_size-1 < j; --k, --j) {
    const double X_diff = lat_max[k].P1 - lat_max[j].P1;
    const double Y_diff = lat_max[k].P2 - lat_max[j].P2;
    const double X_p1_diff = lat_max[k].P1 - p1;
    const double X_p2_diff = lat_max[k].P2 - p2;
    const double Y_p1_diff = lat_max[j].P1 - p1;
    const double Y_p2_diff = lat_max[j].P2 - p2;
    const double d1d2 = (X_p1_diff * X_p1_diff) + (X_p2_diff * X_p2_diff) +
                        (Y_p1_diff * Y_p1_diff) + (Y_p2_diff * Y_p2_diff);
    const double d3 = (X_diff * X_diff) + (Y_diff * Y_diff);
    if (d1d2 < d3 || svr::isEqual(d1d2, d3)) {
      return  (lat_max.size()-1) % k;
    }
  }

  return lat_max.size() + 1;
}

inline int calculateLonVoxelIDFromPoints(
    const std::vector<LineSegment> &lon_max, const double p1,
    double p2) noexcept {

  std::size_t i = 0;
  for (std::size_t j = 1; j < lon_max.size(); ++i, ++j) {
    const double X_diff = lon_max[i].P1 - lon_max[j].P1;
    const double Y_diff = lon_max[i].P2 - lon_max[j].P2;
    const double X_p1_diff = lon_max[i].P1 - p1;
    const double X_p2_diff = lon_max[i].P2 - p2;
    const double Y_p1_diff = lon_max[j].P1 - p1;
    const double Y_p2_diff = lon_max[j].P2 - p2;
    const double d1d2 = (X_p1_diff * X_p1_diff) + (X_p2_diff * X_p2_diff) +
                        (Y_p1_diff * Y_p1_diff) + (Y_p2_diff * Y_p2_diff);
    const double d3 = (X_diff * X_diff) + (Y_diff * Y_diff);
    if (d1d2 < d3 || svr::isEqual(d1d2, d3)) return i;
  }
  return lon_max.size() + 1;
}

// Returns true if the "step" taken from the current voxel ID remains in
// the grid bounds.
inline bool inBoundsLon(const SphericalVoxelGrid &grid, const int step,
                              const int lon_voxel) noexcept {
  if (svr::isEqual( grid.sphereMaxBoundLon(), 2 * M_PI)) {return true;}
  const double min_bound = lon_voxel * grid.deltaPhi();
  const double max_bound = (lon_voxel + 1) * grid.deltaPhi();
  const double min_step_val = min_bound + step * grid.deltaPhi();
  const double min_step = min_step_val < 0 ? 2 * M_PI + min_step_val : min_step_val;
  const double max_step = max_bound + step * grid.deltaPhi();

  return max_step <= grid.sphereMaxBoundLon() &&
         min_step >= grid.sphereMinBoundLon() &&
         min_step <  grid.sphereMaxBoundLon();
}

// Returns true if the "step" taken from the current voxel ID remains in
// the grid bounds.
inline bool inBoundsLat(const SphericalVoxelGrid &grid, const int step,
                          const int lat_voxel) noexcept {
  const double radian = (lat_voxel + 1) * grid.deltaTheta();
  const double angval = radian + step * grid.deltaTheta();


  return angval <= grid.sphereMaxBoundLat() &&
         angval >= grid.sphereMinBoundLat();
}

// Initializes an angular voxel ID. For lat initialization, *_2 represents
// the y-plane. For lon initialization, it represents the z-plane. If the
// number of sections is 1 or the squared euclidean distance of the ray_sphere
// vector in the given plane is zero, the voxel ID is set to 0. Otherwise, we
// find the traversal point of the ray and the sphere center with the projected
// circle given by the entry_radius.
inline int initializeLonVoxelID(const SphericalVoxelGrid &grid,
                                    std::size_t number_of_sections,
                                    const FreeVec3 &ray_sphere,
                                    const std::vector<LineSegment> &lon_max,
                                    double ray_sphere_2, double grid_sphere_2,
                                    double entry_radius) noexcept {
  if (number_of_sections == 1) return 0;
  const double SED =
      ray_sphere.x() * ray_sphere.x() + ray_sphere_2 * ray_sphere_2;
  if (SED == 0.0) return 0;
  const double r = entry_radius / std::sqrt(SED);
  const double p1 = grid.sphereCenter().x() - ray_sphere.x() * r;
  const double p2 = grid_sphere_2 - ray_sphere_2 * r;
  return calculateLonVoxelIDFromPoints(lon_max, p1, p2);
}

inline int initializeLatVoxelID(const SphericalVoxelGrid &grid,
                                    std::size_t number_of_sections,
                                    const FreeVec3 &ray_sphere,
                                    const std::vector<LineSegment> &lat_max,
                                    double ray_sphere_2, double grid_sphere_2,
                                    double entry_radius) noexcept {
  if (number_of_sections == 1) return 0;
  const double SED =
      ray_sphere.x() * ray_sphere.x() + ray_sphere_2 * ray_sphere_2;
  if (SED == 0.0) return 0;
  const double r = entry_radius / std::sqrt(SED);
  const double p1 = grid.sphereCenter().x() - ray_sphere.x() * r;
  const double p2 = grid_sphere_2 - ray_sphere_2 * r;
  return calculateLatVoxelIDFromPoints(lat_max, p1, p2);
}

// Determines whether a radial hit occurs for the given ray. A radial hit is
// considered an intersection with the ray and a radial section. To determine
// line-sphere intersection, this follows closely the mathematics presented in:
// http://cas.xav.free.fr/Graphics%20Gems%204%20-%20Paul%20S.%20Heckbert.pdf
// One also needs to determine when the hit parameter's tStep should go from +1
// to -1, since the radial voxels go from 1..N..1, where N is the number of
// radial sections. This is performed with 'radial_step_has_transitioned'.
inline HitParameters radialHit(const Ray &ray,
                               const svr::SphericalVoxelGrid &grid,
                               bool &radial_step_has_transitioned,
                               int current_radial_voxel, double v,
                               double rsvd_minus_v_squared, double t,
                               double max_t) noexcept {
  if (radial_step_has_transitioned) {
    const double d_b =
        std::sqrt(grid.deltaRadiiSquared(current_radial_voxel - 1) -
                  rsvd_minus_v_squared);
    const double intersection_t = ray.timeOfIntersectionAt(v + d_b);
    if (intersection_t < max_t) return {.tMax = intersection_t, .tStep = -1};
  } else {
    const std::size_t previous_idx =
        std::min(static_cast<std::size_t>(current_radial_voxel),
                 grid.numRadialSections() - 1);
    const double r_a = grid.deltaRadiiSquared(
        previous_idx -
        (grid.deltaRadiiSquared(previous_idx) < rsvd_minus_v_squared));
    const double d_a = std::sqrt(r_a - rsvd_minus_v_squared);
    const double t_entrance = ray.timeOfIntersectionAt(v - d_a);
    const double t_exit = ray.timeOfIntersectionAt(v + d_a);

    const bool t_entrance_gt_t = t_entrance > t;
    if (t_entrance_gt_t && t_entrance == t_exit) {
      // Tangential hit.
      radial_step_has_transitioned = true;
      return {.tMax = t_entrance, .tStep = 0};
    }
    if (t_entrance_gt_t && t_entrance < max_t) {
      return {.tMax = t_entrance, .tStep = 1};
    }
    if (t_exit < max_t) {
      // t_exit is the "further" point of intersection of the current sphere.
      // Since t_entrance is not within our time bounds, it must be true that
      // this is a radial transition.
      radial_step_has_transitioned = true;
      return {.tMax = t_exit, .tStep = -1};
    }
  }
  // There does not exist an intersection time X such that t < X < max_t.
  return {.tMax = DOUBLE_MAX, .tStep = 0};
}

// A generalized version of the latter half of the lat and lon hit
// parameters. Since the only difference is the 2-d plane for which they exist
// in, this portion can be generalized to a single function. The calculations
// presented below follow closely the works of [Foley et al, 1996], [O'Rourke,
// 1998]. Reference:
// http://geomalgorithms.com/a05-_intersect-1.html#intersect2D_2Segments()
HitParameters LonHit(
    const svr::SphericalVoxelGrid &grid, const Ray &ray, double perp_uv_min,
    double perp_uv_max, double perp_uw_min, double perp_uw_max,
    double perp_vw_min, double perp_vw_max, const RaySegment &ray_segment,
    const std::array<double, 2> &collinear_times, double t, double max_t,
    double ray_direction_2, double sphere_center_2,
    const std::vector<svr::LineSegment> &P_max, int current_voxel) noexcept {
  const bool is_parallel_min = svr::isEqual(perp_uv_min, 0.0);
  const bool is_collinear_min = is_parallel_min &&
                                svr::isEqual(perp_uw_min, 0.0) &&
                                svr::isEqual(perp_vw_min, 0.0);
  const bool is_parallel_max = svr::isEqual(perp_uv_max, 0.0);
  const bool is_collinear_max = is_parallel_max &&
                                svr::isEqual(perp_uw_max, 0.0) &&
                                svr::isEqual(perp_vw_max, 0.0);
  double a, b;
  double t_min = collinear_times[is_collinear_min];
  bool is_intersect_min = false;
  if (!is_parallel_min) {
    const double inv_perp_uv_min = 1.0 / perp_uv_min;
    a = perp_vw_min * inv_perp_uv_min;
    b = perp_uw_min * inv_perp_uv_min;
    if (!((svr::lessThan(a, 0.0) || svr::lessThan(1.0, a)) ||
          svr::lessThan(b, 0.0) || svr::lessThan(1.0, b))) {
      is_intersect_min = true;
      t_min = ray_segment.intersectionTimeAt(b, ray);
    }
  }
  double t_max = collinear_times[is_collinear_max];
  bool is_intersect_max = false;
  if (!is_parallel_max) {
    const double inv_perp_uv_max = 1.0 / perp_uv_max;
    a = perp_vw_max * inv_perp_uv_max;
    b = perp_uw_max * inv_perp_uv_max;
    if (!((svr::lessThan(a, 0.0) || svr::lessThan(1.0, a)) ||
          svr::lessThan(b, 0.0) || svr::lessThan(1.0, b))) {
      is_intersect_max = true;
      t_max = ray_segment.intersectionTimeAt(b, ray);
    }
  }
  const bool t_t_max_eq = svr::isEqual(t, t_max);
  const bool t_max_within_bounds = t < t_max && !t_t_max_eq && t_max < max_t;
  const bool t_t_min_eq = svr::isEqual(t, t_min);
  const bool t_min_within_bounds = t < t_min && !t_t_min_eq && t_min < max_t;
  const int  nphi = grid.numLonSections() - 1;
  if (!t_max_within_bounds && !t_min_within_bounds) {
    return {.tMax = DOUBLE_MAX, .tStep = 0};
  }
  if (is_intersect_max && !is_intersect_min && !is_collinear_min &&
      t_max_within_bounds) {
    return {.tMax = t_max, .tStep = 1};
  }
  if (is_intersect_min && !is_intersect_max && !is_collinear_max &&
      t_min_within_bounds) {
    return {.tMax = t_min, .tStep = -1};
  }
  if ((is_intersect_min && is_intersect_max) ||
      (is_intersect_min && is_collinear_max) ||
      (is_intersect_max && is_collinear_min)) {
    const bool min_max_eq = svr::isEqual(t_min, t_max);
    if (min_max_eq && t_min_within_bounds) {
      const double perturbed_t = 0.1;
      a = -ray.direction().x() * perturbed_t;
      b = -ray_direction_2 * perturbed_t;
      const double max_radius_over_plane_length =
          grid.sphereMaxRadius() / std::sqrt(a * a + b * b);
      const double p1 =
          grid.sphereCenter().x() - max_radius_over_plane_length * a;
      const double p2 = sphere_center_2 - max_radius_over_plane_length * b;
      const int next_step = std::abs(
          current_voxel - calculateLonVoxelIDFromPoints(P_max, p1, p2));
      return {.tMax = t_max,
              .tStep = ray.direction().x() < 0.0 || ray_direction_2 < 0.0
                           ? next_step
                           : -next_step};
    }
    if (t_min_within_bounds && ((t_min < t_max && !min_max_eq) || t_t_max_eq)) {
      return {.tMax = t_min, .tStep = -1};
    }
    if (t_max_within_bounds && ((t_max < t_min && !min_max_eq) || t_t_min_eq)) {
      return {.tMax = t_max, .tStep = 1};
    }
  }
  return {.tMax = DOUBLE_MAX, .tStep = 0};
}

HitParameters LatHit(
    const svr::SphericalVoxelGrid &grid, const Ray &ray, double perp_uv_min,
    double perp_uv_max, double perp_uw_min, double perp_uw_max,
    double perp_vw_min, double perp_vw_max, const RaySegment &ray_segment,
    const std::array<double, 2> &collinear_times, double t, double max_t,
    double ray_direction_2, double sphere_center_2,
    const std::vector<svr::LineSegment> &P_max, int current_voxel, bool sym)
    noexcept {
  const int ntheta = grid.numLatSections();
  const bool is_parallel_min = svr::isEqual(perp_uv_min, 0.0);
  const bool is_collinear_min = is_parallel_min &&
                                svr::isEqual(perp_uw_min, 0.0) &&
                                svr::isEqual(perp_vw_min, 0.0);
  const bool is_parallel_max = svr::isEqual(perp_uv_max, 0.0);
  const bool is_collinear_max = is_parallel_max &&
                                svr::isEqual(perp_uw_max, 0.0) &&
                                svr::isEqual(perp_vw_max, 0.0);
  double a, b;
  double t_min = collinear_times[is_collinear_min];
  bool is_intersect_min = false;
  if (!is_parallel_min) {
    const double inv_perp_uv_min = 1.0 / perp_uv_min;
    a = perp_vw_min * inv_perp_uv_min;
    b = perp_uw_min * inv_perp_uv_min;
    if (!((svr::lessThan(a, 0.0) || svr::lessThan(1.0, a)) ||
          svr::lessThan(b, 0.0) || svr::lessThan(1.0, b))) {
      is_intersect_min = true;
      t_min = ray_segment.intersectionTimeAt(b, ray);
    }
  }
  double t_max = collinear_times[is_collinear_max];
  bool is_intersect_max = false;
  if (!is_parallel_max) {
    const double inv_perp_uv_max = 1.0 / perp_uv_max;
    a = perp_vw_max * inv_perp_uv_max;
    b = perp_uw_max * inv_perp_uv_max;
    if (!((svr::lessThan(a, 0.0) || svr::lessThan(1.0, a)) ||
          svr::lessThan(b, 0.0) || svr::lessThan(1.0, b))) {
      is_intersect_max = true;
      t_max = ray_segment.intersectionTimeAt(b, ray);
    }
  }
  const bool t_t_max_eq = svr::isEqual(t, t_max);
  const bool t_max_within_bounds = t < t_max && !t_t_max_eq && t_max < max_t;
  const bool t_t_min_eq = svr::isEqual(t, t_min);
  const bool t_min_within_bounds = t < t_min && !t_t_min_eq && t_min < max_t;
  if (!t_max_within_bounds && !t_min_within_bounds) {
    return {.tMax = DOUBLE_MAX, .tStep = 0};
  }
  if (is_intersect_max && !is_intersect_min && !is_collinear_min &&
      t_max_within_bounds) {
      if (current_voxel - 1 == 2 * ntheta - 1 ||
          (current_voxel == ntheta - 1 &&
           svr::isEqual(grid.sphereMaxBoundLat(),M_PI))) {
        return {.tMax = t_max, .tStep = 0};
      }
    return {.tMax = t_max, .tStep = sym ? -1 : 1};
  }
  if (is_intersect_min && !is_intersect_max && !is_collinear_max &&
      t_min_within_bounds) {
      if (current_voxel == 0 ||
          (current_voxel - 1 == ntheta &&
           svr::isEqual(grid.sphereMaxBoundLat(),M_PI))) {
        return {.tMax = t_min, .tStep = 0};
      }
    return {.tMax = t_min, .tStep = sym ? 1 : -1};
  }
  if ((is_intersect_min && is_intersect_max) ||
      (is_intersect_min && is_collinear_max) ||
      (is_intersect_max && is_collinear_min)) {
    const bool min_max_eq = svr::isEqual(t_min, t_max);
    if (min_max_eq && t_min_within_bounds) {
      const double perturbed_t = 0.1;
      a = -ray.direction().x() * perturbed_t;
      b = -ray_direction_2 * perturbed_t;
      const double max_radius_over_plane_length =
          grid.sphereMaxRadius() / std::sqrt(a * a + b * b);
      const double p1 =
          grid.sphereCenter().x() - max_radius_over_plane_length * a;
      const double p2 = sphere_center_2 - max_radius_over_plane_length * b;
      const int next_voxel = calculateLatVoxelIDFromPoints(P_max, p1, p2);
      const int voxel_id   = 2 * grid.numLatSections() - current_voxel;
      const int next_step = sym ?  std::abs(voxel_id - next_voxel)  :
        std::abs(current_voxel - next_voxel);

      return {.tMax = t_max,
              .tStep = ray.direction().x() < 0.0 || ray_direction_2 < 0.0
                           ? next_step
                           : -next_step};
    }
    if (t_min_within_bounds && ((t_min < t_max && !min_max_eq) || t_t_max_eq)) {
      if (current_voxel == 0 ||
         (current_voxel - 1 == ntheta &&
          svr::isEqual(grid.sphereMaxBoundLat(),M_PI))) {
        return {.tMax = t_min, .tStep = 0};
      }
      return {.tMax = t_min, .tStep = sym ? 1 : -1};
    }
    if (t_max_within_bounds && ((t_max < t_min && !min_max_eq) || t_t_min_eq)) {
      if (current_voxel - 1 == 2 * ntheta - 1 ||
          (current_voxel == ntheta - 1 &&
           svr::isEqual(grid.sphereMaxBoundLat(),M_PI))) {
        return {.tMax = t_max, .tStep = 0};
      }
      return {.tMax = t_max, .tStep = sym ? -1 : 1};
    }
  }
  return {.tMax = DOUBLE_MAX, .tStep = 0};
}



// Determines whether a lat hit occurs for the given ray. A lat hit is
// considered an intersection with the ray and a lat section. The lat
// sections live in the XZ plane.
inline HitParameters LatitudeHit(const Ray &ray,
                              const svr::SphericalVoxelGrid &grid,
                              const RaySegment &ray_segment,
                              const std::array<double, 2> &collinear_times,
                              int current_lat_voxel, bool sym, double t,
                              double max_t) noexcept {
  // Calculate the voxel boundary vectors.
  const BoundVec3 p_one(grid.pMaxLat(current_lat_voxel).P1,
                        0.0,
                        grid.pMaxLat(current_lat_voxel).P2);
  const BoundVec3 p_two(grid.pMaxLat(current_lat_voxel + 1).P1,
                        0.0,
                        grid.pMaxLat(current_lat_voxel + 1).P2);
  const BoundVec3 *u_min = &grid.centerToLatBound(current_lat_voxel);
  const BoundVec3 *u_max = &grid.centerToLatBound(current_lat_voxel + 1);
  const FreeVec3 w_min = p_one - ray_segment.P1();
  const FreeVec3 w_max = p_two - ray_segment.P1();
  const double perp_uv_min = u_min->x() * ray_segment.vector().z() -
                             u_min->z() * ray_segment.vector().x();
  const double perp_uv_max = u_max->x() * ray_segment.vector().z() -
                             u_max->z() * ray_segment.vector().x();
  const double perp_uw_min = u_min->x() * w_min.z() - u_min->z() * w_min.x();
  const double perp_uw_max = u_max->x() * w_max.z() - u_max->z() * w_max.x();
  const double perp_vw_min = ray_segment.vector().x() * w_min.z() -
                             ray_segment.vector().z() * w_min.x();
  const double perp_vw_max = ray_segment.vector().x() * w_max.z() -
                             ray_segment.vector().z() * w_max.x();
  return LatHit(grid, ray, perp_uv_min, perp_uv_max, perp_uw_min,
                  perp_uw_max, perp_vw_min, perp_vw_max, ray_segment,
                  collinear_times, t, max_t, ray.direction().z(),
                  grid.sphereCenter().z(), grid.pMaxLat(),
                  current_lat_voxel, sym);
}

// Determines whether an lon hit occurs for the given ray. An lon
// hit is considered an intersection with the ray and an lon section. The
// lon sections live in the XY plane.
inline HitParameters LongitudeHit(const Ray &ray,
                                  const svr::SphericalVoxelGrid &grid,
                                  const RaySegment &ray_segment,
                                  const std::array<double, 2> &collinear_times,
                                  int current_lon_voxel, double t,
                                  double max_t) noexcept {
  // Calculate the voxel boundary vectors.
  const BoundVec3 p_one(grid.pMaxLon(current_lon_voxel).P1,
                        grid.pMaxLon(current_lon_voxel).P2,
                        0.0);
  const BoundVec3 p_two(grid.pMaxLon(current_lon_voxel + 1).P1,
                        grid.pMaxLon(current_lon_voxel + 1).P2,
                        0.0);
  const BoundVec3 *u_min =
      &grid.centerToLonBound(current_lon_voxel);
  const BoundVec3 *u_max =
      &grid.centerToLonBound(current_lon_voxel + 1);
  const FreeVec3 w_min = p_one - ray_segment.P1();
  const FreeVec3 w_max = p_two - ray_segment.P1();
  const double perp_uv_min = u_min->x() * ray_segment.vector().y() -
                             u_min->y() * ray_segment.vector().x();
  const double perp_uv_max = u_max->x() * ray_segment.vector().y() -
                             u_max->y() * ray_segment.vector().x();
  const double perp_uw_min = u_min->x() * w_min.y() - u_min->y() * w_min.x();
  const double perp_uw_max = u_max->x() * w_max.y() - u_max->y() * w_max.x();
  const double perp_vw_min = ray_segment.vector().x() * w_min.y() -
                             ray_segment.vector().y() * w_min.x();
  const double perp_vw_max = ray_segment.vector().x() * w_max.y() -
                             ray_segment.vector().y() * w_max.x();
  return LonHit(grid, ray, perp_uv_min, perp_uv_max, perp_uw_min,
                    perp_uw_max, perp_vw_min, perp_vw_max, ray_segment,
                    collinear_times, t, max_t, ray.direction().y(),
                    grid.sphereCenter().y(), grid.pMaxLon(),
                    current_lon_voxel);
}

// Calculates the voxel(s) with the minimal tMax for the next intersection.
// Since t is being updated with each interval of the algorithm, this must check
// the following cases:
// 1. tMaxR is the minimum.
// 2. tMaxTheta is the minimum.
// 3. tMaxPhi is the minimum.
// 4. tMaxR, tMaxTheta, tMaxPhi equal intersection.
// 5. tMaxR, tMaxTheta equal intersection.
// 6. tMaxR, tMaxPhi equal intersection.
// 7. tMaxTheta, tMaxPhi equal intersection.
// For each case, the following must hold: t < tMax < max_t
// For reference, uses the following shortform naming:
//        RP = Radial - Lat
//        RA = Radial - Lon
//        PA = Lat  - Lon
inline VoxelIntersectionType minimumIntersection(
    const HitParameters &radial, const HitParameters &lat,
    const HitParameters &lon) noexcept {
  const bool RP_eq = svr::isEqual(radial.tMax, lat.tMax);
  const bool RA_eq = svr::isEqual(radial.tMax, lon.tMax);
  const bool RP_lt = radial.tMax < lat.tMax;
  const bool RA_lt = radial.tMax < lon.tMax;
  if (RP_lt && !RP_eq && RA_lt && !RA_eq) return Radial;

  const bool PA_eq = svr::isEqual(lat.tMax, lon.tMax);
  const bool PA_lt = lat.tMax < lon.tMax;
  if (!RP_lt && !RP_eq && PA_lt && !PA_eq) return Lat;
  if (!PA_lt && !PA_eq && !RA_lt && !RA_eq) return Lon;
  if (RP_eq && RA_eq) return RadialLatLon;
  if (PA_eq) return LatLon;
  if (RP_eq) return RadialLat;
  return RadialLon;
}

// Initialize an array of values representing the points of intersection between
// the lines corresponding to voxel boundaries and a given radial voxel in the
// XY plane and XZ plane. Here, P_* represents these points with a given radius.
//
// The calculations used for P_lat are in the XZ plane, where the longitudinal
// angle phi is 0, so X = rcos(0)sin(theta) and Z = rcos(theta)
// The calculations used for P_lon are in the XY plane, where the latitudinal
// angle theta is pi/2, so X = rcos(phi)sin(pi/2) and Y = rsin(phi)sin(pi/2)
inline void initializeVoxelBoundarySegments(
    std::vector<svr::LineSegment> &P_lat,
    std::vector<svr::LineSegment> &P_lon, bool ray_origin_is_outside_grid,
    const svr::SphericalVoxelGrid &grid, double current_radius) noexcept {
  if (ray_origin_is_outside_grid) {
    P_lat = grid.pMaxLat();
    P_lon = grid.pMaxLon();
    return;
  }
  std::transform(
      grid.latTrigValues().cbegin(), grid.latTrigValues().cend(),
      P_lat.begin(),
      [current_radius, &grid](const TrigonometricValues &tv) -> LineSegment {
        return {.P1 = current_radius * tv.sine   + grid.sphereCenter().x(),
                .P2 = current_radius * tv.cosine + grid.sphereCenter().z()};
      });
  std::transform(
      grid.latTrigValues().rbegin(), grid.latTrigValues().rend(),
      P_lat.begin() + grid.numLatSections() + 1,
      [current_radius, &grid](const TrigonometricValues &tv) -> LineSegment {
        return {.P1 = current_radius * -1 * tv.sine   + grid.sphereCenter().x(),
                .P2 = current_radius      * tv.cosine + grid.sphereCenter().z()};
      });
  std::transform(
      grid.lonTrigValues().cbegin(), grid.lonTrigValues().cend(),
      P_lon.begin(),
      [current_radius, &grid](const TrigonometricValues &tv) -> LineSegment {
        return {.P1 = current_radius * tv.cosine + grid.sphereCenter().x(),
                .P2 = current_radius * tv.sine   + grid.sphereCenter().y()};
      });
}

}  // namespace

std::vector<svr::SphericalVoxel> walkSphericalVolume(
    const Ray &ray, const svr::SphericalVoxelGrid &grid,
    double max_t) noexcept {
  if (max_t <= 0.0) {
    return {};
  }
  const FreeVec3 rsv =
      grid.sphereCenter() - ray.pointAtParameter(0.0);  // Ray Sphere Vector.
  const double SED_from_center = rsv.squared_length();
  int radial_entrance_voxel = 0;
  while (SED_from_center < grid.deltaRadiiSquared(radial_entrance_voxel)) {
    ++radial_entrance_voxel;
  }
  const bool ray_origin_is_outside_grid = (radial_entrance_voxel == 0);

  const std::size_t vector_index =
      radial_entrance_voxel - !ray_origin_is_outside_grid;
  const double entry_radius_squared = grid.deltaRadiiSquared(vector_index);
  const double entry_radius =
      grid.deltaRadius() *
      static_cast<double>(grid.numRadialSections() - vector_index);
  const double rsvd = rsv.dot(rsv);
  const double v = rsv.dot(ray.direction().to_free());
  const double rsvd_minus_v_squared = rsvd - v * v;

  if (entry_radius_squared <= rsvd_minus_v_squared) {
    return {};
  }
  const double d = std::sqrt(entry_radius_squared - rsvd_minus_v_squared);
  const double t_ray_exit = ray.timeOfIntersectionAt(v + d);
  if (t_ray_exit < 0.0) {
   return {};
}
  const double t_ray_entrance = ray.timeOfIntersectionAt(v - d);
  int current_radial_voxel = radial_entrance_voxel + ray_origin_is_outside_grid;

  std::vector<svr::LineSegment> P_lat(2 * (grid.numLatSections() + 1));
  std::vector<svr::LineSegment> P_lon(grid.numLonSections() + 1);
  initializeVoxelBoundarySegments(
      P_lat, P_lon, ray_origin_is_outside_grid, grid, entry_radius);

  const FreeVec3 ray_sphere =
      ray_origin_is_outside_grid
          ? grid.sphereCenter() - ray.pointAtParameter(t_ray_entrance)
          : SED_from_center == 0.0 ? rsv - ray.direction().to_free() : rsv;

  int current_lat_voxel = initializeLatVoxelID(
      grid, grid.numLatSections(), ray_sphere, P_lat,
      ray_sphere.z(), grid.sphereCenter().z(), entry_radius);
  if (static_cast<std::size_t>(current_lat_voxel) >=
      grid.numLatSections()) {
    return {};
  }

  int current_lon_voxel = initializeLonVoxelID(
      grid, grid.numLonSections(), ray_sphere, P_lon,
      ray_sphere.y(), grid.sphereCenter().y(), entry_radius);
  if (static_cast<std::size_t>(current_lon_voxel) >=
      grid.numLonSections()) {
    return {};
  }

  std::vector<svr::SphericalVoxel> voxels;
  voxels.reserve(grid.numRadialSections() + grid.numLatSections() +
                 grid.numLonSections());
  voxels.push_back({.radial = current_radial_voxel,
                    .lat = current_lat_voxel,
                    .lon = current_lon_voxel});

  double t = t_ray_entrance * ray_origin_is_outside_grid;
  const double unitized_ray_time = max_t * grid.sphereMaxDiameter() +
                                   t_ray_entrance * ray_origin_is_outside_grid;
  max_t = ray_origin_is_outside_grid ? std::min(t_ray_exit, unitized_ray_time)
                                     : unitized_ray_time;

  // Initialize the time in case of collinear min or collinear max for angular
  // plane hits. In the case where the hit is not collinear, a time of 0.0 is
  // inputted.
  const std::array<double, 2> collinear_times = {
      0.0, ray.timeOfIntersectionAt(grid.sphereCenter())};

  RaySegment ray_segment(max_t, ray);
  bool radial_step_has_transitioned = false;
  while (true) {
    const auto radial =
        radialHit(ray, grid, radial_step_has_transitioned, current_radial_voxel,
                  v, rsvd_minus_v_squared, t, max_t);
    ray_segment.updateAtTime(t, ray);
    // For latitude, all voxels projected onto the XZ plane have four bounds,
    // so we call the LatitudeHit function twice.
    const auto lat1 = LatitudeHit(ray, grid, ray_segment, collinear_times,
                                current_lat_voxel, false, t, max_t);
    const int symm_lat_ind = 2 * grid.numLatSections() - current_lat_voxel;
    const auto lat_symm = LatitudeHit(ray, grid, ray_segment, collinear_times,
                                  symm_lat_ind, true, t, max_t);
    const auto sym_bool = lat1.tMax < lat_symm.tMax ? false : true ;
    const auto lat = !sym_bool ? lat1 : lat_symm ;
    const auto lon = LongitudeHit(ray, grid, ray_segment, collinear_times,
                                  current_lon_voxel, t, max_t);

    if (current_radial_voxel + radial.tStep == 0 ||
        (radial.tMax == DOUBLE_MAX && lat.tMax == DOUBLE_MAX &&
         lon.tMax == DOUBLE_MAX)) {
      voxels.back().exit_t = t_ray_exit;
      return voxels;
    }
    const auto voxel_intersection =
        minimumIntersection(radial, lat, lon);
    switch (voxel_intersection) {
      case Radial: {
        t = radial.tMax;
        current_radial_voxel += radial.tStep;
        break;
      }
      case Lat: {
        t = lat.tMax;
        if (!inBoundsLat(grid, lat.tStep, current_lat_voxel)) {
          voxels.back().exit_t = t_ray_exit;
          return voxels;
        }
        current_lat_voxel =
            (current_lat_voxel + lat.tStep) % grid.numLatSections();
        break;
      }
      case Lon: {
        if (!inBoundsLon(grid, lon.tStep,
                               current_lon_voxel)) {
          voxels.back().exit_t = t_ray_exit;
          return voxels;
        }
        t = lon.tMax;
        current_lon_voxel = (current_lon_voxel + lon.tStep) %
                                  grid.numLonSections();
        break;
      }
      case RadialLat: {
        t = radial.tMax;
        if (!inBoundsLat(grid, lat.tStep, current_lat_voxel)) {
          voxels.back().exit_t = t_ray_exit;
          return voxels;
        }
        current_radial_voxel += radial.tStep;
        current_lat_voxel =
            (current_lat_voxel + lat.tStep) % grid.numLatSections();
        break;
      }
      case RadialLon: {
        t = radial.tMax;
        if (!inBoundsLon(grid, lon.tStep,
                               current_lon_voxel)) {
          voxels.back().exit_t = t_ray_exit;
          return voxels;
        }
        current_radial_voxel += radial.tStep;
        current_lon_voxel = (current_lon_voxel + lon.tStep) %
                                  grid.numLonSections();
        break;
      }
      case LatLon: {
        t = lat.tMax;
        if (!inBoundsLon(grid, lon.tStep,
                               current_lon_voxel) ||
            !(inBoundsLat(grid, lat.tStep, current_lat_voxel))) {

          voxels.back().exit_t = t_ray_exit;
          return voxels;
        }
        current_lat_voxel =
            (current_lat_voxel + lat.tStep) % grid.numLatSections();
        current_lon_voxel = (current_lon_voxel + lon.tStep) %
                                  grid.numLonSections();
        break;
      }
      case RadialLatLon: {
        t = radial.tMax;
        if (!inBoundsLon(grid, lon.tStep,
                               current_lon_voxel) ||
            !(inBoundsLat(grid, lat.tStep, current_lat_voxel))) {
          voxels.back().exit_t = t_ray_exit;
          return voxels;
        }
        current_radial_voxel += radial.tStep;
        current_lat_voxel =
            (current_lat_voxel + lat.tStep) % grid.numLatSections();
        current_lon_voxel = (current_lon_voxel + lon.tStep) %
                                  grid.numLonSections();
        break;
      }
    }
    if (voxels.back().radial == current_radial_voxel &&
        voxels.back().lat == current_lat_voxel &&
        voxels.back().lon == current_lon_voxel) {
      continue;
    }
    voxels.back().exit_t = t;
    voxels.push_back({.radial = current_radial_voxel,
                      .lat = current_lat_voxel,
                      .lon = current_lon_voxel,
                      .enter_t = t});
  }
}

// LCOV_EXCL_START
std::vector<svr::SphericalVoxel> walkSphericalVolume(
    double *ray_origin, double *ray_direction, double *min_bound,
    double *max_bound, std::size_t num_radial_voxels,
    std::size_t num_lat_voxels, std::size_t num_lon_voxels,
    double *sphere_center, double max_t) noexcept {
  return svr::walkSphericalVolume(
      Ray(BoundVec3(ray_origin[0], ray_origin[1], ray_origin[2]),
          UnitVec3(ray_direction[0], ray_direction[1], ray_direction[2])),
      svr::SphericalVoxelGrid(
          svr::SphereBound{.radial = min_bound[0],
                           .lat = min_bound[1],
                           .lon = min_bound[2]},
          svr::SphereBound{.radial = max_bound[0],
                           .lat = max_bound[1],
                           .lon = max_bound[2]},
          num_radial_voxels, num_lat_voxels, num_lon_voxels,
          BoundVec3(sphere_center[0], sphere_center[1], sphere_center[2])),
      max_t);
}
// LCOV_EXCL_STOP

}  // namespace svr
