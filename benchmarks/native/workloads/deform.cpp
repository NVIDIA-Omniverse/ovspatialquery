/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Deforming-sheet publication: stage authoring through the pinned Ovstage SDK, the installed Ovstage
// lane on CPU and CUDA plans, direct PhysX recook and refit lanes, the triangle oracle, parity and timing.
#include "deform.h"

#include "../installed_adapter.h"
#include "../measurement.h"
#include "ovstage_support.h"
#include "pairs.h"

#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovspatialquery/ovspatialquery_pairs.h>
#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace bench
{
namespace
{
// The stage is populated at ordinal 1 and its hierarchy derived into ordinal 2, the first sealed read.
constexpr ovstage_ordinal_t initialOrdinal = 2;
constexpr double            cellPitch = 4.0, sheetSize = 2.0, sheetHeight = 1.0, rayHeight = 4.0, rayRange = 8.0;
constexpr double            waveAmplitude = 0.15, crumpleAmplitude = 0.6, rigidShift = 0.25, toolRadius = 0.05;
constexpr double            extentMargin = 1e-4;
// Sheet subdivisions per side, cycling over the cells: 2,048, 4,050, 8,192 and 20,000 triangles.
constexpr size_t sheetSides[]      = {32, 45, 64, 100};
constexpr size_t paritySteps       = 4;
constexpr size_t topologyEvery     = 4; // Mixed mode replaces one changed sheet's indices every fourth step.
constexpr char   worldMatrixName[] = "omni:fabric:worldMatrix";
constexpr char   worldExtentName[] = "_worldExtent";

enum class Deform : uint8_t
{
    none,    // Republish a sealed ordinal without any write: the unchanged control.
    rigid,   // Transform-only: the changed sheets shift along x, points untouched.
    vertex,  // Vertex-only: the changed sheets follow an analytic travelling wave, indices untouched.
    fold,    // Vertex-only: the sheet folds into a stack of eight pleats, so a tree built flat overlaps itself.
    crumple, // Vertex-only with pseudo-random displacements of the sheet's own size: adversarial for any tree.
    mixed,   // Vertex-only waves plus an occasional index rewrite (flipped diagonals) on one changed sheet.
};
constexpr size_t foldPleats = 8;

Deform deformMode(const Options& o)
{
    const std::string suffix = o.workload.substr(std::strlen("deform-"));
    if (suffix == "none")
        return Deform::none;
    if (suffix == "rigid")
        return Deform::rigid;
    if (suffix == "vertex")
        return Deform::vertex;
    if (suffix == "crumple")
        return Deform::crumple;
    if (suffix == "fold")
        return Deform::fold;
    require(suffix == "mixed", "deform mode must be none, rigid, vertex, fold, crumple or mixed");
    return Deform::mixed;
}
const char* modeName(Deform mode)
{
    switch (mode)
    {
    case Deform::none: return "none";
    case Deform::rigid: return "rigid";
    case Deform::vertex: return "vertex";
    case Deform::fold: return "fold";
    case Deform::crumple: return "crumple";
    case Deform::mixed: return "mixed";
    }
    return "unknown";
}

double frac(double x) { return x - std::floor(x); }
// splitmix64: a deterministic per-(cell, vertex, step) offset for the crumple mode.
double unitNoise(uint64_t cell, uint64_t vertex, uint64_t step, uint64_t axis)
{
    uint64_t z = (cell + 1) * 0x9E3779B97F4A7C15ull + (vertex + 1) * 0xBF58476D1CE4E5B9ull +
                 (step + 1) * 0x94D049BB133111EBull + (axis + 1) * 0xD6E8FEB86659FD93ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return double(z >> 11) / double(1ull << 53) * 2.0 - 1.0;
}

// One sheet per cell: a triangle grid of `side` x `side` quads split along one diagonal, in the plane
// z = 0 of its own frame, translated to the cell. Points and the diagonal choice change over time; the
// vertex count, the face count and the object identity never do.
struct Sheet
{
    uint64_t id   = 0; // Fixture identity, 1-based; the population object is the prim path.
    size_t   side = 32;
    Vec      cell{};         // Cell translation at rest (the sheet occupies [0, sheetSize]^2 in x, y).
    double   shift      = 0; // Rigid x shift of the current step.
    bool     flipped    = false;
    size_t   placedStep = 0; // The step whose wave (or crumple) the current points follow.
    size_t   vertexCount() const { return (side + 1) * (side + 1); }
    size_t   triangleCount() const { return 2 * side * side; }
};
std::vector<std::array<uint32_t, 3>> triangulate(size_t side, bool flipped)
{
    std::vector<std::array<uint32_t, 3>> triangles;
    triangles.reserve(2 * side * side);
    for (size_t r = 0; r < side; ++r)
        for (size_t c = 0; c < side; ++c)
        {
            const uint32_t v00 = uint32_t(r * (side + 1) + c), v10 = v00 + 1, v01 = uint32_t(v00 + side + 1),
                           v11 = v01 + 1;
            // Counter-clockwise seen from +z, so the single-sided front faces the rays from above.
            if (!flipped)
            {
                triangles.push_back({v00, v10, v11});
                triangles.push_back({v00, v11, v01});
            }
            else
            {
                triangles.push_back({v00, v10, v01});
                triangles.push_back({v10, v11, v01});
            }
        }
    return triangles;
}
// The analytic travelling wave of `vertex` and `mixed`: pinned along y = 0, 1, 2, breathing in amplitude
// so that the sheet's world extent, and with it the tool pairs, change from step to step.
double waveAmplitudeAt(size_t cell, size_t step)
{
    return waveAmplitude * (0.55 + 0.45 * std::sin(0.4 * double(step) + 0.3 * double(cell)));
}
double waveZ(double x, double y, size_t cell, size_t step)
{
    const double pi = std::acos(-1.0);
    return waveAmplitudeAt(cell, step) * std::sin(pi * x + 0.7 * double(step) + 0.3 * double(cell)) * std::sin(pi * y);
}

struct Scene
{
    Deform             mode = Deform::vertex;
    std::vector<Sheet> sheets;
    // f.meshes[i] holds sheet i's current local vertices and triangles, f.objects[i] its pose; the
    // ray queries live in f.queries. The oracle and the shared comparison read this fixture.
    Fixture           f;
    std::vector<Aabb> extents; // Authored world extents, the pair oracle's bounds.
    size_t            changedPerStep = 0;
    // State of the current step.
    size_t                step = 0;
    std::vector<size_t>   changedNow;
    std::optional<size_t> topologyNow; // The sheet whose indices this step rewrote.
    size_t                changedTriangles = 0;
};

Aabb sheetExtent(const Scene& s, size_t i)
{
    const auto& mesh = s.f.meshes[i];
    const auto& ob   = s.f.objects[i];
    Aabb        b{{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}};
    for (const auto& v : mesh.vertices)
        for (size_t a = 0; a < 3; ++a)
        {
            const double w = double(v[a]) + ob.p[a];
            b.minimum[a]   = std::min(b.minimum[a], w - extentMargin);
            b.maximum[a]   = std::max(b.maximum[a], w + extentMargin);
        }
    return b;
}
void placeVertices(Scene& s, size_t i)
{
    Sheet& sheet = s.sheets[i];
    auto&  mesh  = s.f.meshes[i];
    mesh.vertices.resize(sheet.vertexCount());
    sheet.placedStep    = s.step;
    const bool crumpled = s.mode == Deform::crumple && s.step > 0;
    const bool folded   = s.mode == Deform::fold && s.step > 0;
    for (size_t r = 0; r <= sheet.side; ++r)
        for (size_t c = 0; c <= sheet.side; ++c)
        {
            const size_t v = r * (sheet.side + 1) + c;
            const double x = sheetSize * double(c) / double(sheet.side), y = sheetSize * double(r) / double(sheet.side);
            double       p[3] = {x, y, waveZ(x, y, i, s.step)};
            if (crumpled)
                for (size_t a = 0; a < 3; ++a)
                    p[a] = (a == 2 ? 0.0 : p[a]) + crumpleAmplitude * unitNoise(i, v, s.step, a);
            else if (folded)
            {
                // Accordion: pleat k of width L / P lies at height k * gap, alternating direction, so
                // the eight pleats share one x range and a tree split by x while flat visits them all.
                // Each pleat is stretched to half the sheet's width so most rays over the sheet still hit.
                const double pleatWidth = sheetSize / double(foldPleats), stackWidth = sheetSize / 2;
                const double s_     = std::min(x / pleatWidth, double(foldPleats) - 1e-9);
                const size_t pleat  = size_t(s_);
                const double within = s_ - double(pleat);
                // The gap and ripple depend on the cell, so no two sheets share content.
                const double gap = 0.02 + 0.004 * double(s.step % 5) + 0.001 * double(i % 7);
                p[0]             = (pleat % 2 == 0 ? within : 1 - within) * stackWidth + 0.5 * (sheetSize - stackWidth);
                p[2]             = double(pleat) * gap + 0.002 * std::sin(6 * y + double(s.step) + 0.5 * double(i));
            }
            mesh.vertices[v] = {float(p[0]), float(p[1]), float(p[2])};
        }
}
// Advances the scene to `step`: picks the changed sheets (a rotating window of `changedPerStep`),
// applies the mode's change to them and refreshes their extents. Step 0 places every sheet.
void simulate(Scene& s, size_t step)
{
    s.step = step;
    s.changedNow.clear();
    s.topologyNow.reset();
    s.changedTriangles = 0;
    if (step == 0)
    {
        for (size_t i = 0; i < s.sheets.size(); ++i)
        {
            placeVertices(s, i);
            s.f.meshes[i].triangles = triangulate(s.sheets[i].side, false);
            s.f.objects[i].p        = s.sheets[i].cell;
            s.extents[i]            = sheetExtent(s, i);
        }
        return;
    }
    if (s.mode == Deform::none)
        return;
    const size_t start = (step * s.changedPerStep) % s.sheets.size();
    for (size_t j = 0; j < s.changedPerStep; ++j)
        s.changedNow.push_back((start + j) % s.sheets.size());
    for (size_t k = 0; k < s.changedNow.size(); ++k)
    {
        const size_t i     = s.changedNow[k];
        Sheet&       sheet = s.sheets[i];
        if (s.mode == Deform::rigid)
        {
            sheet.shift      = (step % 2) ? rigidShift : -rigidShift;
            s.f.objects[i].p = {sheet.cell[0] + sheet.shift, sheet.cell[1], sheet.cell[2]};
        }
        else
        {
            placeVertices(s, i);
            s.changedTriangles += sheet.triangleCount();
            if (s.mode == Deform::mixed && k == 0 && step % topologyEvery == 0)
            {
                sheet.flipped           = !sheet.flipped;
                s.f.meshes[i].triangles = triangulate(sheet.side, sheet.flipped);
                s.topologyNow           = i;
            }
        }
        s.extents[i] = sheetExtent(s, i);
    }
}
Scene buildScene(const Options& o)
{
    Scene s;
    s.mode           = deformMode(o);
    s.changedPerStep = s.mode == Deform::none ? 0 : std::max<size_t>(1, (o.objects * o.updatePercent + 50) / 100);
    s.sheets.resize(o.objects);
    s.f.meshes.resize(o.objects);
    s.f.objects.resize(o.objects);
    s.extents.resize(o.objects);
    s.f.kind = QueryKind::rays;
    // The installed CPU path returns PhysX float mesh contacts verbatim; the sheets are coarse and the
    // rays short, so a 0.5 mm contact tolerance covers them (the analytic checks stay tight).
    s.f.contactTolerance            = 5e-4;
    s.f.approximateBaselineContacts = true;
    for (size_t i = 0; i < o.objects; ++i)
    {
        Sheet& sheet = s.sheets[i];
        sheet.id     = i + 1;
        sheet.side   = sheetSides[i % (sizeof(sheetSides) / sizeof(sheetSides[0]))];
        sheet.cell   = {cellPitch * double(i % 8), cellPitch * double(i / 8), sheetHeight};
        Object& ob   = s.f.objects[i];
        ob.id = ob.owner          = sheet.id;
        ob.sphere                 = false;
        ob.q                      = identity;
        ob.mesh                   = i + 1;
        s.f.meshes[i].doubleSided = false;
    }
    // Rays round-robin over the cells: 7 of 8 per cell over the sheet's interior (safe from the rigid
    // shift), 1 of 8 beside it and bound to miss; 1 of 4 tilted by up to 0.06 rad so not every ray is
    // axis aligned.
    s.f.queries.resize(o.queries);
    for (size_t k = 0; k < o.queries; ++k)
    {
        const Sheet& sheet   = s.sheets[k % o.objects];
        const size_t ordinal = k / o.objects; // The ray's number within its cell.
        const double u = frac(double(k) * 0.6180339887498949), v = frac(double(k) * 0.7548776662466927);
        Query&       q = s.f.queries[k];
        q.p = {sheet.cell[0] + (ordinal % 8 == 7 ? 2.5 + 0.3 * u : 0.3 + 1.4 * u), sheet.cell[1] + 0.3 + 1.4 * v,
               rayHeight};
        q.nearDistance = 0;
        q.farDistance  = rayRange;
        q.bits         = 0;
        if (ordinal % 4 == 3)
        {
            const double dx = 0.06 * std::cos(double(k)), dy = 0.06 * std::sin(double(k));
            const double length = std::sqrt(dx * dx + dy * dy + 1);
            q.direction         = {float(dx / length), float(dy / length), float(-1 / length)};
        }
        else
            q.direction = {0, 0, -1};
    }
    simulate(s, 0);
    return s;
}

// The independent oracle: every triangle of the ray's own cell through the fixture's double-precision
// face evaluation (Moller-Trumbore, front faces only), closest within range. Sheets of other cells are at least two
// metres away.
struct OracleHit
{
    bool     hit      = false;
    uint64_t sheet    = 0;
    double   distance = 0;
    uint32_t triangle = 0;
    Vec      point{}, normal{};
};
// Unnormalized normal of a world triangle; its counter-clockwise side is the front.
Vec faceNormal(const std::array<Vec, 3>& c)
{
    const Vec e1{c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]};
    const Vec e2{c[2][0] - c[0][0], c[2][1] - c[0][1], c[2][2] - c[0][2]};
    return {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
}
double rayPlaneDistance(const Vec& o, const Vec& d, const std::array<Vec, 3>& c)
{
    const Vec    n           = faceNormal(c);
    const double denominator = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
    if (denominator == 0)
        return INFINITY;
    return (n[0] * (c[0][0] - o[0]) + n[1] * (c[0][1] - o[1]) + n[2] * (c[0][2] - o[2])) / denominator;
}
std::array<Vec, 3> worldTriangle(const Fixture& f, size_t sheetIndex, uint32_t triangle)
{
    const auto&        mesh = f.meshes[sheetIndex];
    const auto&        ob   = f.objects[sheetIndex];
    const auto&        tri  = mesh.triangles[triangle];
    std::array<Vec, 3> c{};
    for (size_t k = 0; k < 3; ++k)
        for (size_t a = 0; a < 3; ++a)
            c[k][a] = double(mesh.vertices[tri[k]][a]) + ob.p[a];
    return c;
}
OracleHit oracle(const Fixture& f, size_t rayIndex, size_t sheetIndex)
{
    const Query& q = f.queries[rayIndex];
    OracleHit    best;
    Hit          face{f.objects[sheetIndex].id};
    face.geometryType  = OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
    face.geometryValue = f.objects[sheetIndex].mesh;
    for (uint32_t tri = 0; tri < f.meshes[sheetIndex].triangles.size(); ++tri)
    {
        face.sourceFace = tri;
        const double t  = analyticFaceDistance(f, rayIndex, face);
        if (t < q.nearDistance || t > q.farDistance || (best.hit && t >= best.distance))
            continue;
        best.hit            = true;
        best.sheet          = face.id;
        best.distance       = t;
        best.triangle       = tri;
        best.normal         = faceNormal(worldTriangle(f, sheetIndex, tri));
        const double length = std::sqrt(best.normal[0] * best.normal[0] + best.normal[1] * best.normal[1] +
                                        best.normal[2] * best.normal[2]);
        for (size_t a = 0; a < 3; ++a)
        {
            best.normal[a] /= length;
            best.point[a] = q.p[a] + t * double(q.direction[a]);
        }
    }
    return best;
}
std::vector<OracleHit> oracleAll(const Scene& s)
{
    std::vector<OracleHit> hits(s.f.queries.size());
    for (size_t k = 0; k < hits.size(); ++k)
        hits[k] = oracle(s.f, k, k % s.sheets.size());
    return hits;
}
// Checks one lane's rows against the oracle: identities, completeness, distance, the face (or an
// analytically tied neighbour), the contact on the ray and the normal. `exact` lanes are the installed
// double-precision results; the direct PhysX lane's float contacts are compared on identity, distance
// within 1e-4 of the travelled distance and face only.
void checkAgainstOracle(const Scene& s, const Results& rows, const std::vector<OracleHit>& expected,
                        const std::string& label, bool exact)
{
    require(rows.size() == expected.size(), label + " row count");
    for (size_t k = 0; k < rows.size(); ++k)
    {
        const std::string where = label + " ray " + std::to_string(k);
        const auto&       hits  = rows[k].hits;
        const OracleHit&  e     = expected[k];
        require(hits.size() == (e.hit ? 1u : 0u),
                where + (e.hit ? " missed the sheet the oracle hits" : " hit where the oracle misses"));
        if (!e.hit)
            continue;
        const Hit& h = hits[0];
        require(h.id == e.sheet, where + " identity " + std::to_string(h.id) + " differs from the oracle's sheet " +
                                     std::to_string(e.sheet));
        require(h.geometryType == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, where + " geometry family");
        const double tolerance = exact ? 0 : 1e-4 * e.distance + 1e-5;
        require(near(h.distance, e.distance, tolerance), where + " distance " + std::to_string(h.distance) +
                                                             " differs from the oracle's " +
                                                             std::to_string(e.distance));
        const size_t sheetIndex = size_t(e.sheet - 1);
        if (h.sourceFace != e.triangle)
        {
            // A ray through a shared edge may report either face (float inside tests differ by a hair
            // there); the reported face's plane must then be crossed at the oracle's distance.
            require(h.sourceFace < s.f.meshes[sheetIndex].triangles.size(), where + " face out of range");
            const Query& q = s.f.queries[k];
            const Vec    d{double(q.direction[0]), double(q.direction[1]), double(q.direction[2])};
            const double t = rayPlaneDistance(q.p, d, worldTriangle(s.f, sheetIndex, uint32_t(h.sourceFace)));
            require(std::isfinite(t) && near(t, e.distance, tolerance),
                    where + " face " + std::to_string(h.sourceFace) + " is not the oracle's face " +
                        std::to_string(e.triangle) + " nor a neighbour crossed at the same distance");
        }
        if (!exact)
            continue;
        double offRay = 0, normalError = 0;
        for (size_t a = 0; a < 3; ++a)
        {
            offRay      = std::max(offRay, std::abs(h.position[a] - e.point[a]));
            normalError = std::max(normalError, std::abs(double(h.normal[a]) - e.normal[a]));
        }
        require(offRay <= s.f.contactTolerance, where + " contact " + std::to_string(offRay) + " m off the ray point");
        require(normalError <= 1e-3, where + " normal differs from the face normal by " + std::to_string(normalError));
        // The analytic surface: the contact must lie on the wave the sheet's points were last placed
        // on, within the chord error of the triangulation (the crumple is not analytic).
        if (s.mode != Deform::crumple && s.mode != Deform::fold)
        {
            const Sheet& sheet = s.sheets[sheetIndex];
            const double x = h.position[0] - s.f.objects[sheetIndex].p[0], y = h.position[1] - sheet.cell[1];
            const double z       = sheetHeight + waveZ(x, y, sheetIndex, sheet.placedStep);
            const double spacing = sheetSize / double(sheet.side), pi = std::acos(-1.0);
            const double chord = waveAmplitudeAt(sheetIndex, sheet.placedStep) * pi * pi * spacing * spacing / 4 + 1e-3;
            require(std::abs(h.position[2] - z) <= chord, where + " contact height " + std::to_string(h.position[2]) +
                                                              " is off the analytic wave " + std::to_string(z));
        }
    }
}

// The host side: a stage of Mesh prims, one per sheet, populated from usda and hierarchy-derived; per
// step the changed prims' points, indices or matrices and their world extents are rewritten and the
// ordinal sealed. The world extent is the host's duty on this pin, as in the other Ovstage fixtures.
class SheetStage
{
  public:
    ovstage_instance_t*                    stage      = nullptr;
    path_dictionary_instance_t*            dictionary = nullptr;
    std::vector<ovx_primpath_t>            paths;
    std::unordered_map<uint64_t, uint64_t> idByPath;
    ovstage_ordinal_t                      sealed = initialOrdinal;
    double                                 openMs = 0, hierarchyMs = 0;
    size_t                                 usdaBytes = 0;

    explicit SheetStage(const Scene& s)
    {
        // Value-initialized: Ovstage 0.4 adds `implementation`, which 0.3 lacks.
        ovstage_instance_desc_t desc{};
        desc.name = "ovspatialquery-scene-benchmark";
        require(ovstage_create_instance(&desc, &stage) == OVSTAGE_OK && stage != nullptr, "create ovstage instance");
        dictionary = ovstage_get_path_dictionary(stage);
        require(dictionary != nullptr, "ovstage path dictionary");
        const std::string source                            = usda(s);
        usdaBytes                                           = source.size();
        const auto                                openStart = Clock::now();
        const ovstage_population_enqueue_result_t population =
            openUsdWithPhysics(stage, ovx_string_t{source.data(), source.size()}, 1u, 0.0);
        require(population.status == OVSTAGE_OK, "populate stage from usda");
        ovstage_population_op_wait_result_t populated{};
        require(ovstage_population_wait_op(stage, population.op_index, OVSTAGE_TIMEOUT_INFINITE, &populated) ==
                        OVSTAGE_OK &&
                    populated.error_op_id_count == 0,
                "stage population failed");
        const ovstage_write_floor_desc_t floorOne{1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
        stageOp(stage, ovstage_advance_write_floor(stage, &floorOne), "seal ordinal 1");
        const auto hierarchyStart = Clock::now();
        stageOp(stage, ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u),
                "compute hierarchy");
        hierarchyMs = elapsed(hierarchyStart);
        std::vector<std::string>  names;
        std::vector<ovx_string_t> strings;
        for (const auto& sheet : s.sheets)
        {
            names.push_back("/World/cell" + std::to_string(sheet.id) + "/sheet");
            strings.push_back({names.back().data(), names.back().size()});
        }
        paths.assign(s.sheets.size(), OVX_INVALID_PRIMPATH);
        dictionaryOk(dictionary,
                     dictionary->vtable->create_paths_from_strings(dictionary->context, strings.data(), strings.size(),
                                                                   paths.data()),
                     "intern prim paths");
        for (size_t i = 0; i < paths.size(); ++i)
        {
            require(paths[i] != OVX_INVALID_PRIMPATH, "invalid interned prim path");
            require(idByPath.emplace(paths[i].raw_id, s.sheets[i].id).second, "duplicate prim path");
        }
        // Every sheet's conservative world extent at the first sealed ordinal.
        std::vector<size_t> all(s.sheets.size());
        for (size_t i = 0; i < all.size(); ++i)
            all[i] = i;
        writeExtents(s, all, initialOrdinal);
        const ovstage_write_floor_desc_t floorTwo{initialOrdinal, OVSTAGE_SCOPE_ALL, nullptr, 0u};
        stageOp(stage, ovstage_advance_write_floor(stage, &floorTwo), "seal ordinal 2");
        openMs = elapsed(openStart);
    }
    ~SheetStage()
    {
        if (stage != nullptr && ovstage_destroy_instance(stage) != OVSTAGE_OK)
        {
            std::cerr << "destroy ovstage instance failed during teardown\n";
            std::exit(EXIT_FAILURE);
        }
    }
    SheetStage(const SheetStage&) = delete;

    // Writes this step's changes at `target` and seals it: nothing for `none`, matrices and extents for
    // `rigid`, points and extents otherwise, plus the indices of a sheet whose topology was replaced.
    void author(const Scene& s, ovstage_ordinal_t target)
    {
        if (!s.changedNow.empty())
        {
            if (s.mode == Deform::rigid)
                writeMatrices(s, s.changedNow, target);
            else
                writePoints(s, s.changedNow, target);
            writeExtents(s, s.changedNow, target);
            if (s.topologyNow)
                writeIndices(s, *s.topologyNow, target);
        }
        const ovstage_write_floor_desc_t floor{target, OVSTAGE_SCOPE_ALL, nullptr, 0u};
        stageOp(stage, ovstage_advance_write_floor(stage, &floor), "seal ordinal");
        sealed = target;
    }

  private:
    std::vector<double>   matrices, extents;
    std::vector<DLTensor> tensors;
    std::vector<int64_t>  shapes;

    static std::string usda(const Scene& s)
    {
        std::string out = "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Z\"\n    metersPerUnit = 1\n)\n"
                          "def Xform \"World\"\n{\n";
        for (size_t i = 0; i < s.sheets.size(); ++i)
        {
            const Sheet& sheet = s.sheets[i];
            const Mesh&  mesh  = s.f.meshes[i];
            out += "    def Xform \"cell" + std::to_string(sheet.id) + "\"\n    {\n";
            out += "        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (";
            appendNumber(out, sheet.cell[0]);
            out += ", ";
            appendNumber(out, sheet.cell[1]);
            out += ", ";
            appendNumber(out, sheet.cell[2]);
            out += ", 1))\n        uniform token[] xformOpOrder = [\"xformOp:transform\"]\n";
            out += "        def Mesh \"sheet\"\n        {\n            uniform bool doubleSided = 0\n"
                   "            uniform token orientation = \"rightHanded\"\n"
                   "            uniform token subdivisionScheme = \"none\"\n            int[] faceVertexCounts = [";
            for (size_t t = 0; t < mesh.triangles.size(); ++t)
                out += t ? ", 3" : "3";
            out += "]\n            int[] faceVertexIndices = [";
            for (size_t t = 0; t < mesh.triangles.size(); ++t)
                for (size_t k = 0; k < 3; ++k)
                    out += (t || k ? ", " : "") + std::to_string(mesh.triangles[t][k]);
            out += "]\n            point3f[] points = [";
            for (size_t v = 0; v < mesh.vertices.size(); ++v)
            {
                out += v ? ", (" : "(";
                for (size_t a = 0; a < 3; ++a)
                {
                    if (a)
                        out += ", ";
                    appendNumber(out, mesh.vertices[v][a]);
                }
                out += ')';
            }
            Aabb local{{INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}};
            for (const auto& v : mesh.vertices)
                for (size_t a = 0; a < 3; ++a)
                {
                    local.minimum[a] = std::min(local.minimum[a], double(v[a]));
                    local.maximum[a] = std::max(local.maximum[a], double(v[a]));
                }
            out += "]\n            float3[] extent = [(";
            for (size_t a = 0; a < 3; ++a)
            {
                if (a)
                    out += ", ";
                appendNumber(out, local.minimum[a]);
            }
            out += "), (";
            for (size_t a = 0; a < 3; ++a)
            {
                if (a)
                    out += ", ";
                appendNumber(out, local.maximum[a]);
            }
            out += ")]\n        }\n    }\n";
        }
        out += "}\n";
        return out;
    }
    // A path-list query over the selected sheets, handed to `use`, then released.
    template <class Use>
    void withQuery(const std::vector<size_t>& selected, Use&& use)
    {
        std::vector<ovx_primpath_t> list;
        for (size_t i : selected)
            list.push_back(paths[i]);
        ovx_primpath_list_t pathList = OVX_INVALID_PRIMPATH_LIST;
        dictionaryOk(
            dictionary,
            dictionary->vtable->create_path_list_from_paths(dictionary->context, list.data(), list.size(), &pathList),
            "changed prim path list");
        ovstage_query_handle_t query = OVSTAGE_INVALID_QUERY_HANDLE;
        require(ovstage_query_from_path_list(stage, pathList, &query) == OVSTAGE_OK &&
                    query != OVSTAGE_INVALID_QUERY_HANDLE,
                "changed prim query");
        use(query);
        stageOp(stage, ovstage_release_query(stage, query), "release changed query");
        const ovx_api_result_t released =
            dictionary->vtable->release_path_list_reference(dictionary->context, pathList);
        dictionaryOk(dictionary, released, "release changed path list");
    }
    void writeExtents(const Scene& s, const std::vector<size_t>& selected, ovstage_ordinal_t target)
    {
        extents.resize(6 * selected.size());
        for (size_t k = 0; k < selected.size(); ++k)
        {
            const Aabb& b = s.extents[selected[k]];
            std::copy(b.minimum.begin(), b.minimum.end(), extents.begin() + ptrdiff_t(6 * k));
            std::copy(b.maximum.begin(), b.maximum.end(), extents.begin() + ptrdiff_t(6 * k + 3));
        }
        int64_t  rows = int64_t(selected.size());
        DLTensor tensor{};
        tensor.data               = extents.data();
        tensor.device.device_type = kDLCPU;
        tensor.ndim               = 1;
        tensor.dtype              = {kDLFloat, 64, 6};
        tensor.shape              = &rows;
        ovstage_attribute_write_t write{};
        write.attribute.string  = text(worldExtentName);
        write.data.tensors      = &tensor;
        write.data.tensor_count = 1;
        write.data.semantic     = OVSTAGE_SEMANTIC_NONE;
        withQuery(selected,
                  [&](ovstage_query_handle_t query)
                  {
                      stageOp(stage,
                              ovstage_write_attributes(stage, query, &write, 1, target, OVSTAGE_PRIM_MODE_UPSERT),
                              "write world extents");
                  });
    }
    void writeMatrices(const Scene& s, const std::vector<size_t>& selected, ovstage_ordinal_t target)
    {
        matrices.assign(16 * selected.size(), 0.0);
        for (size_t k = 0; k < selected.size(); ++k)
        {
            const Vec& p = s.f.objects[selected[k]].p;
            double*    m = matrices.data() + 16 * k;
            m[0] = m[5] = m[10] = m[15] = 1;
            m[12]                       = p[0];
            m[13]                       = p[1];
            m[14]                       = p[2];
        }
        int64_t  rows = int64_t(selected.size());
        DLTensor tensor{};
        tensor.data               = matrices.data();
        tensor.device.device_type = kDLCPU;
        tensor.ndim               = 1;
        tensor.dtype              = {kDLFloat, 64, 16};
        tensor.shape              = &rows;
        ovstage_attribute_write_t write{};
        write.attribute.string  = text(worldMatrixName);
        write.data.tensors      = &tensor;
        write.data.tensor_count = 1;
        write.data.semantic     = OVSTAGE_SEMANTIC_MATRIX;
        withQuery(selected,
                  [&](ovstage_query_handle_t query)
                  {
                      stageOp(stage,
                              ovstage_write_attributes(stage, query, &write, 1, target, OVSTAGE_PRIM_MODE_UPSERT),
                              "write world matrices");
                  });
    }
    // Per-row array transport: one point tensor per changed prim in one write.
    void writePoints(const Scene& s, const std::vector<size_t>& selected, ovstage_ordinal_t target)
    {
        tensors.assign(selected.size(), DLTensor{});
        shapes.resize(selected.size());
        for (size_t k = 0; k < selected.size(); ++k)
        {
            const Mesh& mesh              = s.f.meshes[selected[k]];
            shapes[k]                     = int64_t(mesh.vertices.size());
            tensors[k].data               = const_cast<std::array<float, 3>*>(mesh.vertices.data());
            tensors[k].device.device_type = kDLCPU;
            tensors[k].ndim               = 1;
            tensors[k].dtype              = {kDLFloat, 32, 3};
            tensors[k].shape              = &shapes[k];
        }
        ovstage_attribute_write_t write{};
        write.attribute.string  = text("points");
        write.data.tensors      = tensors.data();
        write.data.tensor_count = uint32_t(tensors.size());
        write.data.semantic     = OVSTAGE_SEMANTIC_POINT;
        write.data.is_array     = true;
        withQuery(selected,
                  [&](ovstage_query_handle_t query)
                  {
                      stageOp(stage,
                              ovstage_write_attributes(stage, query, &write, 1, target, OVSTAGE_PRIM_MODE_UPSERT),
                              "write points");
                  });
    }
    void writeIndices(const Scene& s, size_t sheetIndex, ovstage_ordinal_t target)
    {
        const Mesh& mesh  = s.f.meshes[sheetIndex];
        int64_t     shape = int64_t(mesh.triangles.size() * 3);
        DLTensor    tensor{};
        tensor.data               = const_cast<std::array<uint32_t, 3>*>(mesh.triangles.data());
        tensor.device.device_type = kDLCPU;
        tensor.ndim               = 1;
        tensor.dtype              = {kDLInt, 32, 1};
        tensor.shape              = &shape;
        ovstage_attribute_write_t write{};
        write.attribute.string  = text("faceVertexIndices");
        write.data.tensors      = &tensor;
        write.data.tensor_count = 1;
        write.data.semantic     = OVSTAGE_SEMANTIC_NONE;
        write.data.is_array     = true;
        withQuery({sheetIndex},
                  [&](ovstage_query_handle_t query)
                  {
                      stageOp(stage,
                              ovstage_write_attributes(stage, query, &write, 1, target, OVSTAGE_PRIM_MODE_UPSERT),
                              "write face vertex indices");
                  });
    }
};

// The installed lane: an Ovstage population over the sheets, a small explicit population of tool
// spheres hovering above them (one per cell, at staggered heights so that the breathing sheets enter and
// leave their pairs), a pair reporter between the two, and ray plans on the CPU and, when asked, CUDA.
class OvstageLane
{
  public:
    SheetStage                            stage;
    Spatial                               spatial;
    ovspatialquery_population_handle_t    tools        = OVSPATIALQUERY_INVALID_POPULATION;
    ovspatialquery_ordinal_t              toolsOrdinal = 0;
    ovspatialquery_pair_reporter_handle_t reporter     = OVSPATIALQUERY_INVALID_PAIR_REPORTER;
    std::unique_ptr<Spatial::BatchPlan>   cpuPlan, cudaPlan;
    std::vector<Vec>                      toolCenters;
    double                                populationMs = 0, toolsMs = 0;
    Publication                           initial;

    OvstageLane(const Scene& s, bool withCpu, bool withCuda) : stage(s), spatial(Spatial::InstanceOnly{})
    {
        static const ovx_string_t types[] = {text("Mesh")};
        ovstage_predicate_t       predicate{};
        predicate.attribute.string = text("usd-prim-type");
        predicate.op               = OVSTAGE_FILTER_OP_IN;
        predicate.values           = types;
        predicate.value_count      = 1;
        ovspatialquery_ovstage_population_desc_t desc{};
        desc.stage                            = stage.stage;
        desc.filter                           = ovstage_filter_t{&predicate, 1};
        desc.common.retained_ordinal_capacity = 2;
        const auto start                      = Clock::now();
        check(ovspatialquery_create_ovstage_population(spatial.instance, &desc, &spatial.population),
              "create ovstage population");
        const auto enqueueStart = Clock::now();
        initial                 = finishPublication(
            spatial,
            ovspatialquery_enqueue_ovstage_population_set(spatial.instance, spatial.population, initialOrdinal),
            enqueueStart, s.sheets.size(), "ovstage set");
        populationMs = elapsed(start);

        const auto toolsStart = Clock::now();
        Fixture    toolFixture;
        toolFixture.kind = QueryKind::rays;
        for (size_t i = 0; i < s.sheets.size(); ++i)
        {
            const Sheet& sheet = s.sheets[i];
            Object       ob{};
            ob.id = ob.owner = sheet.id;
            ob.sphere        = true;
            ob.q             = identity;
            ob.p             = {sheet.cell[0] + 1.0, sheet.cell[1] + 1.0, sheetHeight + 0.06 + 0.2 * double(i % 5) / 4};
            toolFixture.objects.push_back(ob);
            toolCenters.push_back(ob.p);
        }
        {
            uint64_t                                 id     = 1;
            double                                   radius = toolRadius;
            Input                                    ids(&id, 1, 0, kDLUInt, 64), radii(&radius, 1, 0, kDLFloat, 64);
            ovspatialquery_sphere_definition_table_t spheres{&ids.value, &radii.value};
            spatial.finish(ovspatialquery_enqueue_sphere_registration(spatial.instance, &spheres), false,
                           "tool sphere registration");
        }
        tools = spatial.publishPopulation(toolFixture, toolsOrdinal);
        ovspatialquery_pair_reporter_desc_t reporterDesc{};
        reporterDesc.first                = spatial.target(false);
        reporterDesc.second               = toolsRef();
        reporterDesc.active_pair_capacity = s.sheets.size();
        reporterDesc.delta_pair_capacity  = 2 * s.sheets.size();
        reporterDesc.result_slot_count    = 1;
        reporterDesc.contact_tolerance    = 0;
        reporterDesc.result_device        = {kDLCPU, 0};
        check(ovspatialquery_create_pair_reporter(spatial.instance, &reporterDesc, &reporter), "create pair reporter");
        toolsMs = elapsed(toolsStart);
        if (withCpu)
            cpuPlan = std::make_unique<Spatial::BatchPlan>(spatial, s.f, s.f.queries.size());
        if (withCuda)
            cudaPlan = std::make_unique<Spatial::BatchPlan>(spatial, s.f, s.f.queries.size(), DLDevice{kDLCUDA, 0});
    }
    ~OvstageLane()
    {
        cudaPlan.reset();
        cpuPlan.reset();
        if (reporter != OVSPATIALQUERY_INVALID_PAIR_REPORTER)
            Spatial::checkRelease(ovspatialquery_destroy_pair_reporter(spatial.instance, reporter),
                                  "destroy pair reporter");
        if (tools != OVSPATIALQUERY_INVALID_POPULATION)
            Spatial::checkRelease(ovspatialquery_destroy_population(spatial.instance, tools), "destroy tools");
        Spatial::checkRelease(
            ovspatialquery_destroy_geometry_definition(spatial.instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, 1}),
            "destroy tool sphere definition");
    }
    OvstageLane(const OvstageLane&) = delete;

    ovspatialquery_target_ref_t toolsRef() const
    {
        ovspatialquery_target_ref_t ref{};
        ref.kind              = OVSPATIALQUERY_TARGET_POPULATION;
        ref.handle.population = tools;
        return ref;
    }
    Publication step(const Scene& s, double& authoringMs)
    {
        const ovstage_ordinal_t target = stage.sealed + 1;
        const auto              start  = Clock::now();
        stage.author(s, target);
        authoringMs              = elapsed(start);
        const auto  enqueueStart = Clock::now();
        Publication p            = finishPublication(spatial,
                                                     ovspatialquery_enqueue_ovstage_population_update(
                                              spatial.instance, spatial.population, spatial.ordinal, target),
                                                     enqueueStart, s.sheets.size(), "ovstage update");
        require(p.result.published_ordinal == ovspatialquery_ordinal_t(target),
                "published ordinal does not follow the sealed ordinal");
        return p;
    }
    // Ovstage object identities are prim paths and geometry values are adapter-assigned references;
    // parity compares fixture identities and records the reference each sheet reported, so a step can
    // check which sheets kept their geometry and which took new content.
    void mapIdentities(Results& results, std::vector<uint64_t>* referenceBySheet = nullptr) const
    {
        for (auto& row : results)
            for (auto& hit : row.hits)
            {
                const auto found = stage.idByPath.find(hit.id);
                require(found != stage.idByPath.end(), "hit identity is not a stage prim path");
                hit.id = found->second;
                if (referenceBySheet != nullptr)
                {
                    uint64_t& recorded = referenceBySheet->at(size_t(hit.id - 1));
                    require(recorded == 0 || recorded == hit.geometryValue,
                            "one sheet reported two geometry references in one publication");
                    recorded = hit.geometryValue;
                }
                hit.geometryValue = hit.id; // The fixture's mesh index.
            }
    }
    struct Advance
    {
        PairSet entered, exited;
        double  ms = 0;
    };
    Advance advance()
    {
        const auto                                 start = Clock::now();
        ovspatialquery_pair_reporter_advance_t     selection{};
        ovspatialquery_pair_reporter_reservation_t reservation{};
        const ovspatialquery_enqueue_result_t      admitted =
            ovspatialquery_enqueue_pair_reporter_advance(spatial.instance, reporter, &selection, &reservation);
        check({admitted.status}, "enqueue reporter advance");
        ovspatialquery_op_wait_result_t waited{};
        check(ovspatialquery_wait_op(spatial.instance, admitted.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited),
              "wait reporter advance");
        if (waited.state != OVSPATIALQUERY_OP_COMPLETED)
        {
            const auto error = ovspatialquery_get_last_op_error(spatial.instance, admitted.op_id);
            throw std::runtime_error("reporter advance failed: " +
                                     std::string(error.ptr ? error.ptr : "", size_t(error.length)));
        }
        ovspatialquery_pair_reporter_result_t result{};
        ovspatialquery_op_id_t                op = admitted.op_id;
        check(ovspatialquery_fetch_pair_reporter(spatial.instance, op, &result), "fetch reporter result");
        require(result.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED, "reporter capacity outcome");
        require(result.first_source_ordinal == spatial.ordinal && result.second_source_ordinal == toolsOrdinal,
                "reporter ordinals differ from the heads");
        Advance a;
        a.entered = toSet(result.entered_pairs);
        a.exited  = toSet(result.exited_pairs);
        check(ovspatialquery_destroy_op(spatial.instance, &op), "destroy advance op");
        a.ms = elapsed(start);
        return a;
    }
    // The pair oracle: authored sheet extents against the tool spheres' boxes, in double precision.
    PairSet oraclePairs(const Scene& s) const
    {
        PairSet set;
        for (size_t i = 0; i < s.sheets.size(); ++i)
            for (size_t j = 0; j < toolCenters.size(); ++j)
            {
                bool overlap = true;
                for (size_t a = 0; a < 3 && overlap; ++a)
                    overlap = s.extents[i].minimum[a] <= toolCenters[j][a] + toolRadius &&
                              toolCenters[j][a] - toolRadius <= s.extents[i].maximum[a];
                if (overlap)
                    set.emplace_back(stage.paths[i].raw_id, j + 1);
            }
        std::sort(set.begin(), set.end());
        return set;
    }
};

// Direct pinned PhysX: one static actor per sheet in a scene, a ray per query. The `recook` variant
// cooks a fresh mesh for every changed sheet; the `refit` variant cooks non-quantized BVH34 trees once
// and refits them in place through getVerticesForModification/refitBVH, recooking only when the indices
// change. Both are raw baselines for what the pinned SDK offers, not snapshot-isolated publications.
class DirectSheets
{
  public:
    enum class Kind : uint8_t
    {
        recook, // A fresh cook for every changed sheet.
        refit,  // Non-quantized trees cooked once, refitted in place.
        clone,  // Non-quantized trees cooked once to bytes; each change loads a new mesh from them and refits it,
                // leaving the previous mesh untouched (the shape a snapshot-isolated library could take).
    };
    struct Variant
    {
        Kind                                kind  = Kind::recook;
        physx::PxScene*                     scene = nullptr;
        std::vector<physx::PxRigidStatic*>  actors;
        std::vector<physx::PxShape*>        shapes;
        std::vector<physx::PxTriangleMesh*> meshes;
        std::vector<std::vector<uint8_t>>   cookedBytes; // clone: the serialized cooked mesh per sheet
        double                              cookMs = 0, lastLoadMs = 0, lastRefitMs = 0;
        bool                                refit() const { return kind != Kind::recook; }
    };
    physx::PxDefaultAllocator      allocator;
    Errors                         errors;
    physx::PxFoundation*           foundation = nullptr;
    physx::PxPhysics*              physics    = nullptr;
    physx::PxDefaultCpuDispatcher* dispatcher = nullptr;
    physx::PxMaterial*             material   = nullptr;
    std::vector<Variant>           variants;
    const Scene&                   scene;

    DirectSheets(const Scene& s, bool recook, bool refit, bool clone = false) : scene(s)
    {
        foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
        require(foundation != nullptr, "PxCreateFoundation failed");
        physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, physx::PxTolerancesScale(), false);
        require(physics != nullptr, "PxCreatePhysics failed");
        dispatcher = physx::PxDefaultCpuDispatcherCreate(0);
        material   = physics->createMaterial(0.5f, 0.5f, 0);
        if (recook)
            addVariant(Kind::recook);
        if (refit)
            addVariant(Kind::refit);
        if (clone)
            addVariant(Kind::clone);
    }
    ~DirectSheets()
    {
        for (auto& v : variants)
        {
            for (auto* actor : v.actors)
                actor->release();
            for (auto* mesh : v.meshes)
                mesh->release();
            v.scene->release();
        }
        material->release();
        dispatcher->release();
        physics->release();
        foundation->release();
    }
    DirectSheets(const DirectSheets&) = delete;

    static physx::PxTriangleMeshDesc describe(const Mesh& mesh)
    {
        physx::PxTriangleMeshDesc desc;
        desc.points.count     = uint32_t(mesh.vertices.size());
        desc.points.stride    = sizeof(mesh.vertices[0]);
        desc.points.data      = mesh.vertices.data();
        desc.triangles.count  = uint32_t(mesh.triangles.size());
        desc.triangles.stride = sizeof(mesh.triangles[0]);
        desc.triangles.data   = mesh.triangles.data();
        return desc;
    }
    physx::PxCookingParams cookingParams(bool refit) const
    {
        physx::PxCookingParams params(physics->getTolerancesScale());
        // Preserve the authored triangle order and the 1-to-1 vertex mapping the refit path needs.
        params.meshPreprocessParams |= physx::PxMeshPreprocessingFlag::eDISABLE_CLEAN_MESH;
        if (refit)
            params.midphaseDesc.mBVH34Desc.quantized = false;
        return params;
    }
    physx::PxTriangleMesh* cook(const Mesh& mesh, bool refit)
    {
        auto* cooked =
            PxCreateTriangleMesh(cookingParams(refit), describe(mesh), physics->getPhysicsInsertionCallback());
        require(cooked != nullptr && cooked->getNbTriangles() == mesh.triangles.size(), "PhysX mesh cooking failed");
        return cooked;
    }
    // Cooks to bytes once; `load` turns those bytes into a fresh mesh whose vertices are then replaced.
    std::vector<uint8_t> cookBytes(const Mesh& mesh)
    {
        physx::PxDefaultMemoryOutputStream stream;
        require(PxCookTriangleMesh(cookingParams(true), describe(mesh), stream), "PhysX mesh cooking to bytes failed");
        return std::vector<uint8_t>(stream.getData(), stream.getData() + stream.getSize());
    }
    physx::PxTriangleMesh* load(const std::vector<uint8_t>& bytes, const Mesh& mesh)
    {
        physx::PxDefaultMemoryInputData input(const_cast<uint8_t*>(bytes.data()), uint32_t(bytes.size()));
        auto*                           loaded = physics->createTriangleMesh(input);
        require(loaded != nullptr && loaded->getNbTriangles() == mesh.triangles.size() &&
                    loaded->getNbVertices() == mesh.vertices.size(),
                "PhysX mesh load from cooked bytes failed");
        return loaded;
    }
    static void replaceVertices(physx::PxTriangleMesh& target, const Mesh& mesh)
    {
        physx::PxVec3* vertices = target.getVerticesForModification();
        require(vertices != nullptr, "PhysX refit vertices unavailable");
        for (size_t index = 0; index < mesh.vertices.size(); ++index)
            vertices[index] = meshVertex(mesh.vertices[index]);
        target.refitBVH();
    }
    void addVariant(Kind kind)
    {
        Variant            v;
        physx::PxSceneDesc desc(physics->getTolerancesScale());
        desc.cpuDispatcher   = dispatcher;
        desc.filterShader    = physx::PxDefaultSimulationFilterShader;
        desc.staticStructure = physx::PxPruningStructureType::eSTATIC_AABB_TREE;
        v.kind               = kind;
        v.scene              = physics->createScene(desc);
        require(v.scene != nullptr, "PhysX scene creation failed");
        const auto start = Clock::now();
        for (size_t i = 0; i < scene.sheets.size(); ++i)
        {
            physx::PxTriangleMesh* mesh = nullptr;
            if (kind == Kind::clone)
            {
                v.cookedBytes.push_back(cookBytes(scene.f.meshes[i]));
                mesh = load(v.cookedBytes.back(), scene.f.meshes[i]);
            }
            else
                mesh = cook(scene.f.meshes[i], v.refit());
            auto* actor = physics->createRigidStatic(pose(scene.f.objects[i].p, identity));
            auto* shape = physics->createShape(physx::PxTriangleMeshGeometry(mesh), *material, true);
            actor->attachShape(*shape);
            shape->release();
            actor->userData = const_cast<Sheet*>(&scene.sheets[i]);
            v.scene->addActor(*actor);
            v.meshes.push_back(mesh);
            v.actors.push_back(actor);
            v.shapes.push_back(shape);
        }
        v.scene->flushQueryUpdates();
        v.cookMs = elapsed(start);
        variants.push_back(v);
    }
    // Applies the scene's current step to one variant; returns the time spent.
    double update(Variant& v)
    {
        const auto start = Clock::now();
        v.lastLoadMs = v.lastRefitMs = 0;
        for (size_t i : scene.changedNow)
        {
            const Mesh& mesh = scene.f.meshes[i];
            if (scene.mode == Deform::rigid)
            {
                v.actors[i]->setGlobalPose(pose(scene.f.objects[i].p, identity));
                continue;
            }
            const bool topology = scene.topologyNow && *scene.topologyNow == i;
            if (v.kind == Kind::refit && !topology)
            {
                require(v.meshes[i]->getNbVertices() == mesh.vertices.size(), "PhysX refit vertex count");
                const auto refitStart = Clock::now();
                replaceVertices(*v.meshes[i], mesh);
                v.lastRefitMs += elapsed(refitStart);
                v.shapes[i]->setGeometry(physx::PxTriangleMeshGeometry(v.meshes[i]));
            }
            else if (v.kind == Kind::clone && !topology)
            {
                const auto loadStart = Clock::now();
                auto*      loaded    = load(v.cookedBytes[i], mesh);
                v.lastLoadMs += elapsed(loadStart);
                const auto refitStart = Clock::now();
                replaceVertices(*loaded, mesh);
                v.lastRefitMs += elapsed(refitStart);
                v.shapes[i]->setGeometry(physx::PxTriangleMeshGeometry(loaded));
                v.meshes[i]->release();
                v.meshes[i] = loaded;
            }
            else
            {
                physx::PxTriangleMesh* cooked = nullptr;
                if (v.kind == Kind::clone)
                {
                    v.cookedBytes[i] = cookBytes(mesh);
                    cooked           = load(v.cookedBytes[i], mesh);
                }
                else
                    cooked = cook(mesh, v.refit());
                v.shapes[i]->setGeometry(physx::PxTriangleMeshGeometry(cooked));
                v.meshes[i]->release();
                v.meshes[i] = cooked;
            }
        }
        v.scene->flushQueryUpdates();
        require(!errors.failed, "PhysX reported an update error");
        return elapsed(start);
    }
    void query(const Variant& v, Results& results) const
    {
        const physx::PxQueryFilterData fd(physx::PxQueryFlag::eSTATIC);
        const physx::PxHitFlags        hitFlags =
            physx::PxHitFlag::ePOSITION | physx::PxHitFlag::eNORMAL | physx::PxHitFlag::eFACE_INDEX;
        for (size_t i = 0; i < scene.f.queries.size(); ++i)
        {
            const auto& q    = scene.f.queries[i];
            auto&       hits = results[i].hits;
            hits.clear();
            const PxVec3           direction(q.direction[0], q.direction[1], q.direction[2]);
            physx::PxRaycastBuffer buffer;
            v.scene->raycast(px(q.p), direction, float(q.farDistance), buffer, hitFlags, fd);
            if (!buffer.hasBlock)
                continue;
            const auto&  h     = buffer.block;
            const Sheet& sheet = *static_cast<const Sheet*>(h.actor->userData);
            const auto*  mesh  = v.meshes[size_t(sheet.id - 1)];
            require(h.faceIndex < mesh->getNbTriangles(), "invalid cooked face index");
            const auto* remap = mesh->getTrianglesRemap();
            hits.push_back({sheet.id,
                            double(h.distance),
                            {h.position.x, h.position.y, h.position.z},
                            {h.normal.x, h.normal.y, h.normal.z},
                            OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH,
                            sheet.id,
                            remap ? remap[h.faceIndex] : h.faceIndex});
        }
    }
};

void deformParity(const Options& o, Scene& s)
{
    std::string           cudaMessage;
    const bool            cuda = cudaPlansAvailable(cudaMessage);
    DirectSheets          direct(s, true, true, true);
    OvstageLane           lane(s, true, cuda);
    Results               rows(s.f.queries.size()), retainedRows(s.f.queries.size());
    Scene                 previous = s; // The retained head's geometry, checked after every publication.
    PairSet               accepted;
    std::vector<uint64_t> previousReferences;
    std::vector<ovspatialquery_ordinal_t> published{lane.initial.result.published_ordinal};
    size_t                                topologyReplacements = 0, checkedRays = 0, checkedHits = 0, pairsSeen = 0;
    const auto                            checkLanes = [&](size_t step)
    {
        const std::string            where    = "step " + std::to_string(step) + " ";
        const std::vector<OracleHit> expected = oracleAll(s);
        for (const auto& e : expected)
            checkedHits += e.hit ? 1 : 0;
        checkedRays += expected.size();
        for (const auto& v : direct.variants)
        {
            direct.query(v, rows);
            const char* name = v.kind == DirectSheets::Kind::recook  ? "physx"
                               : v.kind == DirectSheets::Kind::refit ? "physx-refit"
                                                                     : "physx-clone";
            checkAgainstOracle(s, rows, expected, where + name, false);
        }
        lane.cpuPlan->batch(s.f, rows);
        std::vector<uint64_t> references(s.sheets.size(), 0);
        lane.mapIdentities(rows, &references);
        checkAgainstOracle(s, rows, expected, where + "ovstage CPU plan", true);
        // Geometry references: an unchanged or rigidly moved sheet keeps its reference, a sheet whose
        // points or indices changed takes a new one (every sheet is hit by at least one ray).
        for (size_t i = 0; i < s.sheets.size(); ++i)
        {
            require(references[i] != 0, where + "sheet " + std::to_string(i + 1) + " was not hit by any ray");
            if (previousReferences.empty())
                continue;
            const bool contentChanged =
                s.mode != Deform::rigid && std::find(s.changedNow.begin(), s.changedNow.end(), i) != s.changedNow.end();
            require((references[i] != previousReferences[i]) == contentChanged,
                    where + "sheet " + std::to_string(i + 1) +
                        (contentChanged ? " kept its geometry reference although its content changed"
                                        : " took a new geometry reference without a content change"));
        }
        previousReferences = references;
        // The shared comparison adds its own face-plane, on-ray and analytic-face checks.
        Results physxRows(s.f.queries.size());
        direct.query(direct.variants[0], physxRows);
        compare(s.f, physxRows, rows, where + "physx/ovstage", false, false);
        if (lane.cudaPlan)
        {
            Results gpu(s.f.queries.size());
            lane.cudaPlan->batch(s.f, gpu);
            lane.mapIdentities(gpu);
            checkAgainstOracle(s, gpu, expected, where + "ovstage CUDA plan", true);
            compare(s.f, rows, gpu, where + "CPU/CUDA", false, true);
        }
        // Pairs: the reporter's deltas against the oracle's set difference, and the accepted set.
        const PairSet oraclePairs = lane.oraclePairs(s);
        const auto    a           = lane.advance();
        PairSet       entered, exited;
        std::set_difference(oraclePairs.begin(), oraclePairs.end(), accepted.begin(), accepted.end(),
                            std::back_inserter(entered));
        std::set_difference(accepted.begin(), accepted.end(), oraclePairs.begin(), oraclePairs.end(),
                            std::back_inserter(exited));
        requireEqual(entered, a.entered, where + "entered pairs");
        requireEqual(exited, a.exited, where + "exited pairs");
        accepted = oraclePairs;
        pairsSeen += oraclePairs.size();
    };
    checkLanes(0);
    for (size_t step = 1; step <= paritySteps; ++step)
    {
        previous = s;
        simulate(s, step);
        topologyReplacements += s.topologyNow ? 1 : 0;
        for (auto& v : direct.variants)
            direct.update(v);
        double     authoring = 0;
        const auto p         = lane.step(s, authoring);
        published.push_back(p.result.published_ordinal);
        checkLanes(step);
        // The retained head must still answer with the geometry it was published with, whatever the
        // new head shares with it.
        const std::vector<OracleHit> retainedExpected = oracleAll(previous);
        lane.cpuPlan->batch(s.f, retainedRows, ovspatialquery_ordinal_t(published[published.size() - 2]));
        lane.mapIdentities(retainedRows);
        checkAgainstOracle(previous, retainedRows, retainedExpected,
                           "step " + std::to_string(step) + " retained head, ovstage CPU plan", true);
        if (lane.cudaPlan)
        {
            lane.cudaPlan->batch(s.f, retainedRows, ovspatialquery_ordinal_t(published[published.size() - 2]));
            lane.mapIdentities(retainedRows);
            checkAgainstOracle(previous, retainedRows, retainedExpected,
                               "step " + std::to_string(step) + " retained head, ovstage CUDA plan", true);
        }
    }
    uint32_t major = 0, minor = 0, patch = 0;
    ovstage_get_version(lane.stage.stage, &major, &minor, &patch);
    size_t triangles = 0;
    for (const auto& mesh : s.f.meshes)
        triangles += mesh.triangles.size();
    std::cout << "{\"backend\":\"parity\",\"workload\":\"" << o.workload << "\",\"mode\":\"" << modeName(s.mode)
              << "\",\"cells\":" << o.objects << ",\"queries\":" << o.queries << ",\"triangles\":" << triangles
              << ",\"changed_per_step\":" << s.changedPerStep << ",\"steps\":" << paritySteps
              << ",\"topology_replacements\":" << topologyReplacements << ",\"checked_rays\":" << checkedRays
              << ",\"checked_hits\":" << checkedHits << ",\"pairs_seen\":" << pairsSeen
              << ",\"parity\":true,\"triangle_oracle\":true,\"analytic_wave\":"
              << (s.mode != Deform::crumple && s.mode != Deform::fold ? "true" : "false")
              << ",\"retained_head_checked\":true,\"cuda_parity\":\""
              << (cuda ? "parity" : "unavailable: " + cudaMessage) << "\",\"ovstage_version\":[" << major << ','
              << minor << ',' << patch << "],\"published_ordinals\":[";
    for (size_t i = 0; i < published.size(); ++i)
        std::cout << (i ? "," : "") << published[i];
    std::cout << "],\"physx_version\":" << PX_PHYSICS_VERSION << ",\"physx_config\":\"" << BENCH_PHYSX_CONFIG
              << "\",\"identity_mapping\":\"prim path to fixture identity through the stage path dictionary\"}\n";
}
} // namespace

bool deformWorkload(const Options& o) { return o.workload.rfind("deform-", 0) == 0; }

int deformRun(const Options& o)
{
    Scene s = buildScene(o);
    if (o.backend == "parity")
    {
        deformParity(o, s);
        return 0;
    }
    const bool physxLane      = o.backend == "physx" || o.backend == "physx-refit" || o.backend == "physx-clone";
    const bool cudaLane       = o.backend == "ovstage-cuda";
    const auto memoryBaseline = memorySnapshot(OVSPATIALQUERY_INVALID_INSTANCE);
    const auto setupStart     = Clock::now();
    std::unique_ptr<DirectSheets> direct;
    std::unique_ptr<OvstageLane>  lane;
    if (physxLane)
        direct = std::make_unique<DirectSheets>(s, o.backend == "physx", o.backend == "physx-refit",
                                                o.backend == "physx-clone");
    else
        lane = std::make_unique<OvstageLane>(s, !cudaLane, cudaLane);
    const double setupMs      = elapsed(setupStart);
    const auto   instance     = lane ? lane->spatial.instance : OVSPATIALQUERY_INVALID_INSTANCE;
    const auto   deviceCharge = [&]() -> std::optional<uint64_t>
    {
        if (!cudaLane)
            return std::nullopt;
        uint64_t bytes = 0;
        check(ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCUDA, 0}, &bytes), "device usage");
        return bytes;
    };
    const auto          memoryReady = memorySnapshot(instance);
    Spatial::BatchPlan* plan        = lane ? (cudaLane ? lane->cudaPlan.get() : lane->cpuPlan.get()) : nullptr;
    Results             result(o.queries);
    Observation         observation(o.queries);
    for (auto& row : result)
        row.hits.reserve(1);
    const auto query = [&](ovspatialquery_ordinal_t exact = -1)
    {
        if (direct)
            direct->query(direct->variants[0], result);
        else
            plan->batch(s.f, result, exact);
    };
    // First use: the cold ray batch against the initial head, then the first reporter advance.
    const auto firstUseStart = Clock::now();
    query();
    const double firstUseQueryMs = elapsed(firstUseStart);
    double       firstPairsMs    = 0;
    if (lane)
        firstPairsMs = lane->advance().ms;
    observation.consume(s.f, result, false);
    const double firstUseChecksum = observation.checksum;
    observation                   = Observation(o.queries);
    const auto memoryFirstUse     = memorySnapshot(instance);
    struct StepSample
    {
        double simulate = 0, authoring = 0, capture = 0, readiness = 0, publish = 0, firstQuery = 0, warmQuery = 0,
               retainedQuery = 0, pairs = 0, observation = 0, cudaEnqueue = 0, cudaWait = 0, cudaFetch = 0,
               cudaReadback               = 0;
        double                  cloneLoad = 0, cloneRefit = 0;
        size_t                  changedMeshes = 0, changedTriangles = 0, topology = 0, hits = 0;
        uint64_t                hostCharged = 0;
        std::optional<uint64_t> deviceCharged;
    };
    std::vector<StepSample>  samples;
    ovspatialquery_ordinal_t previousOrdinal = lane ? lane->spatial.ordinal : 0;
    for (size_t frame = 0; frame < o.frames + o.warmup; ++frame)
        try
        {
            StepSample sample;
            auto       start = Clock::now();
            simulate(s, frame + 1);
            sample.simulate         = elapsed(start);
            sample.changedMeshes    = s.changedNow.size();
            sample.changedTriangles = s.changedTriangles;
            sample.topology         = s.topologyNow ? 1 : 0;
            if (direct)
            {
                sample.publish    = direct->update(direct->variants[0]);
                sample.cloneLoad  = direct->variants[0].lastLoadMs;
                sample.cloneRefit = direct->variants[0].lastRefitMs;
            }
            else
            {
                const auto p     = lane->step(s, sample.authoring);
                sample.capture   = p.capture;
                sample.readiness = p.readiness;
                sample.publish   = p.capture + p.readiness;
            }
            start = Clock::now();
            query();
            sample.firstQuery = elapsed(start);
            for (const auto& row : result)
                sample.hits += row.hits.empty() ? 0 : 1;
            if (plan)
            {
                sample.cudaEnqueue  = plan->last.enqueue;
                sample.cudaWait     = plan->last.wait;
                sample.cudaFetch    = plan->last.fetch;
                sample.cudaReadback = plan->last.readback;
            }
            start = Clock::now();
            observation.consume(s.f, result, false);
            sample.observation = elapsed(start);
            start              = Clock::now();
            query();
            sample.warmQuery = elapsed(start);
            if (lane)
            {
                start = Clock::now();
                query(previousOrdinal);
                sample.retainedQuery = elapsed(start);
                sample.pairs         = lane->advance().ms;
                previousOrdinal      = lane->spatial.ordinal;
                const auto snapshot  = memorySnapshot(instance);
                sample.hostCharged   = snapshot.ovspatialqueryHostChargedBytes.value_or(0);
                sample.deviceCharged = deviceCharge();
            }
            if (frame >= o.warmup)
                samples.push_back(sample);
        }
        catch (const std::exception& error)
        {
            throw std::runtime_error("frame " + std::to_string(frame + 1) + ": " + error.what());
        }
    const auto memorySteady = memorySnapshot(instance);
    uint32_t   major = 0, minor = 0, patch = 0;
    check(ovspatialquery_get_version(&major, &minor, &patch), "runtime version");
    uint32_t stageMajor = 0, stageMinor = 0, stagePatch = 0;
    if (lane)
        ovstage_get_version(lane->stage.stage, &stageMajor, &stageMinor, &stagePatch);
    size_t triangles = 0;
    for (const auto& mesh : s.f.meshes)
        triangles += mesh.triangles.size();
    std::cout << std::setprecision(12) << "{\"backend\":\"" << o.backend << "\",\"workload\":\"" << o.workload
              << "\",\"mode\":\"" << modeName(s.mode) << "\",\"objects\":" << o.objects << ",\"cells\":" << o.objects
              << ",\"triangles\":" << triangles << ",\"queries\":" << o.queries << ",\"frames\":" << o.frames
              << ",\"warmup\":" << o.warmup << ",\"update_percent\":" << o.updatePercent
              << ",\"changed_per_step\":" << s.changedPerStep
              << ",\"synthetic_analogue\":true,\"physx_version\":" << PX_PHYSICS_VERSION << ",\"physx_config\":\""
              << BENCH_PHYSX_CONFIG << "\",\"spatial_version\":[" << major << ',' << minor << ',' << patch
              << "],\"ovstage_version\":";
    if (lane)
        std::cout << '[' << stageMajor << ',' << stageMinor << ',' << stagePatch << ']';
    else
        std::cout << "null";
    std::cout << ",\"host_budget_bytes\":" << hostBudget << ",\"query_threads\":1,\"setup_ms\":" << setupMs
              << ",\"stage_open_ms\":" << (lane ? std::to_string(lane->stage.openMs) : "null")
              << ",\"stage_hierarchy_ms\":" << (lane ? std::to_string(lane->stage.hierarchyMs) : "null")
              << ",\"stage_usda_bytes\":" << (lane ? std::to_string(lane->stage.usdaBytes) : "null")
              << ",\"geometry_setup_ms\":" << (direct ? direct->variants[0].cookMs : 0)
              << ",\"publication_ms\":" << (lane ? lane->populationMs : direct->variants[0].cookMs)
              << ",\"tools_setup_ms\":" << (lane ? std::to_string(lane->toolsMs) : "null")
              << ",\"initial_capture_ms\":" << (lane ? std::to_string(lane->initial.capture) : "null")
              << ",\"initial_readiness_ms\":" << (lane ? std::to_string(lane->initial.readiness) : "null")
              << ",\"first_use_query_ms\":" << firstUseQueryMs << ",\"first_use_pairs_ms\":" << firstPairsMs
              << ",\"first_use_checksum\":" << firstUseChecksum << ",\"update_scope\":\""
              << (direct ? (o.backend == "physx-refit"
                                ? "publish_ms refits the changed cooked meshes in place (recooks on an index rewrite), "
                                  "setGeometry and flushQueryUpdates"
                            : o.backend == "physx-clone"
                                ? "publish_ms loads a new mesh from the sheet's cooked bytes (load_ms), replaces its "
                                  "vertices and refits (refit_ms), setGeometry and flushQueryUpdates; the previous "
                                  "mesh is untouched until released"
                                : "publish_ms cooks a fresh PhysX mesh per changed sheet, setGeometry and "
                                  "flushQueryUpdates")
                         : "authoring_ms is the host's stage writes and seal, outside frame_ms; capture_ms is the "
                           "synchronous enqueue that acquires the sealed ordinal; readiness_ms is wait, fetch and "
                           "destruction; first_query_ms is the ray batch on the new head, warm_query_ms the same "
                           "batch again, retained_query_ms the batch on the previous ordinal; pairs_ms is one "
                           "reporter advance against the tool population")
              << "\",\"frame_scope\":\"publish + first query + pairs + observation\""
              << ",\"memory\":";
    printMemoryRecord(
        lane != nullptr, lane ? "the process also holds the Ovstage runtime and stage" : "", memoryBaseline,
        {{"ready", memoryReady, true}, {"first_use_ready", memoryFirstUse, false}, {"steady", memorySteady, true}});
    std::cout << ",\"checksum\":" << observation.checksum << ",\"samples\":[";
    for (size_t i = 0; i < samples.size(); ++i)
    {
        const auto& sm = samples[i];
        std::cout << (i ? "," : "") << "{\"simulate_ms\":" << sm.simulate << ",\"authoring_ms\":" << sm.authoring
                  << ",\"capture_ms\":" << sm.capture << ",\"readiness_ms\":" << sm.readiness
                  << ",\"publish_ms\":" << sm.publish << ",\"update_ms\":" << sm.publish
                  << ",\"first_query_ms\":" << sm.firstQuery << ",\"warm_query_ms\":" << sm.warmQuery
                  << ",\"query_ms\":" << sm.warmQuery << ",\"observation_ms\":" << sm.observation
                  << ",\"frame_ms\":" << sm.publish + sm.firstQuery + sm.pairs + sm.observation
                  << ",\"changed_meshes\":" << sm.changedMeshes << ",\"changed_triangles\":" << sm.changedTriangles
                  << ",\"topology_replaced\":" << sm.topology << ",\"hits\":" << sm.hits;
        if (direct && direct->variants[0].kind != DirectSheets::Kind::recook)
            std::cout << ",\"load_ms\":" << sm.cloneLoad << ",\"refit_ms\":" << sm.cloneRefit;
        if (lane)
        {
            std::cout << ",\"retained_query_ms\":" << sm.retainedQuery << ",\"pairs_ms\":" << sm.pairs
                      << ",\"host_charged_bytes\":" << sm.hostCharged;
            if (sm.deviceCharged)
                std::cout << ",\"device_charged_bytes\":" << *sm.deviceCharged
                          << ",\"cuda_enqueue_ms\":" << sm.cudaEnqueue << ",\"cuda_wait_ms\":" << sm.cudaWait
                          << ",\"cuda_fetch_ms\":" << sm.cudaFetch << ",\"cuda_readback_ms\":" << sm.cudaReadback;
        }
        std::cout << '}';
    }
    std::cout << "]}\n";
    return 0;
}
} // namespace bench
