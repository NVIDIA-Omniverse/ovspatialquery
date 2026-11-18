/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_CUDA_FAMILY_PLAN_TYPES_H
#define OVSPATIALQUERY_CUDA_FAMILY_PLAN_TYPES_H

namespace ovspatialquery
{
namespace detail
{
namespace cuda_family
{

enum : unsigned int
{
    kFamilyRaycast     = 0u,
    kFamilySphereSweep = 1u,
    kFamilyOverlap     = 2u,
    kFamilyBoxSweep    = 3u,
};

enum : unsigned int
{
    kPrecisionBoundsCandidate = 0u,
    kPrecisionExact           = 1u,
    kPrecisionExactSingle     = 2u,
};

enum : unsigned int
{
    kHitModeClosest  = 0u,
    kHitModeAny      = 1u,
    kHitModeMultiple = 2u,
};

enum : unsigned int
{
    kOverlapSphere = 0u,
    kOverlapBox    = 1u,
    kOverlapAabb   = 2u,
};

// POD records describing the device-side exact geometry staging of the
// catalogue plan.
enum : unsigned int
{
    kGeometrySphere       = 0u,
    kGeometryBox          = 1u,
    kGeometryCapsule      = 2u,
    kGeometryConvex       = 3u,
    kGeometryTriangleMesh = 4u,
    // A slot of a device-prepared version that holds no object (removed, or not
    // yet assigned when the version was published): no exact test hits it, as
    // its inverted bounds do not reliably fail every candidate gate.
    kGeometryAbsent = 5u,
};

enum : unsigned int
{
    kGeometryFlagDoubleSided = 1u,
};

enum : unsigned int
{
    kBatchRowSuccess         = 0u,
    kBatchRowInvalidArgument = 1u,
    kBatchRowFailed          = 3u,
};

enum : unsigned int
{
    kResultPosition         = 1u,
    kResultNormal           = 2u,
    kResultDistance         = 4u,
    kResultSourceFeature    = 8u,
    kResultGeometryIdentity = 16u,
    kResultQueryPosition    = 32u,
    kResultMaterial         = 64u,
};

enum : unsigned int
{
    kFeatureNone     = 0u,
    kFeatureTriangle = 1u,
};

struct alignas(8) ExactObjectHeader
{
    unsigned int       kind;
    unsigned int       flags;
    unsigned long long familyIndex;
};

struct alignas(8) SphereGeometry
{
    double center[3];
    double radius;
};

struct alignas(8) BoxGeometry
{
    double center[3];
    double axes[9];
    double halfExtents[3];
};

struct alignas(8) CapsuleGeometry
{
    double endpointA[3];
    double endpointB[3];
    double radius;
};

struct alignas(8) ConvexGeometry
{
    unsigned long long vertexOffset;
    unsigned long long vertexCount;
    unsigned long long triangleOffset;
    unsigned long long triangleCount;
    unsigned long long planeOffset;
    unsigned long long planeCount;
    unsigned long long edgeOffset;
    unsigned long long edgeCount;
    // Convex arrays include the linear pose; translation is applied only at query boundaries.
    double translation[3];
};

struct alignas(8) MeshGeometry
{
    unsigned long long definitionIndex;
    // Row-major affine world-from-local (three rows of four doubles).
    double worldFromLocal[12];
    // Row-major inverse linear transform. Translation is subtracted before use.
    double localFromWorldLinear[9];
    // The object's per-source-face materials in the catalogue's face material
    // table (ADR-0066); a zero count when the definition assigns none.
    unsigned long long faceMaterialOffset;
    unsigned long long faceMaterialCount;
};

// One resident mesh definition: device addresses of its local arrays and BVH.
// Definitions live independently of any one catalogue so successive generations
// share them; instances index this table. Node children and leaf triangle runs
// are indices local to the definition's own arrays.
struct alignas(8) MeshDefinition
{
    unsigned long long vertices;           // float[vertexCount * 3]
    unsigned long long triangles;          // unsigned int[triangleCount * 3]
    unsigned long long sourceFaces;        // unsigned long long[triangleCount]
    unsigned long long sourceCorners;      // unsigned int[triangleCount * 3]
    unsigned long long nodes;              // MeshBvhNode[nodeCount]
    unsigned long long traversalTriangles; // unsigned int[triangleCount], leaf order
    unsigned long long vertexCount;
    unsigned long long triangleCount;
    unsigned long long nodeCount;
};

struct alignas(8) MeshBvhNode
{
    float              minimum[3];
    float              maximum[3];
    unsigned long long first;
    unsigned int       count; // nonzero for a leaf; zero means children `first` and `first + 1`
    unsigned int       reserved;
};

// Launch shape of the family kernel: one block of kEvaluateThreads per query,
// then one kScanThreads block that packs the rows, then the output pass. The
// evaluation kernel asks the compiler to fit kEvaluateBlocksPerMultiprocessor of
// its single-warp blocks on one multiprocessor (128 registers per thread on a
// 64K-register multiprocessor), and the shared memory of a block admits about this
// many. Raising residency from 8 to 14 blocks paid once the exact predicates ran
// only on culled candidates; 16 and 24 blocks (a smaller block, 80 registers)
// measured the same, because the double pipe, not memory latency, bounds the
// remaining exact tests.
enum : unsigned int
{
    kEvaluateThreads                 = 32u,
    kEvaluateBlocksPerMultiprocessor = 16u,
    kScanThreads                     = 1024u,
};

enum : unsigned int
{
    kPassEvaluate = 0u,
    kPassWrite    = 1u,
};

// A hashed uniform grid over one device-prepared version's bounds (ADR-0024),
// built by the publication's kernels after the bounds. Each
// slot whose box fits two cells along every axis is linked into the bucket of
// its minimum corner's cell; the others are listed as oversize; tombstoned
// slots are in neither. The header is followed in the version's allocation by
// heads[bucketMask + 1], next[capacity], cellKeys[3 * capacity] and
// oversize[capacity]. The cell width is a power of two, so a coordinate times
// `inverseCell` is exact and the cells a query covers are exact as well.
struct alignas(8) DeviceBoxGrid
{
    unsigned int oversizeComplement; // ~oversizeCount, so one fill of all ones empties the heads and the list
    unsigned int bucketMask;
    double       inverseCell;
    unsigned int capacity;
    unsigned int reserved;
};

// One registered convex hull or triangle mesh a device-prepared population
// references: the hull's local vertices resident on the
// device, or the mesh's local bounds, from which the prepare kernel derives
// the same world AABB the host derives (ADR-0023).
struct alignas(8) DeviceShapeRecord
{
    unsigned long long vertices;       // device double[vertexCount][3], convex only
    unsigned int       vertexCount;    // convex only
    double             localBounds[6]; // minimum[3], maximum[3], mesh only
};

// Where a convex instance of an exact catalogue derived on the device (BH-065)
// finds its hull's local vertices and planes among the definitions' arrays.
struct alignas(8) ExactConvexSource
{
    unsigned long long localVertexOffset; // vertices of three doubles
    unsigned long long localPlaneOffset;  // planes of four doubles
};

// The one by-value parameter of the kernel that derives a readable explicit
// device-prepared version's exact catalogue: what its host copy would hold,
// from each slot's pose in the version and its binding in the slot history.
// The headers, the instance templates (offsets and indices already set) and
// the sources come from the history and are written once per membership; the
// kernel writes every other table for the version.
struct alignas(8) ExactCatalogueRecord
{
    unsigned long long       slotCount;
    const double*            poses;           // the version's: translation, rotation xyzw, scale per slot
    const double*            versionMinimums; // the version's; not finite for a tombstoned slot
    const unsigned int*      versionFailed;   // the version's status word
    const double*            params;          // three authored parameters per slot
    const unsigned int*      definitions;     // each slot's record index
    const DeviceShapeRecord* records;
    const ExactObjectHeader* headers;
    const ConvexGeometry*    convexTemplates;
    const ExactConvexSource* convexSources;
    const double*            localVertices;
    const double*            localPlanes;
    const MeshGeometry*      meshTemplates;
    ExactObjectHeader*       objects; // the version's headers: the history's, or absent
    double*                  minimums;
    double*                  maximums;
    unsigned long long*      materials; // null unless the plan asks for materials
    SphereGeometry*          spheres;
    BoxGeometry*             boxes;
    CapsuleGeometry*         capsules;
    ConvexGeometry*          convexes;
    double*                  convexVertices;
    double*                  convexPlanes;
    MeshGeometry*            meshes;
    unsigned int*            failed; // the version's status, and nonzero where the host copy would fail
};

// One slot of an Ovstage population prepared on the device (ADR-0047): how the
// gather kernel turns the prim's world matrix into the slot's pose. K and its
// inverse describe the prepared shape; the candidate placement must be rigid
// before that shape can be retained.
enum : unsigned int
{
    kOvstageRuleBox       = 0u, // orthogonal axes, independent scales
    kOvstageRuleUniform   = 1u, // orthogonal axes, one scale (sphere, capsule)
    kOvstageRuleRigid     = 2u, // right-handed rigid pose, one scale (convex, mesh)
    kOvstageFrameIdentity = 0u, // the shape's local axes are the prim's
    kOvstageFrameFromX    = 1u, // a capsule along local X: the device capsule's Y is the prim's X
    kOvstageFrameFromZ    = 2u, // a capsule along local Z
};

struct alignas(8) OvstageSlotRecord
{
    unsigned long long id;       // the prim's path value, the object ID
    double             scale[3]; // the prim's axis scales at its last membership change
    unsigned int       rule;     // kOvstageRule*
    unsigned int       frame;    // kOvstageFrame*
    double             k[9]       = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double             inverse[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double             size       = 0.0; // authored Cube size
    double             radius     = 0.0; // authored Sphere/Capsule radius
    double             halfHeight = 0.0; // authored Capsule half height
};

enum : unsigned int
{
    kGridHeaderBytes = 32u,
    kGridEmpty       = 0xffffffffu,
    // A query covering more cells than this, or than the grid has slots, scans
    // every object instead: walking a cell costs about what testing a slot
    // does, so the walk pays only while it visits fewer cells than there are
    // slots. Long thin queries (a conveyor segment over small boxes) cover
    // hundreds of cells and stay on the grid as long as the population is
    // larger than that.
    kGridMinimumScanCells = 64u,
};

// Per-query bookkeeping the evaluation pass leaves for the scan and output passes.
struct alignas(8) RowPlan
{
    unsigned long long semanticCount; // matching hits before any capacity (ANY reports at most one)
    unsigned long long desired;       // rows the query would store under its own hit limit
    unsigned long long granted;       // rows written after the shared slot capacity, set by the scan
    unsigned int       valid;
    unsigned int       reserved;
};

// Bytes reserved per query for the closest or first hit kept between the passes.
enum : unsigned int
{
    kStashRowBytes = 128u,
};

// The plan's settings, the population's tables and the result slot's columns
// one launch reads. The kernel copies them into one local record that the row
// evaluation reads through a reference.
struct alignas(8) FamilyArguments
{
    unsigned int              family, precision, overlapKind, hitMode, resultFields;
    double                    planHalfExtents[3];
    unsigned long long        objectCount;
    const unsigned long long* objectIds;
    const unsigned int*       geometryTypes;
    const unsigned long long* geometryValues;
    const unsigned long long* objectMaterials; // null unless the plan asks for materials
    const unsigned long long* faceMaterials;   // null unless an exact plan's meshes assign face materials
    const unsigned long long* objectBits;
    const double*             objectMinimums;
    const double*             objectMaximums;
    const ExactObjectHeader*  exactObjects;
    const SphereGeometry*     spheres;
    const BoxGeometry*        boxes;
    const CapsuleGeometry*    capsules;
    const ConvexGeometry*     convexes;
    const double*             convexVertices;
    const unsigned int*       convexTriangleIndices;
    const double*             convexPlanes;
    const unsigned int*       convexEdges;
    const MeshGeometry*       meshes;
    const MeshDefinition*     meshDefinitions;
    unsigned long long*       outputObjectIds;
    int*                      outputGeometryTypes;
    unsigned long long*       outputGeometryValues;
    unsigned long long*       outputMaterialIds;
    double*                   outputPositions;
    double*                   outputQueryPositions;
    float*                    outputNormals;
    double*                   outputDistances;
    unsigned int*             outputFeatureKinds;
    unsigned long long*       outputFaceIndices;
    unsigned int*             outputFaceCorners;
    float*                    outputBarycentrics;
    unsigned char*            stash; // kStashRowBytes per query
    const DeviceBoxGrid*      grid;  // a device-prepared version's grid, or null
};

// The one by-value parameter of a plan entry: the shared arguments, the pass
// and the query columns of one launch.
struct alignas(8) PlanLaunchRecord
{
    FamilyArguments           arguments;
    unsigned int              pass;
    unsigned int              rotationBits;
    unsigned int              queryFlags; // ovspatialquery_query_flag_t bits of the launch (ADR-0065)
    double                    contactTolerance;
    double                    planRadius;
    unsigned long long        maximumHitsPerQuery;
    unsigned long long        queryCount;
    unsigned long long        resultCapacity; // the slot's hit capacity, granted to rows in query order
    const unsigned long long* queryBits;
    const double*             queryPositions;
    const float*              queryDirectionsOrRotations;
    const float*              queryRotations;
    const double*             minimumDistances;
    const double*             maximumDistances;
    const double*             radii;
    const unsigned long long* rowMaximumHits;
    const double*             queryHalfExtents;
    unsigned int*             queryStatus;
    unsigned long long*       hitOffsets;
    RowPlan*                  rowPlans;
    const unsigned int*       preparationFailed;
};

// The publish pass of a plan that keeps its results in mapped host memory
// (ADR-0077). Each grid row copies one column from device storage into its
// mapped column as one contiguous run of 16-byte words, so the writes that
// cross the bus are whole lines; a payload column's run is H rows, which the
// pass reads on the device. Column bases are kMappedColumnAlignment-aligned.
enum : unsigned int
{
    kMappedRowsQuery       = 0u, // Q rows
    kMappedRowsHit         = 1u, // H rows
    kMappedRowsOne         = 2u, // one row, H itself
    kMappedColumnsMaximum  = 17u,
    kMappedThreads         = 256u,
    kMappedColumnAlignment = 256u,
};

struct alignas(8) MappedColumnCopy
{
    unsigned long long source;      // device address
    unsigned long long destination; // mapped address
    unsigned int       rowBytes;
    unsigned int       rows; // kMappedRows*
};

struct alignas(8) MappedPublishRecord
{
    MappedColumnCopy          columns[kMappedColumnsMaximum];
    const unsigned long long* storedHitCount;
    unsigned long long        queryCount;
};

static_assert(sizeof(unsigned int) == 4u, "CUDA family u32 width changed");
static_assert(sizeof(unsigned long long) == 8u, "CUDA family u64 width changed");
static_assert(sizeof(ExactObjectHeader) == 16u, "CUDA object header layout changed");
static_assert(alignof(ExactObjectHeader) == 8u, "CUDA object header alignment changed");
static_assert(sizeof(SphereGeometry) == 32u, "CUDA sphere layout changed");
static_assert(sizeof(BoxGeometry) == 120u, "CUDA box layout changed");
static_assert(sizeof(CapsuleGeometry) == 56u, "CUDA capsule layout changed");
static_assert(sizeof(ConvexGeometry) == 88u, "CUDA convex header layout changed");
static_assert(sizeof(MeshGeometry) == 192u, "CUDA mesh instance layout changed");
static_assert(sizeof(MeshDefinition) == 72u, "CUDA mesh definition layout changed");
static_assert(sizeof(MeshBvhNode) == 40u, "CUDA mesh BVH node layout changed");
static_assert(sizeof(RowPlan) == 32u, "CUDA row plan layout changed");
static_assert(sizeof(void*) == 8u, "CUDA launch records assume 64-bit pointers");
static_assert(sizeof(OvstageSlotRecord) == 208u, "CUDA Ovstage slot record layout changed");
static_assert(sizeof(ExactConvexSource) == 16u, "CUDA exact convex source layout changed");
static_assert(sizeof(ExactCatalogueRecord) == 200u, "CUDA exact catalogue record layout changed");
static_assert(sizeof(FamilyArguments) == 320u, "CUDA family arguments layout changed");
static_assert(sizeof(PlanLaunchRecord) == 480u, "CUDA plan launch record layout changed");
static_assert(sizeof(MappedColumnCopy) == 24u, "CUDA mapped column copy layout changed");
static_assert(sizeof(MappedPublishRecord) == 424u, "CUDA mapped publish record layout changed");

} // namespace cuda_family
} // namespace detail
} // namespace ovspatialquery

#endif
