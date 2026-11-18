/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>
#include <ovspatialquery/ovspatialquery_pairs.h>
#if defined(OVSPATIALQUERY_ABI_WITH_OVSTAGE)
#include <ovspatialquery/ovspatialquery_ovstage.h>
#endif

#include <stddef.h>
#include <stdio.h>

#define TYPE(NAME, C_TYPE)          printf("T\t%s\t%zu\t%zu\n", NAME, sizeof(C_TYPE), _Alignof(C_TYPE))
#define FIELD(NAME, C_TYPE, MEMBER) printf("F\t%s\t%s\t%zu\n", NAME, #MEMBER, offsetof(C_TYPE, MEMBER))
#define FIELDS_1(NAME, C_TYPE, A)   FIELD(NAME, C_TYPE, A)
#define FIELDS_2(NAME, C_TYPE, A, B)                                                                                   \
    FIELDS_1(NAME, C_TYPE, A);                                                                                         \
    FIELD(NAME, C_TYPE, B)
#define FIELDS_3(NAME, C_TYPE, A, B, C)                                                                                \
    FIELDS_2(NAME, C_TYPE, A, B);                                                                                      \
    FIELD(NAME, C_TYPE, C)
#define FIELDS_4(NAME, C_TYPE, A, B, C, D)                                                                             \
    FIELDS_3(NAME, C_TYPE, A, B, C);                                                                                   \
    FIELD(NAME, C_TYPE, D)
#define FIELDS_5(NAME, C_TYPE, A, B, C, D, E)                                                                          \
    FIELDS_4(NAME, C_TYPE, A, B, C, D);                                                                                \
    FIELD(NAME, C_TYPE, E)
#define FIELDS_6(NAME, C_TYPE, A, B, C, D, E, F)                                                                       \
    FIELDS_5(NAME, C_TYPE, A, B, C, D, E);                                                                             \
    FIELD(NAME, C_TYPE, F)
#define FIELDS_7(NAME, C_TYPE, A, B, C, D, E, F, G)                                                                    \
    FIELDS_6(NAME, C_TYPE, A, B, C, D, E, F);                                                                          \
    FIELD(NAME, C_TYPE, G)
#define FIELDS_8(NAME, C_TYPE, A, B, C, D, E, F, G, H)                                                                 \
    FIELDS_7(NAME, C_TYPE, A, B, C, D, E, F, G);                                                                       \
    FIELD(NAME, C_TYPE, H)
#define FIELDS_9(NAME, C_TYPE, A, B, C, D, E, F, G, H, I)                                                              \
    FIELDS_8(NAME, C_TYPE, A, B, C, D, E, F, G, H);                                                                    \
    FIELD(NAME, C_TYPE, I)
#define FIELDS_10(NAME, C_TYPE, A, B, C, D, E, F, G, H, I, J)                                                          \
    FIELDS_9(NAME, C_TYPE, A, B, C, D, E, F, G, H, I);                                                                 \
    FIELD(NAME, C_TYPE, J)
#define FIELDS_11(NAME, C_TYPE, A, B, C, D, E, F, G, H, I, J, K)                                                       \
    FIELDS_10(NAME, C_TYPE, A, B, C, D, E, F, G, H, I, J);                                                             \
    FIELD(NAME, C_TYPE, K)
#define FIELDS_12(NAME, C_TYPE, A, B, C, D, E, F, G, H, I, J, K, L)                                                    \
    FIELDS_11(NAME, C_TYPE, A, B, C, D, E, F, G, H, I, J, K);                                                          \
    FIELD(NAME, C_TYPE, L)
#define ENUM(TYPE, NAME, VALUE) printf("E\t%s\t%s\t%d\n", TYPE, NAME, (int)(VALUE))

int main(void)
{
#if defined(OVSPATIALQUERY_ABI_BATCH_COLUMNS_ONLY)
    TYPE("DLDevice", DLDevice);
    TYPE("DLDataType", DLDataType);
    TYPE("DLTensor", DLTensor);
    TYPE("BatchHitRanges", ovspatialquery_batch_hit_ranges_t);
    TYPE("BatchResultCommon", ovspatialquery_batch_result_common_t);
    TYPE("GeometryIdentityColumns", ovspatialquery_geometry_identity_columns_t);
    TYPE("ExactFeatureColumns", ovspatialquery_exact_feature_columns_t);
    TYPE("RaycastBatchColumns", ovspatialquery_raycast_batch_columns_t);
    TYPE("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t);
    TYPE("OverlapBatchColumns", ovspatialquery_overlap_batch_columns_t);
    TYPE("RaycastBatchReservation", ovspatialquery_raycast_batch_reservation_t);
    TYPE("SphereSweepBatchReservation", ovspatialquery_sphere_sweep_batch_reservation_t);
    TYPE("OverlapBatchReservation", ovspatialquery_overlap_batch_reservation_t);
    TYPE("RaycastBatchResult", ovspatialquery_raycast_batch_result_t);
    TYPE("SphereSweepBatchResult", ovspatialquery_sphere_sweep_batch_result_t);
    TYPE("OverlapBatchResult", ovspatialquery_overlap_batch_result_t);
    TYPE("PairReporterResult", ovspatialquery_pair_reporter_result_t);

    FIELDS_2("DLDevice", DLDevice, device_type, device_id);
    FIELDS_3("DLDataType", DLDataType, code, bits, lanes);
    FIELDS_7("DLTensor", DLTensor, data, device, ndim, dtype, shape, strides, byte_offset);
    FIELDS_5("BatchHitRanges", ovspatialquery_batch_hit_ranges_t, query_status, stored_hit_count, hit_offsets,
             hit_counts, truncated);
    FIELDS_7("BatchResultCommon", ovspatialquery_batch_result_common_t, result_slot, query_count, hit_capacity, truth,
             source_ordinal, execution_device, result_ready);
    FIELDS_2("GeometryIdentityColumns", ovspatialquery_geometry_identity_columns_t, types, values);
    FIELDS_4("ExactFeatureColumns", ovspatialquery_exact_feature_columns_t, kinds, source_face_indices,
             source_corner_indices, barycentrics);
    FIELDS_6("RaycastBatchColumns", ovspatialquery_raycast_batch_columns_t, hit_ranges, object_ids, geometry,
             material_ids, contact, reserved);
    FIELD("RaycastBatchColumns", ovspatialquery_raycast_batch_columns_t, contact.candidate.bounds_positions);
    FIELD("RaycastBatchColumns", ovspatialquery_raycast_batch_columns_t, contact.candidate.bounds_normals);
    FIELD("RaycastBatchColumns", ovspatialquery_raycast_batch_columns_t, contact.candidate.distances);
    FIELD("RaycastBatchColumns", ovspatialquery_raycast_batch_columns_t, contact.exact.positions);
    FIELD("RaycastBatchColumns", ovspatialquery_raycast_batch_columns_t, contact.exact.normals);
    FIELD("RaycastBatchColumns", ovspatialquery_raycast_batch_columns_t, contact.exact.distances);
    FIELD("RaycastBatchColumns", ovspatialquery_raycast_batch_columns_t, contact.exact.features);
    FIELDS_6("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t, hit_ranges, object_ids, geometry,
             material_ids, contact, reserved);
    FIELD("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t, contact.candidate.bounds_positions);
    FIELD("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t, contact.candidate.bounds_normals);
    FIELD("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t, contact.candidate.query_positions);
    FIELD("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t, contact.candidate.distances);
    FIELD("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t, contact.exact.positions);
    FIELD("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t, contact.exact.normals);
    FIELD("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t, contact.exact.query_positions);
    FIELD("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t, contact.exact.distances);
    FIELD("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t, contact.exact.features);
    FIELDS_5("OverlapBatchColumns", ovspatialquery_overlap_batch_columns_t, hit_ranges, object_ids, geometry,
             material_ids, reserved);
    FIELDS_2("RaycastBatchReservation", ovspatialquery_raycast_batch_reservation_t, common, columns);
    FIELDS_2("SphereSweepBatchReservation", ovspatialquery_sphere_sweep_batch_reservation_t, common, columns);
    FIELDS_2("OverlapBatchReservation", ovspatialquery_overlap_batch_reservation_t, common, columns);
    FIELDS_2("RaycastBatchResult", ovspatialquery_raycast_batch_result_t, common, columns);
    FIELDS_2("SphereSweepBatchResult", ovspatialquery_sphere_sweep_batch_result_t, common, columns);
    FIELDS_2("OverlapBatchResult", ovspatialquery_overlap_batch_result_t, common, columns);
    FIELDS_8("PairReporterResult", ovspatialquery_pair_reporter_result_t, result_slot, outcome, first_source_ordinal,
             second_source_ordinal, accepted_active_pair_count, result_ready, entered_pairs, exited_pairs);
    return 0;
#else
    TYPE("DLDevice", DLDevice);
    TYPE("DLDataType", DLDataType);
    TYPE("DLTensor", DLTensor);
    TYPE("Result", ovspatialquery_result_t);
    TYPE("String", ovspatialquery_string_t);
    TYPE("EnqueueResult", ovspatialquery_enqueue_result_t);
    TYPE("CudaEvent", ovspatialquery_cuda_event_t);
    TYPE("CudaSync", ovspatialquery_cuda_sync_t);
    TYPE("ObjectId", ovspatialquery_object_id_t);
    TYPE("ParticipationBits", ovspatialquery_participation_bits_t);
    TYPE("GeometryRef", ovspatialquery_geometry_ref_t);
    TYPE("DeviceBudget", ovspatialquery_device_budget_t);
    TYPE("InstanceDesc", ovspatialquery_instance_desc_t);
    TYPE("PopulationDesc", ovspatialquery_population_desc_t);
    TYPE("PopulationInfo", ovspatialquery_population_info_t);
    TYPE("PopulationUpdateResult", ovspatialquery_population_update_result_t);
    TYPE("InputArray", ovspatialquery_input_array_t);
    TYPE("DynamicExecution", ovspatialquery_dynamic_execution_t);
    TYPE("ExplicitInstanceTable", ovspatialquery_explicit_instance_table_t);
    TYPE("ExplicitPopulationSet", ovspatialquery_explicit_population_set_t);
    TYPE("ExplicitPopulationUpdate", ovspatialquery_explicit_population_update_t);
    TYPE("ExplicitViewDesc", ovspatialquery_explicit_view_desc_t);
    TYPE("SphereDefinitionTable", ovspatialquery_sphere_definition_table_t);
    TYPE("BoxDefinitionTable", ovspatialquery_box_definition_table_t);
    TYPE("CapsuleDefinitionTable", ovspatialquery_capsule_definition_table_t);
    TYPE("ConvexDefinitionTable", ovspatialquery_convex_definition_table_t);
    TYPE("TriangleMeshDefinitionTable", ovspatialquery_triangle_mesh_definition_table_t);
    TYPE("TargetRef", ovspatialquery_target_ref_t);
    TYPE("OrdinalSelection", ovspatialquery_ordinal_selection_t);
    TYPE("QueryTarget", ovspatialquery_query_target_t);
    TYPE("QueryCommon", ovspatialquery_query_common_t);
    TYPE("MultipleResult", ovspatialquery_multiple_result_t);
    TYPE("HitIdentity", ovspatialquery_hit_identity_t);
    TYPE("TriangleFeature", ovspatialquery_triangle_feature_t);
    TYPE("GeometryFeature", ovspatialquery_geometry_feature_t);
    TYPE("Ray", ovspatialquery_ray_t);
    TYPE("RaycastDesc", ovspatialquery_raycast_desc_t);
    TYPE("RaycastBoundsCandidate", ovspatialquery_raycast_bounds_candidate_t);
    TYPE("RaycastExactHit", ovspatialquery_raycast_exact_hit_t);
    TYPE("RaycastHit", ovspatialquery_raycast_hit_t);
    TYPE("RaycastHitBuffer", ovspatialquery_raycast_hit_buffer_t);
    TYPE("RaycastSingleResult", ovspatialquery_raycast_single_result_t);
    TYPE("SphereSweep", ovspatialquery_sphere_sweep_t);
    TYPE("SphereSweepDesc", ovspatialquery_sphere_sweep_desc_t);
    TYPE("SphereSweepBoundsCandidate", ovspatialquery_sphere_sweep_bounds_candidate_t);
    TYPE("SphereSweepExactHit", ovspatialquery_sphere_sweep_exact_hit_t);
    TYPE("SphereSweepHit", ovspatialquery_sphere_sweep_hit_t);
    TYPE("SphereSweepClosestResult", ovspatialquery_sphere_sweep_closest_result_t);
    TYPE("SphereSweepHitBuffer", ovspatialquery_sphere_sweep_hit_buffer_t);
    TYPE("BoxSweep", ovspatialquery_box_sweep_t);
    TYPE("BoxSweepDesc", ovspatialquery_box_sweep_desc_t);
    TYPE("OverlapSphere", ovspatialquery_overlap_sphere_t);
    TYPE("OverlapBox", ovspatialquery_overlap_box_t);
    TYPE("OverlapAabb", ovspatialquery_overlap_aabb_t);
    TYPE("OverlapShape", ovspatialquery_overlap_shape_t);
    TYPE("OverlapDesc", ovspatialquery_overlap_desc_t);
    TYPE("OverlapHit", ovspatialquery_overlap_hit_t);
    TYPE("OverlapHitBuffer", ovspatialquery_overlap_hit_buffer_t);
    TYPE("OverlapAnyResult", ovspatialquery_overlap_any_result_t);
    TYPE("BatchHitRanges", ovspatialquery_batch_hit_ranges_t);
    TYPE("OverlapBatchShape", ovspatialquery_overlap_batch_shape_t);
    TYPE("RaycastPlanDesc", ovspatialquery_raycast_plan_desc_t);
    TYPE("SphereSweepPlanDesc", ovspatialquery_sphere_sweep_plan_desc_t);
    TYPE("BoxSweepPlanDesc", ovspatialquery_box_sweep_plan_desc_t);
    TYPE("OverlapPlanDesc", ovspatialquery_overlap_plan_desc_t);
    TYPE("RaycastBatch", ovspatialquery_raycast_batch_t);
    TYPE("SphereSweepBatch", ovspatialquery_sphere_sweep_batch_t);
    TYPE("BoxSweepBatch", ovspatialquery_box_sweep_batch_t);
    TYPE("OverlapBatch", ovspatialquery_overlap_batch_t);
    TYPE("BatchResultCommon", ovspatialquery_batch_result_common_t);
    TYPE("GeometryIdentityColumns", ovspatialquery_geometry_identity_columns_t);
    TYPE("ExactFeatureColumns", ovspatialquery_exact_feature_columns_t);
    TYPE("RaycastBatchColumns", ovspatialquery_raycast_batch_columns_t);
    TYPE("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t);
    TYPE("OverlapBatchColumns", ovspatialquery_overlap_batch_columns_t);
    TYPE("RaycastBatchReservation", ovspatialquery_raycast_batch_reservation_t);
    TYPE("SphereSweepBatchReservation", ovspatialquery_sphere_sweep_batch_reservation_t);
    TYPE("OverlapBatchReservation", ovspatialquery_overlap_batch_reservation_t);
    TYPE("RaycastBatchResult", ovspatialquery_raycast_batch_result_t);
    TYPE("SphereSweepBatchResult", ovspatialquery_sphere_sweep_batch_result_t);
    TYPE("OverlapBatchResult", ovspatialquery_overlap_batch_result_t);
    TYPE("Pair", ovspatialquery_pair_t);
    TYPE("PairBuffer", ovspatialquery_pair_buffer_t);
    TYPE("PairSnapshotDesc", ovspatialquery_pair_snapshot_desc_t);
    TYPE("PairSnapshotResult", ovspatialquery_pair_snapshot_result_t);
    TYPE("PairReporterDesc", ovspatialquery_pair_reporter_desc_t);
    TYPE("PairReporterAdvance", ovspatialquery_pair_reporter_advance_t);
    TYPE("PairReporterReservation", ovspatialquery_pair_reporter_reservation_t);
    TYPE("PairReporterResult", ovspatialquery_pair_reporter_result_t);
    TYPE("OpWaitResult", ovspatialquery_op_wait_result_t);
    TYPE("OpStatus", ovspatialquery_op_status_t);

    FIELDS_2("DLDevice", DLDevice, device_type, device_id);
    FIELDS_3("DLDataType", DLDataType, code, bits, lanes);
    FIELDS_7("DLTensor", DLTensor, data, device, ndim, dtype, shape, strides, byte_offset);
    FIELDS_1("Result", ovspatialquery_result_t, status);
    FIELDS_2("String", ovspatialquery_string_t, ptr, length);
    FIELDS_2("EnqueueResult", ovspatialquery_enqueue_result_t, status, op_id);
    FIELDS_1("CudaEvent", ovspatialquery_cuda_event_t, value);
    FIELDS_3("CudaSync", ovspatialquery_cuda_sync_t, has_stream, stream, wait_event);
    FIELDS_1("ObjectId", ovspatialquery_object_id_t, value);
    FIELDS_1("ParticipationBits", ovspatialquery_participation_bits_t, words);
    FIELDS_2("GeometryRef", ovspatialquery_geometry_ref_t, type, value);
    FIELDS_2("DeviceBudget", ovspatialquery_device_budget_t, device, byte_budget);
    FIELDS_5("InstanceDesc", ovspatialquery_instance_desc_t, operation_capacity, max_cpu_concurrency, host_byte_budget,
             device_budgets, device_budget_count);
    FIELDS_2("PopulationDesc", ovspatialquery_population_desc_t, retained_ordinal_capacity,
             pending_publication_capacity);
    FIELDS_5("PopulationInfo", ovspatialquery_population_info_t, ready, head_ordinal, object_count,
             retained_ordinal_count, update_pending);
    FIELDS_5("PopulationUpdateResult", ovspatialquery_population_update_result_t, published_ordinal, object_count,
             excluded_object_count, excluded_objects, exclusion_reasons);
    FIELDS_2("InputArray", ovspatialquery_input_array_t, tensor, ownership);
    FIELDS_3("DynamicExecution", ovspatialquery_dynamic_execution_t, device, cuda_sync, preparation);
    FIELDS_8("ExplicitInstanceTable", ovspatialquery_explicit_instance_table_t, object_values, geometry_types,
             geometry_values, translations, rotations_xyzw, uniform_scales, participation_bits, material_ids);
    FIELDS_4("ExplicitPopulationSet", ovspatialquery_explicit_population_set_t, target_ordinal, instances,
             dynamic_execution, slot_capacity);
    FIELDS_6("ExplicitPopulationUpdate", ovspatialquery_explicit_population_update_t, base_ordinal, target_ordinal,
             kind, changed_instances, removed_object_values, dynamic_execution);
    FIELDS_1("ExplicitViewDesc", ovspatialquery_explicit_view_desc_t, object_ids);
    FIELDS_2("SphereDefinitionTable", ovspatialquery_sphere_definition_table_t, definition_values, radii);
    FIELDS_2("BoxDefinitionTable", ovspatialquery_box_definition_table_t, definition_values, half_extents);
    FIELDS_3("CapsuleDefinitionTable", ovspatialquery_capsule_definition_table_t, definition_values, radii,
             half_segments);
    FIELDS_11("ConvexDefinitionTable", ovspatialquery_convex_definition_table_t, definition_values, vertex_offsets,
              vertex_counts, face_offsets, face_counts, vertices, face_vertex_offsets, face_vertex_counts,
              face_vertex_indices, enclosure_tolerance, hull_vertex_limit);
    FIELDS_11("TriangleMeshDefinitionTable", ovspatialquery_triangle_mesh_definition_table_t, definition_values,
              vertex_offsets, vertex_counts, triangle_offsets, triangle_counts, vertices, triangle_indices,
              double_sided, source_face_material_ids, source_face_indices, source_face_corner_indices);
    FIELDS_2("TargetRef", ovspatialquery_target_ref_t, kind, handle);
    FIELDS_2("OrdinalSelection", ovspatialquery_ordinal_selection_t, kind, source_ordinal);
    FIELDS_2("QueryTarget", ovspatialquery_query_target_t, target, ordinal);
    FIELDS_6("QueryCommon", ovspatialquery_query_common_t, target, precision, query_bits, contact_tolerance,
             query_flags, reserved);
    FIELDS_3("MultipleResult", ovspatialquery_multiple_result_t, source_ordinal, stored_hit_count, truncated);
    FIELDS_3("HitIdentity", ovspatialquery_hit_identity_t, object_id, geometry, material_id);
    FIELDS_3("TriangleFeature", ovspatialquery_triangle_feature_t, source_face_index, source_face_corner_indices,
             barycentrics);
    FIELDS_2("GeometryFeature", ovspatialquery_geometry_feature_t, kind, data);
    FIELDS_4("Ray", ovspatialquery_ray_t, origin, unit_direction, min_distance, max_distance);
    FIELDS_3("RaycastDesc", ovspatialquery_raycast_desc_t, common, ray, mesh_hits);
    FIELDS_4("RaycastBoundsCandidate", ovspatialquery_raycast_bounds_candidate_t, identity, bounds_position,
             bounds_normal, distance);
    FIELDS_5("RaycastExactHit", ovspatialquery_raycast_exact_hit_t, identity, position, normal, distance, feature);
    FIELDS_2("RaycastHit", ovspatialquery_raycast_hit_t, truth, data);
    FIELDS_2("RaycastHitBuffer", ovspatialquery_raycast_hit_buffer_t, hits, capacity);
    FIELDS_3("RaycastSingleResult", ovspatialquery_raycast_single_result_t, source_ordinal, has_hit, hit);
    FIELDS_5("SphereSweep", ovspatialquery_sphere_sweep_t, center, radius, unit_direction, min_distance, max_distance);
    FIELDS_2("SphereSweepDesc", ovspatialquery_sphere_sweep_desc_t, common, sweep);
    FIELDS_5("SphereSweepBoundsCandidate", ovspatialquery_sphere_sweep_bounds_candidate_t, identity, query_position,
             bounds_position, bounds_normal, distance);
    FIELDS_6("SphereSweepExactHit", ovspatialquery_sphere_sweep_exact_hit_t, identity, query_position, object_position,
             normal, distance, feature);
    FIELDS_2("SphereSweepHit", ovspatialquery_sphere_sweep_hit_t, truth, data);
    FIELDS_3("SphereSweepClosestResult", ovspatialquery_sphere_sweep_closest_result_t, source_ordinal, has_hit, hit);
    FIELDS_2("SphereSweepHitBuffer", ovspatialquery_sphere_sweep_hit_buffer_t, hits, capacity);
    FIELDS_6("BoxSweep", ovspatialquery_box_sweep_t, center, half_extents, rotation_xyzw, unit_direction, min_distance,
             max_distance);
    FIELDS_2("BoxSweepDesc", ovspatialquery_box_sweep_desc_t, common, sweep);
    FIELDS_2("OverlapSphere", ovspatialquery_overlap_sphere_t, center, radius);
    FIELDS_3("OverlapBox", ovspatialquery_overlap_box_t, center, rotation_xyzw, half_extents);
    FIELDS_2("OverlapAabb", ovspatialquery_overlap_aabb_t, min_corner, max_corner);
    FIELDS_2("OverlapShape", ovspatialquery_overlap_shape_t, kind, data);
    FIELDS_2("OverlapDesc", ovspatialquery_overlap_desc_t, common, shape);
    FIELDS_2("OverlapHit", ovspatialquery_overlap_hit_t, truth, identity);
    FIELDS_2("OverlapHitBuffer", ovspatialquery_overlap_hit_buffer_t, hits, capacity);
    FIELDS_3("OverlapAnyResult", ovspatialquery_overlap_any_result_t, source_ordinal, has_hit, hit);
    FIELDS_5("BatchHitRanges", ovspatialquery_batch_hit_ranges_t, query_status, stored_hit_count, hit_offsets,
             hit_counts, truncated);
    FIELDS_2("OverlapBatchShape", ovspatialquery_overlap_batch_shape_t, kind, data);
    FIELDS_11("RaycastPlanDesc", ovspatialquery_raycast_plan_desc_t, target, hit_mode, precision, device,
              maximum_query_count, maximum_hits_per_query, hit_capacity_per_slot, result_slot_count, result_fields,
              contact_tolerance, mesh_hits);
    FIELDS_11("SphereSweepPlanDesc", ovspatialquery_sphere_sweep_plan_desc_t, target, precision, radius, device,
              maximum_query_count, result_slot_count, result_fields, hit_capacity_per_slot, contact_tolerance, hit_mode,
              maximum_hits_per_query);
    FIELDS_11("BoxSweepPlanDesc", ovspatialquery_box_sweep_plan_desc_t, target, precision, half_extents, device,
              maximum_query_count, result_slot_count, result_fields, hit_capacity_per_slot, contact_tolerance, hit_mode,
              maximum_hits_per_query);
    FIELDS_11("OverlapPlanDesc", ovspatialquery_overlap_plan_desc_t, target, hit_mode, precision, shape, device,
              maximum_query_count, maximum_hits_per_query, hit_capacity_per_slot, result_slot_count, result_fields,
              contact_tolerance);
    FIELDS_11("RaycastBatch", ovspatialquery_raycast_batch_t, ordinal, query_count, query_bits, origins,
              unit_directions, min_distances, max_distances, maximum_hits, cuda_sync, query_flags, reserved);
    FIELDS_11("SphereSweepBatch", ovspatialquery_sphere_sweep_batch_t, ordinal, query_count, query_bits, centers,
              unit_directions, min_distances, max_distances, radii, cuda_sync, query_flags, reserved);
    FIELDS_12("BoxSweepBatch", ovspatialquery_box_sweep_batch_t, ordinal, query_count, query_bits, centers,
              rotations_xyzw, unit_directions, min_distances, max_distances, cuda_sync, half_extents, query_flags,
              reserved);
    FIELDS_10("OverlapBatch", ovspatialquery_overlap_batch_t, ordinal, query_count, query_bits, translations,
              rotations_xyzw, half_extents, maximum_hits, cuda_sync, radii, reserved);
    FIELDS_7("BatchResultCommon", ovspatialquery_batch_result_common_t, result_slot, query_count, hit_capacity, truth,
             source_ordinal, execution_device, result_ready);
    FIELDS_2("GeometryIdentityColumns", ovspatialquery_geometry_identity_columns_t, types, values);
    FIELDS_4("ExactFeatureColumns", ovspatialquery_exact_feature_columns_t, kinds, source_face_indices,
             source_corner_indices, barycentrics);
    FIELDS_6("RaycastBatchColumns", ovspatialquery_raycast_batch_columns_t, hit_ranges, object_ids, geometry,
             material_ids, contact, reserved);
    FIELDS_6("SphereSweepBatchColumns", ovspatialquery_sphere_sweep_batch_columns_t, hit_ranges, object_ids, geometry,
             material_ids, contact, reserved);
    FIELDS_5("OverlapBatchColumns", ovspatialquery_overlap_batch_columns_t, hit_ranges, object_ids, geometry,
             material_ids, reserved);
    FIELDS_2("RaycastBatchReservation", ovspatialquery_raycast_batch_reservation_t, common, columns);
    FIELDS_2("SphereSweepBatchReservation", ovspatialquery_sphere_sweep_batch_reservation_t, common, columns);
    FIELDS_2("OverlapBatchReservation", ovspatialquery_overlap_batch_reservation_t, common, columns);
    FIELDS_2("RaycastBatchResult", ovspatialquery_raycast_batch_result_t, common, columns);
    FIELDS_2("SphereSweepBatchResult", ovspatialquery_sphere_sweep_batch_result_t, common, columns);
    FIELDS_2("OverlapBatchResult", ovspatialquery_overlap_batch_result_t, common, columns);
    FIELDS_2("Pair", ovspatialquery_pair_t, first, second);
    FIELDS_2("PairBuffer", ovspatialquery_pair_buffer_t, pairs, capacity);
    FIELDS_5("PairSnapshotDesc", ovspatialquery_pair_snapshot_desc_t, first, first_query_bits, second,
             second_query_bits, contact_tolerance);
    FIELDS_4("PairSnapshotResult", ovspatialquery_pair_snapshot_result_t, first_source_ordinal, second_source_ordinal,
             stored_pair_count, truncated);
    FIELDS_9("PairReporterDesc", ovspatialquery_pair_reporter_desc_t, first, first_query_bits, second,
             second_query_bits, active_pair_capacity, delta_pair_capacity, result_slot_count, contact_tolerance,
             result_device);
    FIELDS_3("PairReporterAdvance", ovspatialquery_pair_reporter_advance_t, first_ordinal, second_ordinal, cuda_sync);
    FIELDS_7("PairReporterReservation", ovspatialquery_pair_reporter_reservation_t, result_slot, result_device,
             delta_pair_capacity, result_ready, entered_pairs, exited_pairs, counts);
    FIELDS_8("PairReporterResult", ovspatialquery_pair_reporter_result_t, result_slot, outcome, first_source_ordinal,
             second_source_ordinal, accepted_active_pair_count, result_ready, entered_pairs, exited_pairs);
    FIELDS_1("OpWaitResult", ovspatialquery_op_wait_result_t, state);
    FIELDS_3("OpStatus", ovspatialquery_op_status_t, state, terminal_status, progress);

    ENUM("ApiStatus", "SUCCESS", OVSPATIALQUERY_API_SUCCESS);
    ENUM("ApiStatus", "ERROR", OVSPATIALQUERY_API_ERROR);
    ENUM("ApiStatus", "TIMEOUT", OVSPATIALQUERY_API_TIMEOUT);
    ENUM("ApiStatus", "INVALID_ARGUMENT", OVSPATIALQUERY_API_INVALID_ARGUMENT);
    ENUM("ApiStatus", "INVALID_HANDLE", OVSPATIALQUERY_API_INVALID_HANDLE);
    ENUM("ApiStatus", "INVALID_STATE", OVSPATIALQUERY_API_INVALID_STATE);
    ENUM("ApiStatus", "NOT_SUPPORTED", OVSPATIALQUERY_API_NOT_SUPPORTED);
    ENUM("ApiStatus", "NOT_READY", OVSPATIALQUERY_API_NOT_READY);
    ENUM("ApiStatus", "BUSY", OVSPATIALQUERY_API_BUSY);
    ENUM("ApiStatus", "STALE_ORDINAL", OVSPATIALQUERY_API_STALE_ORDINAL);
    ENUM("ApiStatus", "ORDINAL_MISMATCH", OVSPATIALQUERY_API_ORDINAL_MISMATCH);
    ENUM("ApiStatus", "ORDINAL_NOT_RETAINED", OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED);
    ENUM("ApiStatus", "BACKWARD_ORDINAL", OVSPATIALQUERY_API_BACKWARD_ORDINAL);
    ENUM("ApiStatus", "OPERATION_CAPACITY", OVSPATIALQUERY_API_OPERATION_CAPACITY);
    ENUM("ApiStatus", "RESULT_CAPACITY", OVSPATIALQUERY_API_RESULT_CAPACITY);
    ENUM("ApiStatus", "HOST_BUDGET", OVSPATIALQUERY_API_HOST_BUDGET);
    ENUM("ApiStatus", "DEVICE_BUDGET", OVSPATIALQUERY_API_DEVICE_BUDGET);
    ENUM("ApiStatus", "OPERATION_FAILED", OVSPATIALQUERY_API_OPERATION_FAILED);
    ENUM("ApiStatus", "OUT_OF_MEMORY", OVSPATIALQUERY_API_OUT_OF_MEMORY);
    ENUM("ApiStatus", "INTERNAL", OVSPATIALQUERY_API_INTERNAL);
    ENUM("ApiStatus", "DEVICE_LOST", OVSPATIALQUERY_API_DEVICE_LOST);
    ENUM("GeometryType", "INVALID", OVSPATIALQUERY_GEOMETRY_INVALID);
    ENUM("GeometryType", "SPHERE", OVSPATIALQUERY_GEOMETRY_SPHERE);
    ENUM("GeometryType", "BOX", OVSPATIALQUERY_GEOMETRY_BOX);
    ENUM("GeometryType", "CAPSULE", OVSPATIALQUERY_GEOMETRY_CAPSULE);
    ENUM("GeometryType", "CONVEX", OVSPATIALQUERY_GEOMETRY_CONVEX);
    ENUM("GeometryType", "TRIANGLE_MESH", OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH);
    ENUM("InputOwnership", "BORROWED_UNTIL_TERMINAL", OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL);
    ENUM("InputOwnership", "COPIED_ON_SUBMISSION", OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION);
    ENUM("TargetKind", "POPULATION", OVSPATIALQUERY_TARGET_POPULATION);
    ENUM("TargetKind", "VIEW", OVSPATIALQUERY_TARGET_VIEW);
    ENUM("OrdinalSelectionKind", "HEAD", OVSPATIALQUERY_ORDINAL_HEAD);
    ENUM("OrdinalSelectionKind", "RETAINED_EXACT", OVSPATIALQUERY_ORDINAL_RETAINED_EXACT);
    ENUM("OrdinalSelectionKind", "ADMITTED_EXACT", OVSPATIALQUERY_ORDINAL_ADMITTED_EXACT);
    ENUM("QueryPrecision", "BOUNDS_CANDIDATE", OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    ENUM("QueryPrecision", "EXACT", OVSPATIALQUERY_QUERY_EXACT);
    ENUM("QueryPrecision", "EXACT_SINGLE", OVSPATIALQUERY_QUERY_EXACT_SINGLE);
    ENUM("HitTruth", "BOUNDS_CANDIDATE", OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);
    ENUM("HitTruth", "EXACT_SURFACE", OVSPATIALQUERY_HIT_EXACT_SURFACE);
    ENUM("GeometryFeatureKind", "NONE", OVSPATIALQUERY_GEOMETRY_FEATURE_NONE);
    ENUM("GeometryFeatureKind", "TRIANGLE", OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
    ENUM("OverlapShapeKind", "SPHERE", OVSPATIALQUERY_OVERLAP_SPHERE);
    ENUM("OverlapShapeKind", "BOX", OVSPATIALQUERY_OVERLAP_BOX);
    ENUM("OverlapShapeKind", "AABB", OVSPATIALQUERY_OVERLAP_AABB);
    ENUM("HitMode", "CLOSEST", OVSPATIALQUERY_HIT_MODE_CLOSEST);
    ENUM("HitMode", "ANY", OVSPATIALQUERY_HIT_MODE_ANY);
    ENUM("HitMode", "MULTIPLE", OVSPATIALQUERY_HIT_MODE_MULTIPLE);
    ENUM("QueryFlag", "MESH_BOTH_SIDES", OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES);
    ENUM("MeshHits", "ALL", OVSPATIALQUERY_MESH_HITS_ALL);
    ENUM("MeshHits", "NEAREST", OVSPATIALQUERY_MESH_HITS_NEAREST);
    ENUM("BatchRowStatus", "SUCCESS", OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    ENUM("BatchRowStatus", "INVALID_ARGUMENT", OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT);
    ENUM("BatchRowStatus", "NOT_SUPPORTED", OVSPATIALQUERY_BATCH_ROW_NOT_SUPPORTED);
    ENUM("BatchRowStatus", "FAILED", OVSPATIALQUERY_BATCH_ROW_FAILED);
    ENUM("BatchResultField", "POSITION", OVSPATIALQUERY_BATCH_RESULT_POSITION);
    ENUM("BatchResultField", "NORMAL", OVSPATIALQUERY_BATCH_RESULT_NORMAL);
    ENUM("BatchResultField", "DISTANCE", OVSPATIALQUERY_BATCH_RESULT_DISTANCE);
    ENUM("BatchResultField", "SOURCE_FEATURE", OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE);
    ENUM("BatchResultField", "GEOMETRY_IDENTITY", OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY);
    ENUM("BatchResultField", "QUERY_POSITION", OVSPATIALQUERY_BATCH_RESULT_QUERY_POSITION);
    ENUM("BatchResultField", "MATERIAL", OVSPATIALQUERY_BATCH_RESULT_MATERIAL);
    ENUM("BatchResultField", "HOST_MAPPED", OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED);
    ENUM("PairReporterOutcome", "ADVANCED", OVSPATIALQUERY_PAIR_REPORTER_ADVANCED);
    ENUM("PairReporterOutcome", "ACTIVE_CAPACITY", OVSPATIALQUERY_PAIR_REPORTER_ACTIVE_CAPACITY);
    ENUM("PairReporterOutcome", "DELTA_CAPACITY", OVSPATIALQUERY_PAIR_REPORTER_DELTA_CAPACITY);
    ENUM("PairReporterOutcome", "FAILED", OVSPATIALQUERY_PAIR_REPORTER_FAILED);
    ENUM("OpState", "PENDING", OVSPATIALQUERY_OP_PENDING);
    ENUM("OpState", "COMPLETED", OVSPATIALQUERY_OP_COMPLETED);
    ENUM("OpState", "FAILURE", OVSPATIALQUERY_OP_FAILURE);
    ENUM("Preparation", "ON_HOST", OVSPATIALQUERY_PREPARE_ON_HOST);
    ENUM("Preparation", "ON_DEVICE", OVSPATIALQUERY_PREPARE_ON_DEVICE);
    ENUM("Preparation", "ON_DEVICE_HOST_READABLE", OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE);
    ENUM("ExplicitUpdateKind", "COMPLETE_ROWS", OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS);
    ENUM("ExplicitUpdateKind", "POSES", OVSPATIALQUERY_EXPLICIT_UPDATE_POSES);
    ENUM("ExplicitUpdateKind", "POSES_AND_BITS", OVSPATIALQUERY_EXPLICIT_UPDATE_POSES_AND_BITS);
    ENUM("Capability", "RAYCAST_PLAN", OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN);
    ENUM("Capability", "RAYCAST_PLAN_EXACT", OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN_EXACT);
    ENUM("Capability", "SPHERE_SWEEP_PLAN", OVSPATIALQUERY_CAPABILITY_SPHERE_SWEEP_PLAN);
    ENUM("Capability", "SPHERE_SWEEP_PLAN_EXACT", OVSPATIALQUERY_CAPABILITY_SPHERE_SWEEP_PLAN_EXACT);
    ENUM("Capability", "BOX_SWEEP_PLAN", OVSPATIALQUERY_CAPABILITY_BOX_SWEEP_PLAN);
    ENUM("Capability", "BOX_SWEEP_PLAN_EXACT", OVSPATIALQUERY_CAPABILITY_BOX_SWEEP_PLAN_EXACT);
    ENUM("Capability", "OVERLAP_PLAN", OVSPATIALQUERY_CAPABILITY_OVERLAP_PLAN);
    ENUM("Capability", "OVERLAP_PLAN_EXACT", OVSPATIALQUERY_CAPABILITY_OVERLAP_PLAN_EXACT);
    ENUM("Capability", "PREPARE_ON_DEVICE", OVSPATIALQUERY_CAPABILITY_PREPARE_ON_DEVICE);
    ENUM("Capability", "PAIR_REPORTER", OVSPATIALQUERY_CAPABILITY_PAIR_REPORTER);

#if defined(OVSPATIALQUERY_ABI_WITH_OVSTAGE)
    TYPE("OvstageFilter", ovstage_filter_t);
    TYPE("OvstageScope", ovspatialquery_ovstage_scope_t);
    TYPE("OvstagePopulationDesc", ovspatialquery_ovstage_population_desc_t);
    TYPE("OvstageViewDesc", ovspatialquery_ovstage_view_desc_t);
    TYPE("OvstageParticipationTable", ovspatialquery_ovstage_participation_table_t);
    TYPE("OvstageDroppedTriangles", ovspatialquery_ovstage_dropped_triangles_t);
    FIELDS_2("OvstageFilter", ovstage_filter_t, predicates, count);
    FIELDS_4("OvstageScope", ovspatialquery_ovstage_scope_t, include_paths, exclude_paths, include_count,
             exclude_count);
    FIELDS_4("OvstagePopulationDesc", ovspatialquery_ovstage_population_desc_t, stage, filter, scope, common);
    FIELDS_2("OvstageViewDesc", ovspatialquery_ovstage_view_desc_t, filter, scope);
    FIELDS_4("OvstageParticipationTable", ovspatialquery_ovstage_participation_table_t, default_bits, paths, bits,
             count);
    FIELDS_2("OvstageDroppedTriangles", ovspatialquery_ovstage_dropped_triangles_t, objects, dropped_triangle_counts);
    ENUM("OvstageExclusionReason", "NO_SPATIAL_SOURCE", OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE);
    ENUM("OvstageExclusionReason", "UNSUPPORTED_TYPE", OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_TYPE);
    ENUM("OvstageExclusionReason", "UNSUPPORTED_SHAPE", OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_SHAPE);
    ENUM("OvstageExclusionReason", "MALFORMED_GEOMETRY", OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY);
    ENUM("OvstageExclusionReason", "RESHAPE_PENDING", OVSPATIALQUERY_OVSTAGE_EXCLUSION_RESHAPE_PENDING);
#endif
    return 0;
#endif
}
