/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_ORACLE_BACKEND_H
#define OVSPATIALQUERY_ORACLE_BACKEND_H

/*
 * Deterministic CPU test backend linked into the static test runtime in place
 * of PhysX. This header is safe to include after the public headers: it names
 * only public types. Gates configured on the default runtime govern every
 * instance created through ovspatialquery_create_instance.
 */
#include <ovspatialquery/ovspatialquery.h>

#include <stdint.h>

typedef struct ovspatialquery_backend_runtime_t ovspatialquery_backend_runtime_t;

#ifdef __cplusplus
extern "C"
{
#endif

    ovspatialquery_api_status_t ovspatialquery_test_oracle_create_runtime(
        ovspatialquery_backend_runtime_t** out_runtime);

    void ovspatialquery_test_oracle_destroy_runtime(ovspatialquery_backend_runtime_t* runtime);

    ovspatialquery_backend_runtime_t* ovspatialquery_test_oracle_get_default_runtime(void);

    void ovspatialquery_test_oracle_block_builds(ovspatialquery_backend_runtime_t* runtime, uint32_t expected_builds);

    uint32_t ovspatialquery_test_oracle_wait_for_blocked_builds(ovspatialquery_backend_runtime_t* runtime);

    void ovspatialquery_test_oracle_release_builds(ovspatialquery_backend_runtime_t* runtime);

    void ovspatialquery_test_oracle_block_rays(ovspatialquery_backend_runtime_t* runtime, uint32_t expected_rays);

    uint32_t ovspatialquery_test_oracle_wait_for_blocked_rays(ovspatialquery_backend_runtime_t* runtime);

    void ovspatialquery_test_oracle_release_rays(ovspatialquery_backend_runtime_t* runtime);

    /* The next population build fails with this status. */
    void ovspatialquery_test_oracle_fail_next_build(ovspatialquery_backend_runtime_t* runtime,
                                                    ovspatialquery_api_status_t       failure);

    /* The next ray evaluation fails with this status. */
    void ovspatialquery_test_oracle_fail_next_ray(ovspatialquery_backend_runtime_t* runtime,
                                                  ovspatialquery_api_status_t       failure);

    void ovspatialquery_test_oracle_block_before_publish(ovspatialquery_backend_runtime_t* runtime,
                                                         ovspatialquery_ordinal_t          ordinal);

    uint32_t ovspatialquery_test_oracle_wait_before_publish(ovspatialquery_backend_runtime_t* runtime);

    void ovspatialquery_test_oracle_release_before_publish(ovspatialquery_backend_runtime_t* runtime);

    void ovspatialquery_test_oracle_block_after_publish(ovspatialquery_backend_runtime_t* runtime,
                                                        ovspatialquery_ordinal_t          ordinal);

    uint32_t ovspatialquery_test_oracle_wait_after_publish(ovspatialquery_backend_runtime_t* runtime);

    void ovspatialquery_test_oracle_release_after_publish(ovspatialquery_backend_runtime_t* runtime);

    void ovspatialquery_test_oracle_block_before_completion(ovspatialquery_backend_runtime_t* runtime,
                                                            ovspatialquery_ordinal_t          ordinal,
                                                            uint32_t                          expected_operations);

    uint32_t ovspatialquery_test_oracle_wait_before_completion(ovspatialquery_backend_runtime_t* runtime);

    void ovspatialquery_test_oracle_release_before_completion(ovspatialquery_backend_runtime_t* runtime);

#ifdef __cplusplus
}
#endif

#endif
