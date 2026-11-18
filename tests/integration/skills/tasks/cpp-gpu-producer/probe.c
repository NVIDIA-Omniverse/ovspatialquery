/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * LD_PRELOAD probe for the cpp-gpu-producer verifier: records how a solution drives the
 * GPU, never what it computes. It interposes the ovspatialquery entry points a producer
 * loop calls and the CUDA driver's copies and synchronizations, whether the program links
 * them, looks them up with dlsym, or gets them from cuGetProcAddress. Calls made by the
 * ovspatialquery library itself are forwarded without being counted. At exit it writes a
 * JSON summary to the file named by OVSQ_GPU_PROBE.
 *
 * Built by verify.py: cc -shared -fPIC -O2 -I<sdk>/include probe.c -o probe.so -ldl -lpthread
 */
#define _GNU_SOURCE
#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------------------------ */
/* State                                                                                */
/* ------------------------------------------------------------------------------------ */

enum
{
    KIND_OTHER   = 0,
    KIND_SET     = 1,
    KIND_PUBLISH = 2,
    KIND_LAUNCH  = 3,
    KIND_COUNT   = 4,
};
static const char* const kKindNames[KIND_COUNT] = {"other", "set", "publication", "launch"};

#define MAX_OPS      65536
#define MAX_POINTERS 64
/* Consecutive pending polls of one older op, with no enqueue between them, that count as a
 * wait. For the newest op of its kind one pending poll is enough: an honest loop submits the
 * next step before it sees the step it just submitted finish, however slowly it polls. */
#define SPIN_POLLS 32

typedef struct
{
    ovspatialquery_op_id_t op;
    int                    kind;
    uint64_t               index; /* 1-based index among enqueues of its kind */
    uint32_t               pending_polls;
} op_record_t;

typedef struct
{
    int      kind;
    int      newest; /* the op was the newest of its kind when the host blocked on it */
    uint64_t kind_count;
    uint64_t index;
} block_record_t;

static pthread_mutex_t        g_lock = PTHREAD_MUTEX_INITIALIZER;
static op_record_t            g_ops[MAX_OPS];
static size_t                 g_op_count;
static uint64_t               g_enqueued[KIND_COUNT];
static ovspatialquery_op_id_t g_newest[KIND_COUNT];
static block_record_t         g_blocks[MAX_OPS];
static size_t                 g_block_count;

static uint64_t  g_publish_cuda, g_publish_on_device, g_publish_rows_cuda;
static uint64_t  g_launch_cuda_rows, g_launch_stream, g_launch_admitted, g_launch_retained, g_launch_head;
static uintptr_t g_launch_pointers[MAX_POINTERS];
static size_t    g_launch_pointer_count;
static uint64_t  g_destroy_after_cuda, g_wait_calls;
static double    g_launch_s[MAX_OPS]; /* CLOCK_MONOTONIC at each launch's admission */

/* CUDA traffic while the loop runs: after the first launch and before the last one. */
enum
{
    CU_D2H  = 0,
    CU_H2D  = 1,
    CU_SYNC = 2, /* cuStreamSynchronize / cuCtxSynchronize: waits for everything queued */
    CU_COUNT,
};
static const char* const kCuNames[CU_COUNT] = {"device_to_host_copies", "host_to_device_copies",
                                               "stream_or_context_synchronizations"};
static uint64_t          g_cu_total[CU_COUNT];
static uint64_t          g_cu_at_first_launch[CU_COUNT];
static uint64_t          g_cu_at_last_launch[CU_COUNT];

/* cuEventSynchronize: on an event recorded after the newest launch (a wait for the step
 * just submitted), or on an older one (back-pressure). */
#define MAX_EVENTS 4096
static uintptr_t g_event_handles[MAX_EVENTS];
static uint64_t  g_event_launches[MAX_EVENTS]; /* launches its recording stream depends on */
static uint64_t  g_event_position[MAX_EVENTS]; /* launches enqueued anywhere when recorded */
static size_t    g_event_count;
static uint64_t  g_event_sync_launches[MAX_OPS]; /* per event sync: launches when recorded, now */
static uint64_t  g_event_sync_now[MAX_OPS];
static size_t    g_event_sync_count;
static uint32_t  g_event_pending_queries[MAX_EVENTS]; /* NOT_READY since the last enqueue */
static uint64_t  g_elapsed_time_calls;                /* successful cuEventElapsedTime */
typedef struct
{
    uint64_t start_launches, end_launches; /* launches enqueued when each event was recorded */
    double   ms;
} elapsed_record_t;
static elapsed_record_t g_elapsed[MAX_EVENTS];
static size_t           g_elapsed_count;
/* cuStreamQuery: NOT_READY results since the last admitted enqueue, per stream. */
#define MAX_STREAMS 64
static uintptr_t g_stream_handles[MAX_STREAMS];
static uint32_t  g_stream_pending_queries[MAX_STREAMS];
static uint64_t  g_stream_last_launch[MAX_STREAMS]; /* 1-based index of its newest launch */
static uint8_t   g_stream_nonblocking[MAX_STREAMS]; /* created with CU_STREAM_NON_BLOCKING */
static size_t    g_stream_count;

/* The slot of `stream`, added if new; MAX_STREAMS when the table is full. */
static size_t stream_slot(uintptr_t stream)
{
    size_t i = 0;
    while (i < g_stream_count && g_stream_handles[i] != stream)
    {
        ++i;
    }
    if (i == g_stream_count && g_stream_count < MAX_STREAMS)
    {
        g_stream_handles[g_stream_count++] = stream;
    }
    return i;
}

/* The launches work queued next on `stream` waits for. The legacy default stream (0 or
 * CU_STREAM_LEGACY, 1) implicitly orders after every blocking stream, so it depends on the
 * newest launch queued on any stream not created CU_STREAM_NON_BLOCKING (including itself);
 * other streams on the launches queued on or forwarded to them. */
static uint64_t stream_dependency(uintptr_t stream)
{
    if (stream == 0u || stream == 1u)
    {
        uint64_t newest = 0;
        for (size_t i = 0; i < g_stream_count; ++i)
        {
            if (!g_stream_nonblocking[i] && g_stream_last_launch[i] > newest)
            {
                newest = g_stream_last_launch[i];
            }
        }
        return newest;
    }
    const size_t slot = stream_slot(stream);
    return slot < MAX_STREAMS ? g_stream_last_launch[slot] : 0;
}

/* ------------------------------------------------------------------------------------ */
/* Real symbols                                                                         */
/* ------------------------------------------------------------------------------------ */

typedef void* (*dlsym_fn)(void*, const char*);
static dlsym_fn g_real_dlsym;
static void*    g_library_handle; /* set when the program looks ovspatialquery up with dlsym */

static dlsym_fn real_dlsym(void)
{
    if (g_real_dlsym == NULL)
    {
        static const char* const versions[] = {"GLIBC_2.34", "GLIBC_2.2.5", "GLIBC_2.17"};
        for (size_t i = 0; i < sizeof(versions) / sizeof(versions[0]) && g_real_dlsym == NULL; ++i)
        {
            g_real_dlsym = (dlsym_fn)dlvsym(RTLD_NEXT, "dlsym", versions[i]);
        }
    }
    return g_real_dlsym;
}

static void* next_symbol(const char* name)
{
    void* symbol = real_dlsym()(RTLD_NEXT, name);
    if (symbol == NULL && g_library_handle != NULL)
    {
        symbol = real_dlsym()(g_library_handle, name);
    }
    if (symbol == NULL)
    {
        fprintf(stderr, "probe: cannot resolve %s\n", name);
        abort();
    }
    return symbol;
}

/* True when `address` is inside the ovspatialquery or CUDA driver library, whose own
 * calls are not the program's. */
static int library_caller(const void* address)
{
    Dl_info info;
    if (address == NULL || dladdr(address, &info) == 0 || info.dli_fname == NULL)
    {
        return 0;
    }
    return strstr(info.dli_fname, "libovspatialquery") != NULL || strstr(info.dli_fname, "libcuda") != NULL;
}

/* ------------------------------------------------------------------------------------ */
/* Operation bookkeeping                                                                */
/* ------------------------------------------------------------------------------------ */

static op_record_t* find_op(ovspatialquery_op_id_t op)
{
    for (size_t i = g_op_count; i > 0; --i)
    {
        if (g_ops[i - 1].op.raw_id == op.raw_id)
        {
            return &g_ops[i - 1];
        }
    }
    return NULL;
}

static void reset_spins(void)
{
    for (size_t i = 0; i < g_op_count; ++i)
    {
        g_ops[i].pending_polls = 0;
    }
    memset(g_event_pending_queries, 0, sizeof(g_event_pending_queries));
    memset(g_stream_pending_queries, 0, sizeof(g_stream_pending_queries));
}

static void record_enqueue(int kind, ovspatialquery_enqueue_result_t result)
{
    pthread_mutex_lock(&g_lock);
    if (result.status == OVSPATIALQUERY_API_SUCCESS)
    {
        reset_spins(); /* only admitted work separates two observations of an op */
    }
    if (result.status == OVSPATIALQUERY_API_SUCCESS && g_op_count < MAX_OPS)
    {
        ++g_enqueued[kind];
        g_newest[kind]    = result.op_id;
        g_ops[g_op_count] = (op_record_t){result.op_id, kind, g_enqueued[kind], 0};
        ++g_op_count;
        if (kind == KIND_LAUNCH)
        {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            if (g_enqueued[kind] <= MAX_OPS)
            {
                g_launch_s[g_enqueued[kind] - 1] = (double)now.tv_sec + 1e-9 * (double)now.tv_nsec;
            }
            if (g_enqueued[kind] == 1)
            {
                memcpy(g_cu_at_first_launch, g_cu_total, sizeof(g_cu_total));
            }
            memcpy(g_cu_at_last_launch, g_cu_total, sizeof(g_cu_total));
        }
    }
    pthread_mutex_unlock(&g_lock);
}

static void record_block(const op_record_t* record)
{
    if (record == NULL || g_block_count >= MAX_OPS)
    {
        return;
    }
    g_blocks[g_block_count++] = (block_record_t){
        record->kind,
        g_newest[record->kind].raw_id == record->op.raw_id,
        g_enqueued[record->kind],
        record->index,
    };
}

/* One nonblocking observation of an op: a run of SPIN_POLLS pending observations with no
 * enqueue between them, ended by a terminal one, is a wait by another name. */
static void observe_poll(op_record_t* record, int pending)
{
    if (record == NULL)
    {
        return;
    }
    if (pending)
    {
        ++record->pending_polls;
        return;
    }
    const int newest = g_newest[record->kind].raw_id == record->op.raw_id;
    if (record->pending_polls >= (newest ? 1u : SPIN_POLLS))
    {
        record_block(record);
    }
    record->pending_polls = 0;
}

static int device_is_cuda(const ovspatialquery_input_array_t* column)
{
    return column != NULL && column->tensor.device.device_type == kDLCUDA;
}

static void record_launch(const ovspatialquery_input_array_t* rows, ovspatialquery_ordinal_selection_t ordinal,
                          ovspatialquery_cuda_sync_t sync)
{
    pthread_mutex_lock(&g_lock);
    g_launch_cuda_rows += device_is_cuda(rows);
    g_launch_stream += sync.has_stream == 1u;
    if (sync.has_stream == 1u)
    {
        const size_t slot = stream_slot(sync.stream);
        if (slot < MAX_STREAMS)
        {
            g_stream_last_launch[slot] = g_enqueued[KIND_LAUNCH] + 1; /* admitted next */
        }
    }
    g_launch_admitted += ordinal.kind == OVSPATIALQUERY_ORDINAL_ADMITTED_EXACT;
    g_launch_retained += ordinal.kind == OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    g_launch_head += ordinal.kind == OVSPATIALQUERY_ORDINAL_HEAD;
    if (rows != NULL)
    {
        const uintptr_t pointer = (uintptr_t)rows->tensor.data + (uintptr_t)rows->tensor.byte_offset;
        size_t          i       = 0;
        while (i < g_launch_pointer_count && g_launch_pointers[i] != pointer)
        {
            ++i;
        }
        if (i == g_launch_pointer_count && g_launch_pointer_count < MAX_POINTERS)
        {
            g_launch_pointers[g_launch_pointer_count++] = pointer;
        }
    }
    pthread_mutex_unlock(&g_lock);
}

/* ------------------------------------------------------------------------------------ */
/* ovspatialquery interposers                                                           */
/* ------------------------------------------------------------------------------------ */

#define REAL(name) ((__typeof__(&name))next_symbol(#name))

ovspatialquery_enqueue_result_t ovspatialquery_enqueue_explicit_population_set(
    ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
    const ovspatialquery_explicit_population_set_t* set)
{
    static __typeof__(&ovspatialquery_enqueue_explicit_population_set) real;
    if (real == NULL)
    {
        real = REAL(ovspatialquery_enqueue_explicit_population_set);
    }
    const ovspatialquery_enqueue_result_t result = real(instance, population, set);
    if (!library_caller(__builtin_return_address(0)))
    {
        record_enqueue(KIND_SET, result);
    }
    return result;
}

ovspatialquery_enqueue_result_t ovspatialquery_enqueue_explicit_population_update(
    ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
    const ovspatialquery_explicit_population_update_t* update)
{
    static __typeof__(&ovspatialquery_enqueue_explicit_population_update) real;
    if (real == NULL)
    {
        real = REAL(ovspatialquery_enqueue_explicit_population_update);
    }
    const ovspatialquery_enqueue_result_t result = real(instance, population, update);
    if (!library_caller(__builtin_return_address(0)))
    {
        if (result.status == OVSPATIALQUERY_API_SUCCESS && update != NULL)
        {
            const ovspatialquery_dynamic_execution_t* execution = update->dynamic_execution;
            pthread_mutex_lock(&g_lock);
            g_publish_cuda += execution != NULL && execution->device.device_type == kDLCUDA;
            g_publish_on_device += execution != NULL && execution->preparation != OVSPATIALQUERY_PREPARE_ON_HOST;
            g_publish_rows_cuda += device_is_cuda(update->changed_instances.translations);
            pthread_mutex_unlock(&g_lock);
        }
        record_enqueue(KIND_PUBLISH, result);
    }
    return result;
}

/* Registers a launch's library-owned result_ready event as covering that launch. */
static void note_result_ready(uintptr_t event);

#define LAUNCH_INTERPOSER(name, batch_type, reservation_type, rows)                                                    \
    ovspatialquery_enqueue_result_t name(ovspatialquery_instance_handle_t   instance,                                  \
                                         ovspatialquery_batch_plan_handle_t plan, const batch_type* batch,             \
                                         reservation_type* reservation)                                                \
    {                                                                                                                  \
        static __typeof__(&name) real;                                                                                 \
        if (real == NULL)                                                                                              \
        {                                                                                                              \
            real = REAL(name);                                                                                         \
        }                                                                                                              \
        const ovspatialquery_enqueue_result_t result = real(instance, plan, batch, reservation);                       \
        if (!library_caller(__builtin_return_address(0)))                                                              \
        {                                                                                                              \
            if (result.status == OVSPATIALQUERY_API_SUCCESS && batch != NULL)                                          \
            {                                                                                                          \
                record_launch(batch->rows, batch->ordinal, batch->cuda_sync);                                          \
            }                                                                                                          \
            record_enqueue(KIND_LAUNCH, result);                                                                       \
            if (result.status == OVSPATIALQUERY_API_SUCCESS && reservation != NULL)                                    \
            {                                                                                                          \
                note_result_ready(reservation->common.result_ready.value);                                             \
            }                                                                                                          \
        }                                                                                                              \
        return result;                                                                                                 \
    }

LAUNCH_INTERPOSER(ovspatialquery_enqueue_overlap_plan, ovspatialquery_overlap_batch_t,
                  ovspatialquery_overlap_batch_reservation_t, translations)
LAUNCH_INTERPOSER(ovspatialquery_enqueue_raycast_plan, ovspatialquery_raycast_batch_t,
                  ovspatialquery_raycast_batch_reservation_t, origins)
LAUNCH_INTERPOSER(ovspatialquery_enqueue_sphere_sweep_plan, ovspatialquery_sphere_sweep_batch_t,
                  ovspatialquery_sphere_sweep_batch_reservation_t, centers)
LAUNCH_INTERPOSER(ovspatialquery_enqueue_box_sweep_plan, ovspatialquery_box_sweep_batch_t,
                  ovspatialquery_box_sweep_batch_reservation_t, centers)

ovspatialquery_result_t ovspatialquery_wait_op(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op_id,
                                               ovspatialquery_timeout_t timeout, ovspatialquery_op_wait_result_t* out)
{
    static __typeof__(&ovspatialquery_wait_op)         real;
    static __typeof__(&ovspatialquery_query_op_status) status;
    if (real == NULL)
    {
        real   = REAL(ovspatialquery_wait_op);
        status = REAL(ovspatialquery_query_op_status);
    }
    if (library_caller(__builtin_return_address(0)))
    {
        return real(instance, op_id, timeout, out);
    }
    if (timeout == 0)
    {
        /* A zero timeout is a poll: account for the state it returned to the caller. */
        const ovspatialquery_result_t result = real(instance, op_id, timeout, out);
        pthread_mutex_lock(&g_lock);
        ++g_wait_calls;
        /* TIMEOUT: this call observed pending work; SUCCESS writes the state it saw. */
        if (result.status == OVSPATIALQUERY_API_TIMEOUT)
        {
            observe_poll(find_op(op_id), 1);
        }
        else if (result.status == OVSPATIALQUERY_API_SUCCESS && out != NULL)
        {
            observe_poll(find_op(op_id), out->state == OVSPATIALQUERY_OP_PENDING);
        }
        pthread_mutex_unlock(&g_lock);
        return result;
    }
    /* Peek without blocking: a wait on a terminal op costs nothing. */
    ovspatialquery_op_status_t observed;
    memset(&observed, 0, sizeof(observed));
    const ovspatialquery_result_t peeked = status(instance, op_id, &observed);
    pthread_mutex_lock(&g_lock);
    ++g_wait_calls;
    if (peeked.status == OVSPATIALQUERY_API_SUCCESS && observed.state == OVSPATIALQUERY_OP_PENDING)
    {
        record_block(find_op(op_id));
    }
    pthread_mutex_unlock(&g_lock);
    return real(instance, op_id, timeout, out);
}

/* A typed fetch observes the op too: NOT_READY while it is pending, SUCCESS or
 * OPERATION_FAILED once it ended. Polling it until it succeeds is a wait like any other. */
#define FETCH_INTERPOSER(name, result_type)                                                                            \
    ovspatialquery_result_t name(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op_id,              \
                                 result_type* out_result)                                                              \
    {                                                                                                                  \
        static __typeof__(&name) real;                                                                                 \
        if (real == NULL)                                                                                              \
        {                                                                                                              \
            real = REAL(name);                                                                                         \
        }                                                                                                              \
        const ovspatialquery_result_t result = real(instance, op_id, out_result);                                      \
        if (!library_caller(__builtin_return_address(0)) &&                                                            \
            (result.status == OVSPATIALQUERY_API_NOT_READY || result.status == OVSPATIALQUERY_API_SUCCESS ||           \
             result.status == OVSPATIALQUERY_API_OPERATION_FAILED))                                                    \
        {                                                                                                              \
            pthread_mutex_lock(&g_lock);                                                                               \
            observe_poll(find_op(op_id), result.status == OVSPATIALQUERY_API_NOT_READY);                               \
            pthread_mutex_unlock(&g_lock);                                                                             \
        }                                                                                                              \
        return result;                                                                                                 \
    }

FETCH_INTERPOSER(ovspatialquery_fetch_overlap_batch, ovspatialquery_overlap_batch_result_t)
FETCH_INTERPOSER(ovspatialquery_fetch_raycast_batch, ovspatialquery_raycast_batch_result_t)
FETCH_INTERPOSER(ovspatialquery_fetch_sphere_sweep_batch, ovspatialquery_sphere_sweep_batch_result_t)
FETCH_INTERPOSER(ovspatialquery_fetch_box_sweep_batch, ovspatialquery_box_sweep_batch_result_t)

ovspatialquery_result_t ovspatialquery_query_op_status(ovspatialquery_instance_handle_t instance,
                                                       ovspatialquery_op_id_t op_id, ovspatialquery_op_status_t* out)
{
    static __typeof__(&ovspatialquery_query_op_status) real;
    if (real == NULL)
    {
        real = REAL(ovspatialquery_query_op_status);
    }
    const ovspatialquery_result_t result = real(instance, op_id, out);
    if (!library_caller(__builtin_return_address(0)) && result.status == OVSPATIALQUERY_API_SUCCESS && out != NULL)
    {
        pthread_mutex_lock(&g_lock);
        observe_poll(find_op(op_id), out->state == OVSPATIALQUERY_OP_PENDING);
        pthread_mutex_unlock(&g_lock);
    }
    return result;
}

ovspatialquery_result_t ovspatialquery_destroy_op_after_cuda(ovspatialquery_instance_handle_t instance,
                                                             ovspatialquery_op_id_t*          inout_op_id,
                                                             ovspatialquery_cuda_event_t*     inout_consumer_done)
{
    static __typeof__(&ovspatialquery_destroy_op_after_cuda) real;
    if (real == NULL)
    {
        real = REAL(ovspatialquery_destroy_op_after_cuda);
    }
    const ovspatialquery_result_t result = real(instance, inout_op_id, inout_consumer_done);
    if (!library_caller(__builtin_return_address(0)) && result.status == OVSPATIALQUERY_API_SUCCESS)
    {
        pthread_mutex_lock(&g_lock);
        ++g_destroy_after_cuda;
        pthread_mutex_unlock(&g_lock);
    }
    return result;
}

/* ------------------------------------------------------------------------------------ */
/* CUDA driver interposers                                                              */
/* ------------------------------------------------------------------------------------ */

/* Every interposed driver function takes at most six integer or pointer arguments, so one
 * forwarding shape fits all of them on x86-64 and AArch64. */
typedef uintptr_t (*cu_fn)(uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t);
typedef int (*cu_pointer_attribute_fn)(void*, int, uintptr_t);

enum
{
    CLASS_D2H,
    CLASS_H2D,
    CLASS_SYNC,
    CLASS_UNIFIED, /* cuMemcpy / cuMemcpyAsync: direction from the pointers */
    CLASS_COPY2D,  /* CUDA_MEMCPY2D descriptor: direction from its memory types */
    CLASS_COPY3D,  /* CUDA_MEMCPY3D descriptor */
    CLASS_EVENT_RECORD,
    CLASS_EVENT_SYNC,
    CLASS_EVENT_QUERY,   /* a spin on one event until it completes is a synchronization */
    CLASS_STREAM_QUERY,  /* so is a stream seen busy and then idle */
    CLASS_STREAM_SYNC,   /* a wait for one stream: newest work only on the newest launch's stream */
    CLASS_STREAM_WAIT,   /* a stream made to wait on an event inherits the launches it covers */
    CLASS_STREAM_CREATE, /* remembers whether a stream is non-blocking */
    CLASS_EVENT_ELAPSED, /* evidence of CUDA-event timing */
};

typedef struct
{
    const char* name;
    int         klass;
    cu_fn       real;
} cu_entry_t;

#define CU_ENTRIES(X)                                                                                                  \
    X(cuMemcpyDtoH, CLASS_D2H)                                                                                         \
    X(cuMemcpyDtoH_v2, CLASS_D2H)                                                                                      \
    X(cuMemcpyDtoHAsync, CLASS_D2H)                                                                                    \
    X(cuMemcpyDtoHAsync_v2, CLASS_D2H)                                                                                 \
    X(cuMemcpyDtoHAsync_v2_ptsz, CLASS_D2H)                                                                            \
    X(cuMemcpyDtoH_v2_ptds, CLASS_D2H)                                                                                 \
    X(cuMemcpyHtoD, CLASS_H2D)                                                                                         \
    X(cuMemcpyHtoD_v2, CLASS_H2D)                                                                                      \
    X(cuMemcpyHtoDAsync, CLASS_H2D)                                                                                    \
    X(cuMemcpyHtoDAsync_v2, CLASS_H2D)                                                                                 \
    X(cuMemcpyHtoDAsync_v2_ptsz, CLASS_H2D)                                                                            \
    X(cuMemcpyHtoD_v2_ptds, CLASS_H2D)                                                                                 \
    X(cuMemcpy, CLASS_UNIFIED)                                                                                         \
    X(cuMemcpy2D, CLASS_COPY2D)                                                                                        \
    X(cuMemcpy2D_v2, CLASS_COPY2D)                                                                                     \
    X(cuMemcpy2D_v2_ptds, CLASS_COPY2D)                                                                                \
    X(cuMemcpy2DUnaligned, CLASS_COPY2D)                                                                               \
    X(cuMemcpy2DUnaligned_v2, CLASS_COPY2D)                                                                            \
    X(cuMemcpy2DUnaligned_v2_ptds, CLASS_COPY2D)                                                                       \
    X(cuMemcpy2DAsync, CLASS_COPY2D)                                                                                   \
    X(cuMemcpy2DAsync_v2, CLASS_COPY2D)                                                                                \
    X(cuMemcpy2DAsync_v2_ptsz, CLASS_COPY2D)                                                                           \
    X(cuMemcpy3D, CLASS_COPY3D)                                                                                        \
    X(cuMemcpy3D_v2, CLASS_COPY3D)                                                                                     \
    X(cuMemcpy3D_v2_ptds, CLASS_COPY3D)                                                                                \
    X(cuMemcpy3DAsync, CLASS_COPY3D)                                                                                   \
    X(cuMemcpy3DAsync_v2, CLASS_COPY3D)                                                                                \
    X(cuMemcpy3DAsync_v2_ptsz, CLASS_COPY3D)                                                                           \
    X(cuMemcpyAsync, CLASS_UNIFIED)                                                                                    \
    X(cuMemcpy_ptds, CLASS_UNIFIED)                                                                                    \
    X(cuMemcpyAsync_ptsz, CLASS_UNIFIED)                                                                               \
    X(cuStreamSynchronize, CLASS_STREAM_SYNC)                                                                          \
    X(cuStreamWaitEvent, CLASS_STREAM_WAIT)                                                                            \
    X(cuStreamCreate, CLASS_STREAM_CREATE)                                                                             \
    X(cuStreamCreateWithPriority, CLASS_STREAM_CREATE)                                                                 \
    X(cuStreamWaitEvent_ptsz, CLASS_STREAM_WAIT)                                                                       \
    X(cuStreamSynchronize_ptsz, CLASS_STREAM_SYNC)                                                                     \
    X(cuCtxSynchronize, CLASS_SYNC)                                                                                    \
    X(cuEventRecord, CLASS_EVENT_RECORD)                                                                               \
    X(cuEventRecord_ptsz, CLASS_EVENT_RECORD)                                                                          \
    X(cuEventRecordWithFlags, CLASS_EVENT_RECORD)                                                                      \
    X(cuEventRecordWithFlags_ptsz, CLASS_EVENT_RECORD)                                                                 \
    X(cuEventSynchronize, CLASS_EVENT_SYNC)                                                                            \
    X(cuEventQuery, CLASS_EVENT_QUERY)                                                                                 \
    X(cuStreamQuery, CLASS_STREAM_QUERY)                                                                               \
    X(cuStreamQuery_ptsz, CLASS_STREAM_QUERY)                                                                          \
    X(cuEventElapsedTime, CLASS_EVENT_ELAPSED)                                                                         \
    X(cuEventElapsedTime_v2, CLASS_EVENT_ELAPSED)

#define CU_INDEX(name, klass) IDX_##name,
enum
{
    CU_ENTRIES(CU_INDEX) CU_ENTRY_COUNT
};
#define CU_ROW(name, klass) {#name, klass, NULL},
static cu_entry_t g_cu[CU_ENTRY_COUNT] = {CU_ENTRIES(CU_ROW)};

static void* g_libcuda;

static void* libcuda(void)
{
    if (g_libcuda == NULL)
    {
        g_libcuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    }
    return g_libcuda;
}

static int is_device_pointer(uintptr_t pointer)
{
    static cu_pointer_attribute_fn attribute;
    if (attribute == NULL && libcuda() != NULL)
    {
        attribute = (cu_pointer_attribute_fn)real_dlsym()(libcuda(), "cuPointerGetAttribute");
    }
    unsigned int type = 0;
    /* CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2; CU_MEMORYTYPE_DEVICE = 2 */
    return attribute != NULL && attribute(&type, 2, pointer) == 0 && type == 2u;
}

static size_t find_event(uintptr_t event)
{
    size_t i = 0;
    while (i < g_event_count && g_event_handles[i] != event)
    {
        ++i;
    }
    return i;
}

static void record_event_sync(uintptr_t event)
{
    if (g_event_sync_count < MAX_OPS)
    {
        const size_t i                            = find_event(event);
        g_event_sync_launches[g_event_sync_count] = i < g_event_count ? g_event_launches[i] : 0;
        g_event_sync_now[g_event_sync_count]      = g_enqueued[KIND_LAUNCH];
        ++g_event_sync_count;
    }
}

/* Called after the driver returned `result` (0 is CUDA_SUCCESS, 600 CUDA_ERROR_NOT_READY). */
static void note_result_ready(uintptr_t event)
{
    if (event == 0)
    {
        return;
    }
    pthread_mutex_lock(&g_lock);
    size_t i = find_event(event);
    if (i == g_event_count && g_event_count < MAX_EVENTS)
    {
        g_event_handles[g_event_count++] = event;
    }
    if (i < MAX_EVENTS)
    {
        g_event_launches[i] = g_enqueued[KIND_LAUNCH]; /* the launch just admitted */
        g_event_position[i] = g_enqueued[KIND_LAUNCH];
    }
    pthread_mutex_unlock(&g_lock);
}

static void count_cu(int klass, uintptr_t a, uintptr_t b, uintptr_t c, uintptr_t result)
{
    int counted = -1;
    if (klass == CLASS_COPY2D || klass == CLASS_COPY3D)
    {
        /* Byte offsets of srcMemoryType, srcDevice, dstMemoryType and dstDevice in
         * CUDA_MEMCPY2D and CUDA_MEMCPY3D (64-bit driver ABI). CU_MEMORYTYPE_HOST is 1,
         * DEVICE 2, ARRAY 3, UNIFIED 4 (classified by its pointer). */
        const unsigned char* d        = (const unsigned char*)a;
        const size_t         src_type = klass == CLASS_COPY2D ? 16 : 32, src_ptr = klass == CLASS_COPY2D ? 32 : 48;
        const size_t         dst_type = klass == CLASS_COPY2D ? 72 : 120, dst_ptr = klass == CLASS_COPY2D ? 88 : 136;
        if (d != NULL)
        {
            unsigned int st = 0, dt = 0;
            uintptr_t    sp = 0, dp = 0;
            memcpy(&st, d + src_type, sizeof(st));
            memcpy(&dt, d + dst_type, sizeof(dt));
            memcpy(&sp, d + src_ptr, sizeof(sp));
            memcpy(&dp, d + dst_ptr, sizeof(dp));
            const int src_device = st == 2u || st == 3u || (st == 4u && is_device_pointer(sp));
            const int dst_device = dt == 2u || dt == 3u || (dt == 4u && is_device_pointer(dp));
            counted              = (!dst_device && src_device) ? CU_D2H : (dst_device && !src_device) ? CU_H2D : -1;
        }
    }
    else if (klass == CLASS_UNIFIED)
    {
        const int dst_device = is_device_pointer(a);
        const int src_device = is_device_pointer(b);
        counted              = (!dst_device && src_device) ? CU_D2H : (dst_device && !src_device) ? CU_H2D : -1;
    }
    else if (klass == CLASS_D2H || klass == CLASS_H2D || klass == CLASS_SYNC)
    {
        counted = klass == CLASS_D2H ? CU_D2H : klass == CLASS_H2D ? CU_H2D : CU_SYNC;
    }
    pthread_mutex_lock(&g_lock);
    if (counted >= 0)
    {
        ++g_cu_total[counted];
    }
    else if (klass == CLASS_EVENT_RECORD)
    {
        size_t i = find_event(a);
        if (i == g_event_count && g_event_count < MAX_EVENTS)
        {
            g_event_handles[g_event_count++] = a;
        }
        if (i < MAX_EVENTS)
        {
            /* cuEventRecord(event, stream): the event completes after the work its stream
             * depends on, so it covers the newest launch only if that stream carries it. */
            g_event_launches[i] = stream_dependency(b);
            g_event_position[i] = g_enqueued[KIND_LAUNCH];
        }
    }
    else if (klass == CLASS_EVENT_SYNC)
    {
        record_event_sync(a);
    }
    else if (klass == CLASS_EVENT_QUERY)
    {
        /* One nonblocking query is a poll; a run of them on one event, with no enqueue
         * between, that ends in completion is a synchronization on that event. */
        const size_t i = find_event(a);
        if (i < g_event_count)
        {
            if ((int)result == 600)
            {
                ++g_event_pending_queries[i];
            }
            else
            {
                /* An event recorded after the newest launch covers the step just submitted. */
                const int newest = g_event_launches[i] >= g_enqueued[KIND_LAUNCH];
                if ((int)result == 0 && g_event_pending_queries[i] >= (newest ? 1u : SPIN_POLLS))
                {
                    record_event_sync(a);
                }
                g_event_pending_queries[i] = 0;
            }
        }
    }
    else if (klass == CLASS_EVENT_ELAPSED && (int)result == 0 && a != 0)
    {
        /* cuEventElapsedTime(float* ms, start, end): keep the value, and whether the start
         * was recorded before the first launch, so the report can find the loop's span. */
        ++g_elapsed_time_calls;
        const size_t start = find_event(b), end = find_event(c);
        if (start < g_event_count && end < g_event_count && g_elapsed_count < MAX_EVENTS)
        {
            g_elapsed[g_elapsed_count++] = (elapsed_record_t){
                g_event_position[start],
                /* the end counts only the launches the GPU orders it after:
                 * an event on an idle stream covers none */
                g_event_launches[end] < g_event_position[end] ? g_event_launches[end] : g_event_position[end],
                *(const float*)a};
        }
    }
    else if (klass == CLASS_STREAM_CREATE && (int)result == 0 && a != 0)
    {
        /* cuStreamCreate(CUstream* out, unsigned flags[, int priority]) */
        const size_t i = stream_slot(*(const uintptr_t*)a);
        if (i < MAX_STREAMS)
        {
            g_stream_nonblocking[i] = (b & 1u) != 0; /* CU_STREAM_NON_BLOCKING */
        }
    }
    else if (klass == CLASS_STREAM_WAIT && (int)result == 0)
    {
        /* cuStreamWaitEvent(stream, event, flags): work later on `stream` runs after the
         * launches the event was recorded behind, so waiting for `stream` waits for them. */
        const size_t i = stream_slot(a), event = find_event(b);
        if (i < MAX_STREAMS && event < g_event_count && g_event_launches[event] > g_stream_last_launch[i])
        {
            g_stream_last_launch[i] = g_event_launches[event];
        }
    }
    else if (klass == CLASS_STREAM_SYNC && (int)result == 0)
    {
        /* Waiting for a stream of older steps is back-pressure; for the stream that holds
         * the newest launch it is a wait for the step just submitted. */
        const uint64_t newest = g_enqueued[KIND_LAUNCH];
        if (newest == 0 || stream_dependency(a) == newest)
        {
            ++g_cu_total[CU_SYNC];
        }
    }
    else if (klass == CLASS_STREAM_QUERY)
    {
        /* Seeing a stream busy and then idle, with no work admitted between, waited for
         * everything queued on it. That is a wait for the step just submitted only when the
         * newest launch was queued on this stream; polling a stream of older steps is
         * back-pressure. */
        const size_t i = stream_slot(a);
        if (i < MAX_STREAMS)
        {
            if ((int)result == 600)
            {
                ++g_stream_pending_queries[i];
            }
            else
            {
                const uint64_t newest = g_enqueued[KIND_LAUNCH];
                if ((int)result == 0 && g_stream_pending_queries[i] > 0 && newest > 0 && stream_dependency(a) == newest)
                {
                    ++g_cu_total[CU_SYNC];
                }
                g_stream_pending_queries[i] = 0;
            }
        }
    }
    pthread_mutex_unlock(&g_lock);
}

static cu_fn resolve_cu(size_t index)
{
    if (g_cu[index].real == NULL && libcuda() != NULL)
    {
        g_cu[index].real = (cu_fn)real_dlsym()(libcuda(), g_cu[index].name);
    }
    return g_cu[index].real;
}

#define CU_WRAPPER(name, klass)                                                                                        \
    static uintptr_t wrap_##name(uintptr_t a, uintptr_t b, uintptr_t c, uintptr_t d, uintptr_t e, uintptr_t f)         \
    {                                                                                                                  \
        cu_fn real = resolve_cu(IDX_##name);                                                                           \
        if (real == NULL)                                                                                              \
        {                                                                                                              \
            return 999; /* CUDA_ERROR_UNKNOWN */                                                                       \
        }                                                                                                              \
        const uintptr_t result = real(a, b, c, d, e, f);                                                               \
        count_cu(klass, a, b, c, result);                                                                              \
        return result;                                                                                                 \
    }                                                                                                                  \
    uintptr_t name(uintptr_t a, uintptr_t b, uintptr_t c, uintptr_t d, uintptr_t e, uintptr_t f)                       \
    {                                                                                                                  \
        if (library_caller(__builtin_return_address(0)))                                                               \
        {                                                                                                              \
            return resolve_cu(IDX_##name)(a, b, c, d, e, f);                                                           \
        }                                                                                                              \
        return wrap_##name(a, b, c, d, e, f);                                                                          \
    }
CU_ENTRIES(CU_WRAPPER)

#define CU_WRAPPER_POINTER(name, klass) (void*)wrap_##name,
static void* const g_cu_wrappers[CU_ENTRY_COUNT] = {CU_ENTRIES(CU_WRAPPER_POINTER)};

void report_probe_marker(void);

/* True when `address` is inside this probe, e.g. what dlsym(RTLD_DEFAULT) finds first. */
static int probe_owned(const void* address)
{
    Dl_info self, other;
    return address != NULL && dladdr((void*)&report_probe_marker, &self) != 0 && dladdr(address, &other) != 0 &&
           self.dli_fbase == other.dli_fbase;
}

static void* cu_wrapper_for(const char* name, void* real)
{
    for (size_t i = 0; i < CU_ENTRY_COUNT; ++i)
    {
        if (strcmp(g_cu[i].name, name) == 0)
        {
            /* Only a driver address is cached; anything else resolves through libcuda. */
            if (real != NULL && !probe_owned(real))
            {
                g_cu[i].real = (cu_fn)real;
            }
            return g_cu_wrappers[i];
        }
    }
    return NULL;
}

/* cuGetProcAddress names the base symbol ("cuMemcpyDtoH") and picks the ABI version; the
 * wrapper forwards to whatever the driver returned. */
typedef int (*get_proc_fn)(const char*, void**, int, uint64_t, void*);

static int get_proc(const char* entry, const char* symbol, void** pfn, int version, uint64_t flags, void* status,
                    const void* caller)
{
    get_proc_fn real = (get_proc_fn)real_dlsym()(libcuda(), entry);
    if (real == NULL)
    {
        return 999;
    }
    const int result = real(symbol, pfn, version, flags, status);
    if (result == 0 && pfn != NULL && *pfn != NULL && !library_caller(caller))
    {
        /* The base name may come back as the _v2 ABI, and with
         * CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM (2) as the per-thread-stream variant:
         * each variant has its own entry, so one lookup never retargets another's wrapper. */
        static const char* const legacy[] = {"_v2", ""};
        /* A function with no per-thread variant resolves to its ordinary entry. */
        static const char* const per_thread[] = {"_v2_ptsz", "_v2_ptds", "_ptsz", "_ptds", "_v2", ""};
        const int                threaded     = (flags & 2u) != 0;
        const char* const*       suffixes     = threaded ? per_thread : legacy;
        const size_t             count        = threaded ? 6 : 2;
        void*                    wrapper      = NULL;
        for (size_t i = 0; i < count && wrapper == NULL; ++i)
        {
            char name[160];
            snprintf(name, sizeof(name), "%s%s", symbol, suffixes[i]);
            wrapper = cu_wrapper_for(name, *pfn);
        }
        if (wrapper != NULL)
        {
            *pfn = wrapper;
        }
    }
    return result;
}

int cuGetProcAddress(const char* symbol, void** pfn, int version, uint64_t flags, void* status)
{
    return get_proc("cuGetProcAddress", symbol, pfn, version, flags, status, __builtin_return_address(0));
}

int cuGetProcAddress_v2(const char* symbol, void** pfn, int version, uint64_t flags, void* status)
{
    return get_proc("cuGetProcAddress_v2", symbol, pfn, version, flags, status, __builtin_return_address(0));
}

void report_probe_marker(void);
void report_probe_marker(void) {}

/* ------------------------------------------------------------------------------------ */
/* dlsym                                                                                */
/* ------------------------------------------------------------------------------------ */

void* dlsym(void* handle, const char* name)
{
    void* symbol = real_dlsym()(handle, name);
    if (symbol == NULL || library_caller(__builtin_return_address(0)))
    {
        return symbol;
    }
    if (strncmp(name, "ovspatialquery_", 15) == 0)
    {
        if (handle != RTLD_DEFAULT && handle != RTLD_NEXT && g_library_handle == NULL)
        {
            g_library_handle = handle;
        }
        /* Replace only the entry points this probe defines. */
        Dl_info self;
        if (dladdr((void*)&report_probe_marker, &self) != 0)
        {
            void*   probe = dlopen(self.dli_fname, RTLD_NOW | RTLD_NOLOAD);
            void*   ours  = probe != NULL ? real_dlsym()(probe, name) : NULL;
            Dl_info where;
            if (ours != NULL && dladdr(ours, &where) != 0 && where.dli_fbase == self.dli_fbase)
            {
                return ours;
            }
        }
        return symbol;
    }
    if (strncmp(name, "cuGetProcAddress", 16) == 0)
    {
        return strcmp(name, "cuGetProcAddress_v2") == 0 ? (void*)&cuGetProcAddress_v2 : (void*)&cuGetProcAddress;
    }
    if (strncmp(name, "cu", 2) == 0)
    {
        void* wrapper = cu_wrapper_for(name, symbol);
        if (wrapper != NULL)
        {
            return wrapper;
        }
    }
    return symbol;
}

/* ------------------------------------------------------------------------------------ */
/* Report                                                                               */
/* ------------------------------------------------------------------------------------ */

__attribute__((destructor)) static void report(void)
{
    const char* path = getenv("OVSQ_GPU_PROBE");
    if (path == NULL)
    {
        return;
    }
    FILE* out = fopen(path, "w");
    if (out == NULL)
    {
        return;
    }
    pthread_mutex_lock(&g_lock);
    fprintf(out, "{\n  \"enqueued\": {");
    for (int kind = 0; kind < KIND_COUNT; ++kind)
    {
        fprintf(out, "%s\"%s\": %llu", kind ? ", " : "", kKindNames[kind], (unsigned long long)g_enqueued[kind]);
    }
    /* A block on the newest op of its kind, while more of that kind were enqueued later,
     * means the host waited for the step it had just submitted. */
    uint64_t newest_blocks[KIND_COUNT] = {0};
    uint64_t back_pressure[KIND_COUNT] = {0};
    uint64_t drain_blocks[KIND_COUNT]  = {0};
    for (size_t i = 0; i < g_block_count; ++i)
    {
        const block_record_t* block = &g_blocks[i];
        if (block->kind_count >= g_enqueued[block->kind])
        {
            ++drain_blocks[block->kind];
        }
        else if (block->newest)
        {
            ++newest_blocks[block->kind];
        }
        else
        {
            ++back_pressure[block->kind];
        }
    }
    fprintf(out, "},\n  \"blocking_waits_on_newest\": {");
    for (int kind = 0; kind < KIND_COUNT; ++kind)
    {
        fprintf(out, "%s\"%s\": %llu", kind ? ", " : "", kKindNames[kind], (unsigned long long)newest_blocks[kind]);
    }
    fprintf(out, "},\n  \"blocking_waits_back_pressure\": {");
    for (int kind = 0; kind < KIND_COUNT; ++kind)
    {
        fprintf(out, "%s\"%s\": %llu", kind ? ", " : "", kKindNames[kind], (unsigned long long)back_pressure[kind]);
    }
    fprintf(out, "},\n  \"blocking_waits_after_last_enqueue\": {");
    for (int kind = 0; kind < KIND_COUNT; ++kind)
    {
        fprintf(out, "%s\"%s\": %llu", kind ? ", " : "", kKindNames[kind], (unsigned long long)drain_blocks[kind]);
    }
    fprintf(out,
            "},\n  \"publications\": {\"cuda\": %llu, \"prepared_on_device\": %llu, \"rows_on_cuda\": %llu},\n"
            "  \"launches\": {\"rows_on_cuda\": %llu, \"cuda_stream\": %llu, \"admitted_exact\": %llu, "
            "\"retained_exact\": %llu, \"head\": %llu, \"distinct_row_pointers\": %llu},\n"
            "  \"wait_op_calls\": %llu,\n  \"destroy_op_after_cuda\": %llu,\n",
            (unsigned long long)g_publish_cuda, (unsigned long long)g_publish_on_device,
            (unsigned long long)g_publish_rows_cuda, (unsigned long long)g_launch_cuda_rows,
            (unsigned long long)g_launch_stream, (unsigned long long)g_launch_admitted,
            (unsigned long long)g_launch_retained, (unsigned long long)g_launch_head,
            (unsigned long long)g_launch_pointer_count, (unsigned long long)g_wait_calls,
            (unsigned long long)g_destroy_after_cuda);
    /* CUDA traffic between the first and the last launch: the loop itself. */
    fprintf(out, "  \"cuda_total\": {");
    for (int i = 0; i < CU_COUNT; ++i)
    {
        fprintf(out, "%s\"%s\": %llu", i ? ", " : "", kCuNames[i], (unsigned long long)g_cu_total[i]);
    }
    fprintf(out, "},\n  \"cuda_in_loop\": {");
    for (int i = 0; i < CU_COUNT; ++i)
    {
        fprintf(out, "%s\"%s\": %llu", i ? ", " : "", kCuNames[i],
                (unsigned long long)(g_cu_at_last_launch[i] - g_cu_at_first_launch[i]));
    }
    uint64_t event_newest = 0, event_back_pressure = 0;
    for (size_t i = 0; i < g_event_sync_count; ++i)
    {
        const uint64_t now = g_event_sync_now[i];
        if (now == 0 || now >= g_enqueued[KIND_LAUNCH])
        {
            continue; /* before the loop or while draining it */
        }
        if (g_event_sync_launches[i] >= now)
        {
            ++event_newest;
        }
        else
        {
            ++event_back_pressure;
        }
    }
    fprintf(out, "},\n  \"event_synchronizations_in_loop\": {\"newest\": %llu, \"back_pressure\": %llu",
            (unsigned long long)event_newest, (unsigned long long)event_back_pressure);
    fprintf(out, "},\n  \"cuda_event_elapsed_time_calls\": %llu", (unsigned long long)g_elapsed_time_calls);
    /* Admission times relative to the first launch; verify.py takes the span of the last
     * `steps` of them, so a warm-up launch before the loop does not stretch it. */
    const uint64_t timed = g_enqueued[KIND_LAUNCH] < MAX_OPS ? g_enqueued[KIND_LAUNCH] : MAX_OPS;
    fprintf(out, ",\n  \"launch_seconds\": [");
    for (uint64_t i = 0; i < timed; ++i)
    {
        fprintf(out, "%s%.6f", i ? ", " : "", g_launch_s[i] - g_launch_s[0]);
    }
    fprintf(out, "]");
    /* Measurements that cover launches: the span containing the most launches (the loop's,
     * when the program timed the loop in one span), and the per-span sum when the spans are
     * distinct launch intervals that do not overlap, with the launches they cover. A span read
     * twice counts once; overlapping spans give no sum. Setup or teardown spans contain no
     * launch. verify.py compares both with the task's step count. */
    double                  widest_ms = -1.0, covering_ms = 0.0;
    uint64_t                widest = 0, covered = 0;
    static elapsed_record_t spans[MAX_EVENTS];
    size_t                  span_count = 0;
    for (size_t i = 0; i < g_elapsed_count; ++i)
    {
        const elapsed_record_t* m = &g_elapsed[i];
        if (m->end_launches <= m->start_launches)
        {
            continue;
        }
        const uint64_t launches = m->end_launches - m->start_launches;
        if (launches > widest || (launches == widest && m->ms > widest_ms))
        {
            widest    = launches;
            widest_ms = m->ms;
        }
        size_t j = 0;
        while (j < span_count &&
               (spans[j].start_launches != m->start_launches || spans[j].end_launches != m->end_launches))
        {
            ++j;
        }
        if (j == span_count)
        {
            /* insertion by start, so overlaps are adjacent */
            size_t at = span_count++;
            while (at > 0 && spans[at - 1].start_launches > m->start_launches)
            {
                spans[at] = spans[at - 1];
                --at;
            }
            spans[at] = *m;
        }
    }
    int overlapping = 0;
    for (size_t i = 0; i < span_count; ++i)
    {
        overlapping |= i > 0 && spans[i].start_launches < spans[i - 1].end_launches;
        covering_ms += spans[i].ms;
        covered += spans[i].end_launches - spans[i].start_launches;
    }
    if (overlapping)
    {
        covering_ms = 0.0;
        covered     = 0;
    }
    fprintf(out,
            ",\n  \"cuda_event_elapsed_ms\": {\"widest\": %.6f, \"widest_launches\": %llu, \"covering_sum\": %.6f, "
            "\"covered_launches\": %llu}",
            widest_ms, (unsigned long long)widest, covering_ms, (unsigned long long)covered);
    fprintf(out, "\n}\n");
    pthread_mutex_unlock(&g_lock);
    fclose(out);
}
