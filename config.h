#ifndef MVZK_CONFIG_H__
#define MVZK_CONFIG_H__

// 1. Core system and execution configuration.

#define MVZK_CONFIG_THREADS_NUM 1
#define MVZK_CONFIG_DEFAULT_LEVEL_INFO LEVEL_ERROR
#define MVZK_CONFIG_INSTANT_ABORT true
#define MVZK_CONFIG_NUM_PARTIES 2
#define MVZK_CONFIG_BROADCAST_USING_THREAD_POOL_THRE 10
#define MVZK_CONFIG_BROADCAST_SYNC_DELAY_MILLSEC 50
#define MVZK_CONFIG_BROADCAST_ASSUME_IDEAL false
#define MVZK_CONFIG_NN_MAX_BATCHSIZE 100

// 2. Cryptographic and data representation parameters.

static constexpr char fix_key[] = "\x61\x7e\x8d\xa2\xa0\x51\x1e\x96\x5e\x41\xc2\x9b\x15\x3f\xc7\x7a";
#define MVZK_CONFIG_SCALE 10
#define MVZK_CONFIG_DEFAULT_BITLEN 32

// 3. Model and non-linear operator parameters.

#define MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K 3
#define MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION true
#define MVZK_CONFIG_MODEL_USING_INTEGRATED_NON_LINEAR true
#define MVZK_CONFIG_NN_PUBLIC_INPUT true
#define MVZK_CONFIG_NN_PUBLIC_WEIGHT false

// 4. Verification buffers and batching limits.

//#define MVZK_CONFIG_AUTO_CHECK_LIMIT 1<<20
#define MVZK_MULT_CHECK_CNT 1024*1024
#define MVZK_CONFIG_TENSOR_CHECK_CNT 2000
#define MVZK_CONFIG_LUT_PUBLIC_TABLE_SIZE 1 << 20
#define MVZK_CONFIG_MULT_PRODUCT_THRESHOLD 256

//#define MVZK_CONFIG_LUT_REQUEST_BUFFER_THRESHOLD 260000
//#define MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD 260000
#define MVZK_CONFIG_LUT_REQUEST_BUFFER_THRESHOLD 32768
#define MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD 260000

#define MVZK_CONFIG_BENCHMARK_CONSISTENCY_CHUNK_LIMIT 10000000ULL

// 5. Polynomial arithmetic optimizations.

#define MVZK_FFT_WARNING_LIMIT 256
//#define MVZK_CONFIG_FFT_WARNING_ON false
#define MVZK_CONFIG_ENABLE_KARATSUBA_OPT 1       // 1 enables Karatsuba; 0 disables it.
#define MVZK_CONFIG_KARATSUBA_THRESHOLD 64       // Minimum degree for Karatsuba.

// 6. OpenMP scheduling thresholds.

// Enable size-level parallelism when total_elements reaches this threshold.
#define MVZK_OMP_SIZE_THRESHOLD 8192

// Enable degree-level parallelism for small tensors with high polynomial degree.
#define MVZK_OMP_DEGREE_THRESHOLD 64

#define MVZK_CONFIG_OMP_FAST_TREE_PRODUCT_SIZE_THRESHOLD 4

#endif
