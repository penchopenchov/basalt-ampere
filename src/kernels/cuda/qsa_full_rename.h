// Renames dense_tc.cu's external symbols so this TU's copy never meets the engine's in libkernels.a.
#pragma once
#define dense_tc mq_dense_tc
#define dense_tc_supported mq_dense_tc_supported
#define dense_tc_type_on mq_dense_tc_type_on
#define dense_tc_pair mq_dense_tc_pair
#define dense_tc_pair_supported mq_dense_tc_pair_supported
#define dense_tc_pair_mask mq_dense_tc_pair_mask
#define dense_tc_l2_prefetch mq_dense_tc_l2_prefetch
#define dense_tc_mask mq_dense_tc_mask
