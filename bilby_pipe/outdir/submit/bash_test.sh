#!/usr/bin/env bash

# test_data0_0-0_generation
# PARENTS 
# CHILDREN test_data0_0-0_analysis_H1L1V1
/home/thibeau.wouters/miniconda3/envs/igwn-py310/bin/bilby_pipe_generation outdir/test_config_complete.ini --label test_data0_0-0_generation --idx 0 --trigger-time 0.0

# test_data0_0-0_analysis_H1L1V1
# PARENTS test_data0_0-0_generation
# CHILDREN test_data0_0-0_analysis_H1L1V1_final_result
/home/thibeau.wouters/miniconda3/envs/igwn-py310/bin/bilby_pipe_analysis outdir/test_config_complete.ini --detectors H1 --detectors L1 --detectors V1 --label test_data0_0-0_analysis_H1L1V1 --data-dump-file outdir/data/test_data0_0-0_generation_data_dump.pickle --sampler dynesty

# test_data0_0-0_analysis_H1L1V1_final_result
# PARENTS test_data0_0-0_analysis_H1L1V1
# CHILDREN 
/home/thibeau.wouters/miniconda3/envs/igwn-py310/bin/bilby_result --result outdir/result/test_data0_0-0_analysis_H1L1V1_result.hdf5 --outdir outdir/final_result --extension hdf5 --max-samples 20000 --lightweight --save

