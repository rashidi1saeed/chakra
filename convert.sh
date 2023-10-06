#!/bin/bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python3 -m et_converter.et_converter    --input_type PyTorch    --input_filename eg_files/dlrm_eg_0_plus.json    --output_filename eg_files/dlrm_chakra.0.et    --num_dims 1 

python3 -m et_converter.et_converter    --input_type PyTorch    --input_filename eg_files/dlrm_eg_1_plus.json    --output_filename eg_files/dlrm_chakra.1.et    --num_dims 1 

python3 -m et_converter.et_converter    --input_type PyTorch    --input_filename eg_files/dlrm_eg_2_plus.json    --output_filename eg_files/dlrm_chakra.2.et    --num_dims 1 

python3 -m et_converter.et_converter    --input_type PyTorch    --input_filename eg_files/dlrm_eg_3_plus.json    --output_filename eg_files/dlrm_chakra.3.et    --num_dims 1 

python3 -m et_converter.et_converter    --input_type PyTorch    --input_filename eg_files/dlrm_eg_4_plus.json    --output_filename eg_files/dlrm_chakra.4.et    --num_dims 1 

python3 -m et_converter.et_converter    --input_type PyTorch    --input_filename eg_files/dlrm_eg_5_plus.json    --output_filename eg_files/dlrm_chakra.5.et    --num_dims 1 

python3 -m et_converter.et_converter    --input_type PyTorch    --input_filename eg_files/dlrm_eg_6_plus.json    --output_filename eg_files/dlrm_chakra.6.et    --num_dims 1 

python3 -m et_converter.et_converter    --input_type PyTorch    --input_filename eg_files/dlrm_eg_7_plus.json    --output_filename eg_files/dlrm_chakra.7.et    --num_dims 1 
