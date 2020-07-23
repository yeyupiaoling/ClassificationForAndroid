
```shell
./opt \
    --model_file=mobilenet_v2/model \
    --param_file=mobilenet_v2/params \
    --optimize_out_type=naive_buffer \
    --optimize_out=mobilenet_v2 \
    --valid_targets=arm opencl \
    --record_tailoring_info=false
```