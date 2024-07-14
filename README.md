# CSNN-1D-Delays
Adding delays to 1D convolutional SNNs using DCLS2-1D

### Project progress:
- Completed CSNN-1D-Delays simple Pipeline

### Current work and next milestone:
- Training and testing current code with CSNN-1D and CSNN-1D-Delays
- Make sure everything is working well, write some basic tests.

- Add Learning Rate Schedulers
- Add simple Wandb logs
- Add depthwise separable convs, either new model or as a configurable part of CSNN-1D-Delays


### Notes:

- Test using GAvgP+FClayer vs only GAvgP
- Maybe Test Conv strides for CSNN-1D-Delays
- Batchnorm stats calculated on all timesteps, is it a bad thing ?
- Unit test reset_model() (is functional.reset_net working well ?)
