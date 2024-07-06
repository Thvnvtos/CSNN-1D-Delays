# CSNN-1D-Delays
Adding delays to 1D convolutional SNNs using DCLS2-1D

### Project progress:
- Completed CSNN-1D simple training pipeline.
- Started CSNN-1D-Delays

### Current work and next milestone:
- Training and testing current code with CSNN-1D
- Debug CSNN-1D (I think final FC is not registered correctly)
- Working on CSNN-1D-Delays (Add parameters init, Sig decreasing, specific things in model.py, etc...)


### Notes:

- Test using GAvgP+FClayer vs only GAvgP
- Maybe Test Conv strides for CSNN-1D-Delays
- Batchnorm stats calculated on all timesteps, is it a bad thing ?
