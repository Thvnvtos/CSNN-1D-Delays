# CSNN-1D-Delays
Adding delays to 1D convolutional SNNs using DCLS2-1D

### Project progress:
- Repository just started, files are mostly empty only showing structure of project.
- Locally, virtual env with necessary modules and datasets are ok.

### Currently work and next steps:
- While testing the dataset module, the shape of examples from SHD are (1,700) ([here](https://github.com/Thvnvtos/DCLS1-2D/blob/main/tests/data_test.ipynb)) instead of (Time, 700). This is mostly due to this [recent change](https://github.com/fangwei123456/spikingjelly/issues/507) in SJ. Will read it and try to fix this.

- Next step would be to have a complete working pipeline for SHD with a basic network, with tests as notebooks in the tests directory.
