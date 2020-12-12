# torchprof

[![PyPI version](https://badge.fury.io/py/torchprof.svg)](https://pypi.org/project/torchprof/)
[![CircleCI](https://circleci.com/gh/awwong1/torchprof.svg?style=svg)](https://circleci.com/gh/awwong1/torchprof)

A minimal dependency library for layer-by-layer profiling of PyTorch models.

All metrics are derived using the PyTorch autograd profiler.

## Quickstart

`pip install torchprof`

```python
import torch
import torchvision
import torchprof

model = torchvision.models.alexnet(pretrained=False).cuda()
x = torch.rand([1, 3, 224, 224]).cuda()

# `profile_memory` was added in PyTorch 1.6, this will output a runtime warning if unsupported.
with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
    model(x)

# equivalent to `print(prof)` and `print(prof.display())`
print(prof.display(show_events=False))
```
```text
Module         | Self CPU total | CPU total | Self CUDA total | CUDA total | Self CPU Mem | CPU Mem | Self CUDA Mem | CUDA Mem  | Number of Calls
---------------|----------------|-----------|-----------------|------------|--------------|---------|---------------|-----------|----------------
AlexNet        |                |           |                 |            |              |         |               |           |
├── features   |                |           |                 |            |              |         |               |           |
│├── 0         | 1.831ms        | 7.260ms   | 1.830ms         | 7.230ms    | 0 b          | 0 b     | 3.71 Mb       | 756.50 Kb | 1
│├── 1         | 46.768us       | 68.950us  | 46.976us        | 70.528us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 2         | 80.361us       | 166.213us | 79.872us        | 149.696us  | 0 b          | 0 b     | 1.60 Mb       | 547.00 Kb | 1
│├── 3         | 277.412us      | 1.205ms   | 492.544us       | 1.932ms    | 0 b          | 0 b     | 2.68 Mb       | 547.00 Kb | 1
│├── 4         | 28.274us       | 40.156us  | 27.872us        | 41.184us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 5         | 57.138us       | 124.176us | 56.512us        | 109.536us  | 0 b          | 0 b     | 1.11 Mb       | 380.50 Kb | 1
│├── 6         | 173.517us      | 674.434us | 210.880us       | 809.824us  | 0 b          | 0 b     | 8.27 Mb       | 253.50 Kb | 1
│├── 7         | 27.382us       | 38.754us  | 27.648us        | 39.936us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 8         | 144.863us      | 556.345us | 207.872us       | 798.368us  | 0 b          | 0 b     | 10.20 Mb      | 169.00 Kb | 1
│├── 9         | 27.552us       | 39.224us  | 26.752us        | 39.072us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 10        | 138.752us      | 531.703us | 173.056us       | 661.568us  | 0 b          | 0 b     | 7.08 Mb       | 169.00 Kb | 1
│├── 11        | 27.743us       | 39.515us  | 27.648us        | 39.936us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│└── 12        | 60.333us       | 133.099us | 59.392us        | 116.768us  | 0 b          | 0 b     | 324.00 Kb     | 108.00 Kb | 1
├── avgpool    | 55.655us       | 110.770us | 57.344us        | 107.456us  | 0 b          | 0 b     | 108.00 Kb     | 36.00 Kb  | 1
└── classifier |                |           |                 |            |              |         |               |           |
 ├── 0         | 77.746us       | 165.089us | 77.696us        | 144.064us  | 0 b          | 0 b     | 171.00 Kb     | 45.00 Kb  | 1
 ├── 1         | 405.262us      | 425.012us | 796.672us       | 796.672us  | 0 b          | 0 b     | 32.00 Kb      | 16.00 Kb  | 1
 ├── 2         | 29.455us       | 42.329us  | 29.472us        | 42.976us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
 ├── 3         | 53.601us       | 120.870us | 53.248us        | 99.328us   | 0 b          | 0 b     | 76.00 Kb      | 20.00 Kb  | 1
 ├── 4         | 63.981us       | 79.811us  | 232.448us       | 232.448us  | 0 b          | 0 b     | 32.00 Kb      | 16.00 Kb  | 1
 ├── 5         | 27.853us       | 39.445us  | 27.648us        | 40.928us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
 └── 6         | 61.656us       | 76.714us  | 96.256us        | 96.256us   | 0 b          | 0 b     | 8.00 Kb       | 4.00 Kb   | 1
```

To see the low level operations that occur within each layer, print the contents of  `prof.display(show_events=True)`.

```text
Module                              | Self CPU total | CPU total | Self CUDA total | CUDA total | Self CPU Mem | CPU Mem | Self CUDA Mem | CUDA Mem  | Number of Calls
------------------------------------|----------------|-----------|-----------------|------------|--------------|---------|---------------|-----------|----------------
AlexNet                             |                |           |                 |            |              |         |               |           |
├── features                        |                |           |                 |            |              |         |               |           |
│├── 0                              |                |           |                 |            |              |         |               |           |
││├── aten::conv2d                  | 15.779us       | 1.831ms   | 14.336us        | 1.830ms    | 0 b          | 0 b     | 756.50 Kb     | 0 b       | 1
││├── aten::convolution             | 10.139us       | 1.815ms   | 8.512us         | 1.816ms    | 0 b          | 0 b     | 756.50 Kb     | 0 b       | 1
││├── aten::_convolution            | 45.115us       | 1.805ms   | 36.288us        | 1.808ms    | 0 b          | 0 b     | 756.50 Kb     | 0 b       | 1
││├── aten::contiguous              | 8.586us        | 8.586us   | 8.160us         | 8.160us    | 0 b          | 0 b     | 0 b           | 0 b       | 3
││├── aten::cudnn_convolution       | 1.646ms        | 1.682ms   | 1.745ms         | 1.749ms    | 0 b          | 0 b     | 756.50 Kb     | -18.00 Kb | 1
││├── aten::empty                   | 21.821us       | 21.821us  | 0.000us         | 0.000us    | 0 b          | 0 b     | 774.50 Kb     | 774.50 Kb | 2
││├── aten::resize_                 | 7.324us        | 7.324us   | 0.000us         | 0.000us    | 0 b          | 0 b     | 0 b           | 0 b       | 2
││├── aten::stride                  | 2.073us        | 2.073us   | 0.000us         | 0.000us    | 0 b          | 0 b     | 0 b           | 0 b       | 4
││├── aten::reshape                 | 5.701us        | 17.603us  | 1.056us         | 1.056us    | 0 b          | 0 b     | 0 b           | 0 b       | 1
││├── aten::view                    | 11.902us       | 11.902us  | 0.000us         | 0.000us    | 0 b          | 0 b     | 0 b           | 0 b       | 1
││└── aten::add_                    | 56.837us       | 56.837us  | 17.408us        | 17.408us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 1                              |                |           |                 |            |              |         |               |           |
││├── aten::relu_                   | 24.586us       | 46.768us  | 23.424us        | 46.976us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
││└── aten::threshold_              | 22.182us       | 22.182us  | 23.552us        | 23.552us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 2                              |                |           |                 |            |              |         |               |           |
...
```


The original [Pytorch EventList](https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile) can be returned by calling `raw()` on the profile instance.

```python
trace, event_lists_dict = prof.raw()
print(trace[2])
# Trace(path=('AlexNet', 'features', '0'), leaf=True, module=Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)))

print(event_lists_dict[trace[2].path][0])
```
```text
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
               aten::conv2d         0.86%      15.779us       100.00%       1.831ms       1.831ms      14.336us         0.78%       1.830ms       1.830ms           0 b           0 b     756.50 Kb           0 b             1  
          aten::convolution         0.55%      10.139us        99.14%       1.815ms       1.815ms       8.512us         0.47%       1.816ms       1.816ms           0 b           0 b     756.50 Kb           0 b             1  
         aten::_convolution         2.46%      45.115us        98.58%       1.805ms       1.805ms      36.288us         1.98%       1.808ms       1.808ms           0 b           0 b     756.50 Kb           0 b             1  
           aten::contiguous         0.20%       3.697us         0.20%       3.697us       3.697us       3.616us         0.20%       3.616us       3.616us           0 b           0 b           0 b           0 b             1  
    aten::cudnn_convolution        89.88%       1.646ms        91.85%       1.682ms       1.682ms       1.745ms        95.31%       1.749ms       1.749ms           0 b           0 b     756.50 Kb     -18.00 Kb             1  
                aten::empty         0.67%      12.313us         0.67%      12.313us      12.313us       0.000us         0.00%       0.000us       0.000us           0 b           0 b     756.50 Kb     756.50 Kb             1  
           aten::contiguous         0.14%       2.575us         0.14%       2.575us       2.575us       2.464us         0.13%       2.464us       2.464us           0 b           0 b           0 b           0 b             1  
              aten::resize_         0.37%       6.843us         0.37%       6.843us       6.843us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
           aten::contiguous         0.13%       2.314us         0.13%       2.314us       2.314us       2.080us         0.11%       2.080us       2.080us           0 b           0 b           0 b           0 b             1  
              aten::resize_         0.03%       0.481us         0.03%       0.481us       0.481us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
               aten::stride         0.07%       1.203us         0.07%       1.203us       1.203us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
               aten::stride         0.02%       0.300us         0.02%       0.300us       0.300us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
               aten::stride         0.02%       0.290us         0.02%       0.290us       0.290us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
               aten::stride         0.02%       0.280us         0.02%       0.280us       0.280us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
                aten::empty         0.52%       9.508us         0.52%       9.508us       9.508us       0.000us         0.00%       0.000us       0.000us           0 b           0 b      18.00 Kb      18.00 Kb             1  
              aten::reshape         0.31%       5.701us         0.96%      17.603us      17.603us       1.056us         0.06%       1.056us       1.056us           0 b           0 b           0 b           0 b             1  
                 aten::view         0.65%      11.902us         0.65%      11.902us      11.902us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
                 aten::add_         3.10%      56.837us         3.10%      56.837us      56.837us      17.408us         0.95%      17.408us      17.408us           0 b           0 b           0 b           0 b             1  
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.831ms
CUDA time total: 1.830ms

```

Layers can be selected for individually using the optional `paths` kwarg. Profiling is ignored for all other layers.

```python
model = torchvision.models.alexnet(pretrained=False)
x = torch.rand([1, 3, 224, 224])

# Layer does not have to be a leaf layer
paths = [("AlexNet", "features", "3"), ("AlexNet", "classifier")]

with torchprof.Profile(model, paths=paths) as prof:
    model(x)

print(prof)
```

```text
Module         | Self CPU total | CPU total | Number of Calls
---------------|----------------|-----------|----------------
AlexNet        |                |           |
├── features   |                |           |
│├── 0         |                |           |
│├── 1         |                |           |
│├── 2         |                |           |
│├── 3         | 2.079ms        | 8.296ms   | 1
│├── 4         |                |           |
│├── 5         |                |           |
│├── 6         |                |           |
│├── 7         |                |           |
│├── 8         |                |           |
│├── 9         |                |           |
│├── 10        |                |           |
│├── 11        |                |           |
│└── 12        |                |           |
├── avgpool    |                |           |
└── classifier | 10.734ms       | 11.282ms  | 1
 ├── 0         |                |           |
 ├── 1         |                |           |
 ├── 2         |                |           |
 ├── 3         |                |           |
 ├── 4         |                |           |
 ├── 5         |                |           |
 └── 6         |                |           |

```

* [Self CPU Time vs CPU Time](https://software.intel.com/en-us/vtune-amplifier-help-self-time-and-total-time)

## Citation

If this software is useful to your research, I would greatly appreciate a citation in your work.

```tex
@misc{awwong1-torchprof,
  title        = {torchprof},
  author       = {Alexander William Wong},
  month        = 12,
  year         = 2020,
  url          = {https://github.com/awwong1/torchprof}
  note         = {https://github.com/awwong1/torchprof}
}
```

## LICENSE
[MIT](LICENSE)
