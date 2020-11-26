# torchprof

[![PyPI version](https://badge.fury.io/py/torchprof.svg)](https://pypi.org/project/torchprof/)
[![CircleCI](https://circleci.com/gh/awwong1/torchprof.svg?style=svg)](https://circleci.com/gh/awwong1/torchprof)

A minimal dependency library for layer-by-layer profiling of Pytorch models.

All metrics are derived using the PyTorch autograd profiler.

## Quickstart

`pip install torchprof`

```python
import torch
import torchvision
import torchprof

model = torchvision.models.alexnet(pretrained=False).cuda()
x = torch.rand([1, 3, 224, 224]).cuda()

with torchprof.Profile(model, use_cuda=True) as prof:
    model(x)

print(prof.display(show_events=False)) # equivalent to `print(prof)` and `print(prof.display())`
```
```text
Module         | Self CPU total | CPU total | CUDA total | Occurrences
---------------|----------------|-----------|------------|------------
AlexNet        |                |           |            |
├── features   |                |           |            |
│├── 0         |        1.636ms |   6.466ms |    6.447ms |           1
│├── 1         |       61.320us |  92.700us |   94.016us |           1
│├── 2         |       87.680us | 177.270us |  163.744us |           1
│├── 3         |      291.539us |   1.225ms |    1.966ms |           1
│├── 4         |       34.550us |  48.850us |   50.112us |           1
│├── 5         |       63.220us | 131.670us |  121.888us |           1
│├── 6         |      202.009us | 768.135us |  846.048us |           1
│├── 7         |       40.440us |  58.130us |   59.264us |           1
│├── 8         |      183.129us | 690.816us |  854.016us |           1
│├── 9         |       35.580us |  50.360us |   51.200us |           1
│├── 10        |      167.769us | 631.019us |  701.088us |           1
│├── 11        |       34.450us |  48.730us |   50.048us |           1
│└── 12        |       64.509us | 134.508us |  123.040us |           1
├── avgpool    |       67.200us | 131.190us |  122.880us |           1
└── classifier |                |           |            |
 ├── 0         |       82.110us | 172.480us |  150.848us |           1
 ├── 1         |      470.078us | 490.848us |  815.104us |           1
 ├── 2         |       44.269us |  68.289us |   59.424us |           1
 ├── 3         |       59.339us | 125.977us |  109.568us |           1
 ├── 4         |       72.319us |  86.819us |  219.136us |           1
 ├── 5         |       34.780us |  49.340us |   49.152us |           1
 └── 6         |       70.070us |  85.290us |   95.232us |           1
```

To see the low level operations that occur within each layer, print the contents of  `prof.display(show_events=True)`.

```text
Module                              | Self CPU total | CPU total | CUDA total | Occurrences
------------------------------------|----------------|-----------|------------|------------
AlexNet                             |                |           |            |
├── features                        |                |           |            |
│├── 0                              |                |           |            |
││├── aten::conv2d                  |       16.320us |   1.636ms |    1.636ms |           1
││├── aten::convolution             |       11.710us |   1.619ms |    1.620ms |           1
││├── aten::_convolution            |       40.950us |   1.607ms |    1.608ms |           1
││├── aten::contiguous              |        2.920us |   2.920us |    2.720us |           1
││├── aten::cudnn_convolution       |        1.467ms |   1.493ms |    1.554ms |           1
││├── aten::empty                   |        6.160us |   6.160us |    0.000us |           1
││├── aten::resize_                 |        0.490us |   0.490us |    0.000us |           1
││├── aten::stride                  |        2.380us |   2.380us |    0.000us |           4
││├── aten::reshape                 |        6.820us |  18.640us |    2.048us |           1
││├── aten::view                    |       11.820us |  11.820us |    0.000us |           1
││└── aten::add_                    |       51.060us |  51.060us |   18.432us |           1
│├── 1                              |                |           |            |
││├── aten::relu_                   |       29.940us |  61.320us |   61.408us |           1
││└── aten::threshold_              |       31.380us |  31.380us |   32.608us |           1
│├── 2                              |                |           |            |
││├── aten::max_pool2d              |       14.680us |  87.680us |   86.016us |           1
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
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
               aten::conv2d         1.00%      16.320us       100.00%       1.636ms       1.636ms      16.032us         0.98%       1.636ms       1.636ms             1
          aten::convolution         0.72%      11.710us        99.00%       1.619ms       1.619ms      12.064us         0.74%       1.620ms       1.620ms             1
         aten::_convolution         2.50%      40.950us        98.29%       1.607ms       1.607ms      29.088us         1.78%       1.608ms       1.608ms             1
           aten::contiguous         0.25%       4.090us         0.25%       4.090us       4.090us       4.032us         0.25%       4.032us       4.032us             1
    aten::cudnn_convolution        89.71%       1.467ms        91.27%       1.493ms       1.493ms       1.548ms        94.64%       1.554ms       1.554ms             1
                aten::empty         0.28%       4.590us         0.28%       4.590us       4.590us       0.000us         0.00%       0.000us       0.000us             1
           aten::contiguous         0.22%       3.530us         0.22%       3.530us       3.530us       3.200us         0.20%       3.200us       3.200us             1
              aten::resize_         0.33%       5.390us         0.33%       5.390us       5.390us       0.000us         0.00%       0.000us       0.000us             1
           aten::contiguous         0.18%       2.920us         0.18%       2.920us       2.920us       2.720us         0.17%       2.720us       2.720us             1
              aten::resize_         0.03%       0.490us         0.03%       0.490us       0.490us       0.000us         0.00%       0.000us       0.000us             1
               aten::stride         0.09%       1.460us         0.09%       1.460us       1.460us       0.000us         0.00%       0.000us       0.000us             1
               aten::stride         0.02%       0.320us         0.02%       0.320us       0.320us       0.000us         0.00%       0.000us       0.000us             1
               aten::stride         0.02%       0.300us         0.02%       0.300us       0.300us       0.000us         0.00%       0.000us       0.000us             1
               aten::stride         0.02%       0.300us         0.02%       0.300us       0.300us       0.000us         0.00%       0.000us       0.000us             1
                aten::empty         0.38%       6.160us         0.38%       6.160us       6.160us       0.000us         0.00%       0.000us       0.000us             1
              aten::reshape         0.42%       6.820us         1.14%      18.640us      18.640us       2.048us         0.13%       2.048us       2.048us             1
                 aten::view         0.72%      11.820us         0.72%      11.820us      11.820us       0.000us         0.00%       0.000us       0.000us             1
                 aten::add_         3.12%      51.060us         3.12%      51.060us      51.060us      18.432us         1.13%      18.432us      18.432us             1
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.636ms
CUDA time total: 1.636ms

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
Module         | Self CPU total | CPU total | CUDA total | Occurrences
---------------|----------------|-----------|------------|------------
AlexNet        |                |           |            |
├── features   |                |           |            |
│├── 0         |                |           |            |
│├── 1         |                |           |            |
│├── 2         |                |           |            |
│├── 3         |        2.908ms |  11.604ms |    0.000us |           1
│├── 4         |                |           |            |
│├── 5         |                |           |            |
│├── 6         |                |           |            |
│├── 7         |                |           |            |
│├── 8         |                |           |            |
│├── 9         |                |           |            |
│├── 10        |                |           |            |
│├── 11        |                |           |            |
│└── 12        |                |           |            |
├── avgpool    |                |           |            |
└── classifier |       12.311ms |  13.077ms |    0.000us |           1
 ├── 0         |                |           |            |
 ├── 1         |                |           |            |
 ├── 2         |                |           |            |
 ├── 3         |                |           |            |
 ├── 4         |                |           |            |
 ├── 5         |                |           |            |
 └── 6         |                |           |            |

```

* [Self CPU Time vs CPU Time](https://software.intel.com/en-us/vtune-amplifier-help-self-time-and-total-time)

## Citation

If this software is useful to your research, I would greatly appreciate a citation in your work.

```tex
@misc{torchprof,
  author       = {Alexander William Wong}, 
  title        = {torchprof},
  howpublished = {github.com},
  month        = 4,
  year         = 2020,
  note         = {A minimal dependency library for layer-by-layer profiling of Pytorch models.}
}
```

## LICENSE
[MIT](LICENSE)
