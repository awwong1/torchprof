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
│├── 0         | 1.808ms        | 7.171ms   | 1.807ms         | 7.133ms    | 0 b          | 0 b     | 3.71 Mb       | 756.50 Kb | 1
│├── 1         | 49.693us       | 72.366us  | 49.152us        | 72.608us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 2         | 78.267us       | 162.737us | 77.824us        | 147.232us  | 0 b          | 0 b     | 1.60 Mb       | 547.00 Kb | 1
│├── 3         | 281.690us      | 1.226ms   | 506.880us       | 1.992ms    | 0 b          | 0 b     | 2.68 Mb       | 547.00 Kb | 1
│├── 4         | 29.124us       | 41.487us  | 29.632us        | 43.968us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 5         | 56.055us       | 121.788us | 55.296us        | 108.544us  | 0 b          | 0 b     | 1.11 Mb       | 380.50 Kb | 1
│├── 6         | 175.320us      | 678.494us | 213.856us       | 818.016us  | 0 b          | 0 b     | 8.27 Mb       | 253.50 Kb | 1
│├── 7         | 28.434us       | 40.487us  | 28.672us        | 40.960us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 8         | 147.237us      | 564.774us | 209.920us       | 801.984us  | 0 b          | 0 b     | 10.20 Mb      | 169.00 Kb | 1
│├── 9         | 28.043us       | 40.005us  | 27.648us        | 40.928us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 10        | 141.357us      | 541.427us | 177.152us       | 671.552us  | 0 b          | 0 b     | 7.08 Mb       | 169.00 Kb | 1
│├── 11        | 28.503us       | 40.405us  | 28.672us        | 41.152us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│└── 12        | 55.224us       | 119.865us | 55.296us        | 106.880us  | 0 b          | 0 b     | 324.00 Kb     | 108.00 Kb | 1
├── avgpool    | 55.585us       | 110.217us | 57.344us        | 106.464us  | 0 b          | 0 b     | 108.00 Kb     | 36.00 Kb  | 1
└── classifier |                |           |                 |            |              |         |               |           |
 ├── 0         | 78.037us       | 165.510us | 76.896us        | 142.432us  | 0 b          | 0 b     | 171.00 Kb     | 45.00 Kb  | 1
 ├── 1         | 399.993us      | 419.901us | 795.648us       | 795.648us  | 0 b          | 0 b     | 32.00 Kb      | 16.00 Kb  | 1
 ├── 2         | 29.937us       | 43.122us  | 29.664us        | 42.944us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
 ├── 3         | 53.331us       | 120.781us | 52.384us        | 99.488us   | 0 b          | 0 b     | 76.00 Kb      | 20.00 Kb  | 1
 ├── 4         | 64.231us       | 79.479us  | 232.448us       | 232.448us  | 0 b          | 0 b     | 32.00 Kb      | 16.00 Kb  | 1
 ├── 5         | 29.045us       | 41.238us  | 29.664us        | 41.952us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
 └── 6         | 63.289us       | 78.356us  | 97.280us        | 97.280us   | 0 b          | 0 b     | 8.00 Kb       | 4.00 Kb   | 1
```

To see the low level operations that occur within each layer, print the contents of  `prof.display(show_events=True)`.

```text
Module                              | Self CPU total | CPU total | Self CUDA total | CUDA total | Self CPU Mem | CPU Mem | Self CUDA Mem | CUDA Mem  | Number of Calls
------------------------------------|----------------|-----------|-----------------|------------|--------------|---------|---------------|-----------|----------------
AlexNet                             |                |           |                 |            |              |         |               |           |
├── features                        |                |           |                 |            |              |         |               |           |
│├── 0                              |                |           |                 |            |              |         |               |           |
││├── aten::conv2d                  | 16.481us       | 1.808ms   | 14.368us        | 1.807ms    | 0 b          | 0 b     | 756.50 Kb     | 0 b       | 1
││├── aten::convolution             | 10.450us       | 1.792ms   | 10.880us        | 1.792ms    | 0 b          | 0 b     | 756.50 Kb     | 0 b       | 1
││├── aten::_convolution            | 41.480us       | 1.781ms   | 34.240us        | 1.782ms    | 0 b          | 0 b     | 756.50 Kb     | 0 b       | 1
││├── aten::contiguous              | 2.514us        | 2.514us   | 2.304us         | 2.304us    | 0 b          | 0 b     | 0 b           | 0 b       | 1
││├── aten::cudnn_convolution       | 1.619ms        | 1.657ms   | 1.718ms         | 1.723ms    | 0 b          | 0 b     | 756.50 Kb     | -18.00 Kb | 1
││├── aten::empty                   | 9.859us        | 9.859us   | 0.000us         | 0.000us    | 0 b          | 0 b     | 18.00 Kb      | 18.00 Kb  | 1
││├── aten::resize_                 | 0.410us        | 0.410us   | 0.000us         | 0.000us    | 0 b          | 0 b     | 0 b           | 0 b       | 1
││├── aten::stride                  | 1.773us        | 1.773us   | 0.000us         | 0.000us    | 0 b          | 0 b     | 0 b           | 0 b       | 4
││├── aten::reshape                 | 6.101us        | 17.853us  | 1.024us         | 1.024us    | 0 b          | 0 b     | 0 b           | 0 b       | 1
││├── aten::view                    | 11.752us       | 11.752us  | 0.000us         | 0.000us    | 0 b          | 0 b     | 0 b           | 0 b       | 1
││└── aten::add_                    | 61.024us       | 61.024us  | 19.456us        | 19.456us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 1                              |                |           |                 |            |              |         |               |           |
││├── aten::relu_                   | 27.020us       | 49.693us  | 25.696us        | 49.152us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
││└── aten::threshold_              | 22.673us       | 22.673us  | 23.456us        | 23.456us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
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
               aten::conv2d         0.91%      16.481us       100.00%       1.808ms       1.808ms      14.368us         0.80%       1.807ms       1.807ms           0 b           0 b     756.50 Kb           0 b             1  
          aten::convolution         0.58%      10.450us        99.09%       1.792ms       1.792ms      10.880us         0.60%       1.792ms       1.792ms           0 b           0 b     756.50 Kb           0 b             1  
         aten::_convolution         2.29%      41.480us        98.51%       1.781ms       1.781ms      34.240us         1.90%       1.782ms       1.782ms           0 b           0 b     756.50 Kb           0 b             1  
           aten::contiguous         0.21%       3.817us         0.21%       3.817us       3.817us       3.680us         0.20%       3.680us       3.680us           0 b           0 b           0 b           0 b             1  
    aten::cudnn_convolution        89.53%       1.619ms        91.64%       1.657ms       1.657ms       1.718ms        95.09%       1.723ms       1.723ms           0 b           0 b     756.50 Kb     -18.00 Kb             1  
                aten::empty         0.73%      13.125us         0.73%      13.125us      13.125us       0.000us         0.00%       0.000us       0.000us           0 b           0 b     756.50 Kb     756.50 Kb             1  
           aten::contiguous         0.15%       2.745us         0.15%       2.745us       2.745us       2.720us         0.15%       2.720us       2.720us           0 b           0 b           0 b           0 b             1  
              aten::resize_         0.43%       7.835us         0.43%       7.835us       7.835us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
           aten::contiguous         0.14%       2.514us         0.14%       2.514us       2.514us       2.304us         0.13%       2.304us       2.304us           0 b           0 b           0 b           0 b             1  
              aten::resize_         0.02%       0.410us         0.02%       0.410us       0.410us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
               aten::stride         0.05%       0.982us         0.05%       0.982us       0.982us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
               aten::stride         0.02%       0.281us         0.02%       0.281us       0.281us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
               aten::stride         0.01%       0.260us         0.01%       0.260us       0.260us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
               aten::stride         0.01%       0.250us         0.01%       0.250us       0.250us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
                aten::empty         0.55%       9.859us         0.55%       9.859us       9.859us       0.000us         0.00%       0.000us       0.000us           0 b           0 b      18.00 Kb      18.00 Kb             1  
              aten::reshape         0.34%       6.101us         0.99%      17.853us      17.853us       1.024us         0.06%       1.024us       1.024us           0 b           0 b           0 b           0 b             1  
                 aten::view         0.65%      11.752us         0.65%      11.752us      11.752us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1  
                 aten::add_         3.37%      61.024us         3.37%      61.024us      61.024us      19.456us         1.08%      19.456us      19.456us           0 b           0 b           0 b           0 b             1  
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.808ms
CUDA time total: 1.807ms

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
