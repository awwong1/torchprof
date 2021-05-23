# torchprof

[![PyPI version](https://badge.fury.io/py/torchprof.svg)](https://pypi.org/project/torchprof/)
[![CircleCI](https://circleci.com/gh/awwong1/torchprof.svg?style=svg)](https://circleci.com/gh/awwong1/torchprof)

> Attention! [This library is deprecated due to the PyTorch 1.9 changes to the torch profiler. Please use the official profiler.](https://pytorch.org/docs/1.9.0/profiler.html?highlight=profiler#module-torch.profiler) Thank you!

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
│├── 0         | 1.832ms        | 7.264ms   | 1.831ms         | 7.235ms    | 0 b          | 0 b     | 756.50 Kb     | 3.71 Mb   | 1
│├── 1         | 51.858us       | 76.564us  | 51.296us        | 76.896us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 2         | 75.993us       | 157.855us | 77.600us        | 145.184us  | 0 b          | 0 b     | 547.00 Kb     | 1.60 Mb   | 1
│├── 3         | 263.526us      | 1.142ms   | 489.472us       | 1.918ms    | 0 b          | 0 b     | 547.00 Kb     | 2.68 Mb   | 1
│├── 4         | 28.824us       | 41.197us  | 28.672us        | 43.008us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 5         | 55.264us       | 120.016us | 55.200us        | 106.400us  | 0 b          | 0 b     | 380.50 Kb     | 1.11 Mb   | 1
│├── 6         | 175.591us      | 681.011us | 212.896us       | 818.080us  | 0 b          | 0 b     | 253.50 Kb     | 8.27 Mb   | 1
│├── 7         | 27.622us       | 39.494us  | 26.848us        | 39.296us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 8         | 140.204us      | 537.162us | 204.832us       | 781.280us  | 0 b          | 0 b     | 169.00 Kb     | 10.20 Mb  | 1
│├── 9         | 27.532us       | 39.364us  | 26.816us        | 39.136us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 10        | 138.621us      | 530.929us | 171.008us       | 650.432us  | 0 b          | 0 b     | 169.00 Kb     | 7.08 Mb   | 1
│├── 11        | 27.712us       | 39.645us  | 27.648us        | 39.936us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│└── 12        | 54.813us       | 118.823us | 55.296us        | 107.360us  | 0 b          | 0 b     | 108.00 Kb     | 324.00 Kb | 1
├── avgpool    | 58.329us       | 116.577us | 58.368us        | 111.584us  | 0 b          | 0 b     | 36.00 Kb      | 108.00 Kb | 1
└── classifier |                |           |                 |            |              |         |               |           |
 ├── 0         | 79.169us       | 167.495us | 78.848us        | 145.408us  | 0 b          | 0 b     | 45.00 Kb      | 171.00 Kb | 1
 ├── 1         | 404.070us      | 423.755us | 793.600us       | 793.600us  | 0 b          | 0 b     | 16.00 Kb      | 32.00 Kb  | 1
 ├── 2         | 30.097us       | 43.512us  | 29.792us        | 43.904us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
 ├── 3         | 53.390us       | 121.042us | 53.248us        | 99.328us   | 0 b          | 0 b     | 20.00 Kb      | 76.00 Kb  | 1
 ├── 4         | 64.622us       | 79.902us  | 236.544us       | 236.544us  | 0 b          | 0 b     | 16.00 Kb      | 32.00 Kb  | 1
 ├── 5         | 28.854us       | 41.067us  | 28.544us        | 41.856us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
 └── 6         | 62.258us       | 77.356us  | 95.232us        | 95.232us   | 0 b          | 0 b     | 4.00 Kb       | 8.00 Kb   | 1
```

To see the low level operations that occur within each layer, print the contents of  `prof.display(show_events=True)`.

```text
Module                              | Self CPU total | CPU total | Self CUDA total | CUDA total | Self CPU Mem | CPU Mem | Self CUDA Mem | CUDA Mem  | Number of Calls
------------------------------------|----------------|-----------|-----------------|------------|--------------|---------|---------------|-----------|----------------
AlexNet                             |                |           |                 |            |              |         |               |           |
├── features                        |                |           |                 |            |              |         |               |           |
│├── 0                              |                |           |                 |            |              |         |               |           |
││├── aten::conv2d                  | 15.630us       | 1.832ms   | 14.176us        | 1.831ms    | 0 b          | 0 b     | 0 b           | 756.50 Kb | 1
││├── aten::convolution             | 9.768us        | 1.816ms   | 9.056us         | 1.817ms    | 0 b          | 0 b     | 0 b           | 756.50 Kb | 1
││├── aten::_convolution            | 45.005us       | 1.807ms   | 34.432us        | 1.808ms    | 0 b          | 0 b     | 0 b           | 756.50 Kb | 1
││├── aten::contiguous              | 8.738us        | 8.738us   | 8.480us         | 8.480us    | 0 b          | 0 b     | 0 b           | 0 b       | 3
││├── aten::cudnn_convolution       | 1.647ms        | 1.683ms   | 1.745ms         | 1.750ms    | 0 b          | 0 b     | -18.00 Kb     | 756.50 Kb | 1
││├── aten::empty                   | 21.249us       | 21.249us  | 0.000us         | 0.000us    | 0 b          | 0 b     | 774.50 Kb     | 774.50 Kb | 2
││├── aten::resize_                 | 7.635us        | 7.635us   | 0.000us         | 0.000us    | 0 b          | 0 b     | 0 b           | 0 b       | 2
││├── aten::stride                  | 1.902us        | 1.902us   | 0.000us         | 0.000us    | 0 b          | 0 b     | 0 b           | 0 b       | 4
││├── aten::reshape                 | 6.081us        | 17.833us  | 2.048us         | 2.048us    | 0 b          | 0 b     | 0 b           | 0 b       | 1
││├── aten::view                    | 11.752us       | 11.752us  | 0.000us         | 0.000us    | 0 b          | 0 b     | 0 b           | 0 b       | 1
││└── aten::add_                    | 57.248us       | 57.248us  | 18.432us        | 18.432us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
│├── 1                              |                |           |                 |            |              |         |               |           |
││├── aten::relu_                   | 27.152us       | 51.858us  | 25.696us        | 51.296us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
││└── aten::threshold_              | 24.706us       | 24.706us  | 25.600us        | 25.600us   | 0 b          | 0 b     | 0 b           | 0 b       | 1
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
               aten::conv2d         0.85%      15.630us       100.00%       1.832ms       1.832ms      14.176us         0.77%       1.831ms       1.831ms           0 b           0 b     756.50 Kb           0 b             1
          aten::convolution         0.53%       9.768us        99.15%       1.816ms       1.816ms       9.056us         0.49%       1.817ms       1.817ms           0 b           0 b     756.50 Kb           0 b             1
         aten::_convolution         2.46%      45.005us        98.61%       1.807ms       1.807ms      34.432us         1.88%       1.808ms       1.808ms           0 b           0 b     756.50 Kb           0 b             1
           aten::contiguous         0.20%       3.707us         0.20%       3.707us       3.707us       3.680us         0.20%       3.680us       3.680us           0 b           0 b           0 b           0 b             1
    aten::cudnn_convolution        89.90%       1.647ms        91.86%       1.683ms       1.683ms       1.745ms        95.27%       1.750ms       1.750ms           0 b           0 b     756.50 Kb     -18.00 Kb             1
                aten::empty         0.66%      12.102us         0.66%      12.102us      12.102us       0.000us         0.00%       0.000us       0.000us           0 b           0 b     756.50 Kb     756.50 Kb             1
           aten::contiguous         0.15%       2.706us         0.15%       2.706us       2.706us       2.560us         0.14%       2.560us       2.560us           0 b           0 b           0 b           0 b             1
              aten::resize_         0.39%       7.164us         0.39%       7.164us       7.164us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
           aten::contiguous         0.13%       2.325us         0.13%       2.325us       2.325us       2.240us         0.12%       2.240us       2.240us           0 b           0 b           0 b           0 b             1
              aten::resize_         0.03%       0.471us         0.03%       0.471us       0.471us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
               aten::stride         0.06%       1.092us         0.06%       1.092us       1.092us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
               aten::stride         0.02%       0.280us         0.02%       0.280us       0.280us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
               aten::stride         0.01%       0.270us         0.01%       0.270us       0.270us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
               aten::stride         0.01%       0.260us         0.01%       0.260us       0.260us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
                aten::empty         0.50%       9.147us         0.50%       9.147us       9.147us       0.000us         0.00%       0.000us       0.000us           0 b           0 b      18.00 Kb      18.00 Kb             1
              aten::reshape         0.33%       6.081us         0.97%      17.833us      17.833us       2.048us         0.11%       2.048us       2.048us           0 b           0 b           0 b           0 b             1
                 aten::view         0.64%      11.752us         0.64%      11.752us      11.752us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
                 aten::add_         3.12%      57.248us         3.12%      57.248us      57.248us      18.432us         1.01%      18.432us      18.432us           0 b           0 b           0 b           0 b             1
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.832ms
CUDA time total: 1.831ms

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
│├── 3         | 3.162ms        | 12.626ms  | 1
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
└── classifier | 11.398ms       | 12.130ms  | 1
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
