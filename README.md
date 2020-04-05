# torchprof

[![PyPI version](https://badge.fury.io/py/torchprof.svg)](https://pypi.org/project/torchprof/)

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
│├── 0         |        1.671ms |   6.589ms |    6.701ms |           1
│├── 1         |       62.430us |  62.430us |   63.264us |           1
│├── 2         |       62.909us | 109.948us |  112.640us |           1
│├── 3         |      225.389us | 858.376us |    1.814ms |           1
│├── 4         |       18.999us |  18.999us |   19.456us |           1
│├── 5         |       29.560us |  52.720us |   54.272us |           1
│├── 6         |      136.959us | 511.216us |  707.360us |           1
│├── 7         |       18.480us |  18.480us |   18.624us |           1
│├── 8         |       84.380us | 300.700us |  590.688us |           1
│├── 9         |       18.249us |  18.249us |   17.632us |           1
│├── 10        |       81.289us | 289.946us |  470.016us |           1
│├── 11        |       17.850us |  17.850us |   18.432us |           1
│└── 12        |       29.350us |  52.260us |   52.288us |           1
├── avgpool    |       41.840us |  70.840us |   76.832us |           1
└── classifier |                |           |            |
 ├── 0         |       66.400us | 122.110us |  125.920us |           1
 ├── 1         |      293.658us | 293.658us |  664.704us |           1
 ├── 2         |       17.600us |  17.600us |   18.432us |           1
 ├── 3         |       27.920us |  49.030us |   51.168us |           1
 ├── 4         |       40.590us |  40.590us |  208.672us |           1
 ├── 5         |       17.570us |  17.570us |   18.432us |           1
 └── 6         |       40.489us |  40.489us |   81.920us |           1
```

To see the low level operations that occur within each layer, print the contents of  `prof.display(show_events=True)`.

```text
Module                        | Self CPU total | CPU total | CUDA total | Occurrences
------------------------------|----------------|-----------|------------|------------
AlexNet                       |                |           |            |
├── features                  |                |           |            |
│├── 0                        |                |           |            |
││├── conv2d                  |       13.370us |   1.671ms |    1.698ms |           1
││├── convolution             |       12.730us |   1.658ms |    1.685ms |           1
││├── _convolution            |       30.660us |   1.645ms |    1.673ms |           1
││├── contiguous              |        6.970us |   6.970us |    7.136us |           1
││└── cudnn_convolution       |        1.608ms |   1.608ms |    1.638ms |           1
│├── 1                        |                |           |            |
││└── relu_                   |       62.430us |  62.430us |   63.264us |           1
│├── 2                        |                |           |            |
││├── max_pool2d              |       15.870us |  62.909us |   63.488us |           1
││└── max_pool2d_with_indices |       47.039us |  47.039us |   49.152us |           1
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
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------
Name                   Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     CUDA total %     CUDA total       CUDA time avg    Number of Calls  Input Shapes
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------
conv2d                 0.80%            13.370us         100.00%          1.671ms          1.671ms          25.34%           1.698ms          1.698ms          1                []
convolution            0.76%            12.730us         99.20%           1.658ms          1.658ms          25.15%           1.685ms          1.685ms          1                []
_convolution           1.83%            30.660us         98.44%           1.645ms          1.645ms          24.97%           1.673ms          1.673ms          1                []
contiguous             0.42%            6.970us          0.42%            6.970us          6.970us          0.11%            7.136us          7.136us          1                []
cudnn_convolution      96.19%           1.608ms          96.19%           1.608ms          1.608ms          24.44%           1.638ms          1.638ms          1                []
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------
Self CPU time total: 1.671ms
CUDA time total: 6.701ms

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
│├── 3         |        3.189ms |  12.717ms |    0.000us |           1
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
└── classifier |       13.403ms |  14.011ms |    0.000us |           1
 ├── 0         |                |           |            |
 ├── 1         |                |           |            |
 ├── 2         |                |           |            |
 ├── 3         |                |           |            |
 ├── 4         |                |           |            |
 ├── 5         |                |           |            |
 └── 6         |                |           |            |

```

* [Self CPU Time vs CPU Time](https://software.intel.com/en-us/vtune-amplifier-help-self-time-and-total-time)

## LICENSE
[MIT](LICENSE)
