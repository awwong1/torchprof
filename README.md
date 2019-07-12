# torchprof

[![PyPI version](https://badge.fury.io/py/torchprof.svg)](https://badge.fury.io/py/torchprof)

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
Module         | Self CPU total | CPU total | CUDA total
---------------|----------------|-----------|-----------
AlexNet        |                |           |           
├── features   |                |           |           
│  ├── 0       |        1.938ms |   7.639ms |    7.696ms
│  ├── 1       |       65.590us |  65.590us |   66.560us
│  ├── 2       |      117.789us | 191.029us |  164.864us
│  ├── 3       |      251.648us | 963.273us |    1.737ms
│  ├── 4       |       18.019us |  18.019us |   19.456us
│  ├── 5       |       30.349us |  53.739us |   54.272us
│  ├── 6       |      130.109us | 482.766us |  645.056us
│  ├── 7       |       17.250us |  17.250us |   18.336us
│  ├── 8       |       83.779us | 297.796us |  538.656us
│  ├── 9       |       16.840us |  16.840us |   17.408us
│  ├── 10      |       85.119us | 301.186us |  441.024us
│  ├── 11      |       16.910us |  16.910us |   17.408us
│  └── 12      |       28.240us |  49.630us |   49.280us
├── avgpool    |       43.489us |  76.088us |   80.896us
└── classifier |                |           |           
  ├── 0        |      626.506us |   1.240ms |    1.362ms
  ├── 1        |      235.148us | 235.148us |  648.192us
  ├── 2        |       18.360us |  18.360us |   19.360us
  ├── 3        |       30.770us |  54.640us |   55.296us
  ├── 4        |       39.189us |  39.189us |  209.920us
  ├── 5        |       16.430us |  16.430us |   17.408us
  └── 6        |       38.270us |  38.270us |   79.648us
```

To see the low level operations that occur within each layer, print the contents of  `prof.display(show_events=True)`.

```text
Module                            | Self CPU total | CPU total | CUDA total
----------------------------------|----------------|-----------|-----------
AlexNet                           |                |           |           
├── features                      |                |           |           
│  ├── 0                          |                |           |           
│  │  ├── conv2d                  |       17.070us |   1.938ms |    1.950ms
│  │  ├── convolution             |       12.240us |   1.921ms |    1.935ms
│  │  ├── _convolution            |       36.129us |   1.908ms |    1.923ms
│  │  ├── contiguous              |        6.820us |   6.820us |    6.688us
│  │  └── cudnn_convolution       |        1.865ms |   1.865ms |    1.882ms
│  ├── 1                          |                |           |           
│  │  └── relu_                   |       65.590us |  65.590us |   66.560us
│  ├── 2                          |                |           |           
│  │  ├── max_pool2d              |       44.549us | 117.789us |   91.136us
│  │  └── max_pool2d_with_indices |       73.240us |  73.240us |   73.728us
│  ├── 3                          |                |           |           

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
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
Name                   Self CPU total %   Self CPU total      CPU total %        CPU total     CPU time avg     CUDA total %       CUDA total    CUDA time avg  Number of Calls
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
conv2d                           0.88%         17.070us          100.00%          1.938ms          1.938ms           25.34%          1.950ms          1.950ms                1
convolution                      0.63%         12.240us           99.12%          1.921ms          1.921ms           25.14%          1.935ms          1.935ms                1
_convolution                     1.86%         36.129us           98.49%          1.908ms          1.908ms           24.99%          1.923ms          1.923ms                1
contiguous                       0.35%          6.820us            0.35%          6.820us          6.820us            0.09%          6.688us          6.688us                1
cudnn_convolution               96.27%          1.865ms           96.27%          1.865ms          1.865ms           24.45%          1.882ms          1.882ms                1
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
Self CPU time total: 1.938ms
CUDA time total: 7.696ms

```

* [Self CPU Time vs CPU Time](https://software.intel.com/en-us/vtune-amplifier-help-self-time-and-total-time)

## LICENSE
[MIT](LICENSE)
