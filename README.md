# torchprof

Measure neural network device specific metrics.

Each nested module is run individually.

## Quickstart

`pip install torchprof`

```python
import torch
import torchvision
import torchprof

model = torchvision.models.alexnet(pretrained=False).cuda()
x = torch.rand([1, 3, 224, 224]).cuda()
observer = torchprof.LatencyObserver(model, use_cuda=True)

raw_measurements = observer.measure_latency(x)
print(raw_measurements[:3])
# [(['AlexNet'], (12289.397999999994, 13599.999755859375)), (['AlexNet', 'features'], (2317.1420000000003, 3342.176134109497)), (['AlexNet', 'features', '0'], (900.206, 1124.0960121154785))]

print(observer)
```
```text
Module         |  CPU Time | CUDA Time
---------------|-----------|----------
AlexNet        |  12.289ms |  13.600ms
├── features   |   2.317ms |   3.342ms
│  ├── 0       | 900.206us |   1.124ms
│  ├── 1       |  18.400us |  20.288us
│  ├── 2       |  50.470us |  51.232us
│  ├── 3       | 275.819us |   1.095ms
│  ├── 4       |  17.480us |  18.432us
│  ├── 5       |  49.990us |  51.008us
│  ├── 6       | 294.550us | 468.704us
│  ├── 7       |  17.250us |  18.432us
│  ├── 8       | 286.510us | 538.464us
│  ├── 9       |  17.470us |  18.432us
│  ├── 10      | 291.070us | 430.752us
│  ├── 11      |  17.380us |  17.408us
│  └── 12      |  50.200us |  51.200us
├── avgpool    |  49.610us |  53.248us
└── classifier | 234.320us | 741.600us
   ├── 0       |  52.160us |  52.288us
   ├── 1       |  37.990us | 431.104us
   ├── 2       |  17.130us |  17.408us
   ├── 3       |  50.718us |  52.096us
   ├── 4       |  38.629us | 206.624us
   ├── 5       |  16.810us |  17.408us
   └── 6       |  38.900us |  78.656us
```

The [Pytorch autograd profile](https://pytorch.org/docs/stable/autograd.html#torch.autograd.profiler.profile) can be returned using the `raw_profile=True` keyword argument during measure_latency.

```python
import torch
import torchvision
import torchprof

model = torchvision.models.alexnet(pretrained=False).cuda()
x = torch.rand([1, 3, 224, 224]).cuda()
observer = torchprof.LatencyObserver(model, use_cuda=True)

raw_measurements = observer.measure_latency(x, raw_profile=True)
trace, profile = raw_measurements[2]
print(trace)
# ['AlexNet', 'features', '0']
print(profile)
```
```text
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
Name                   Self CPU total %   Self CPU total      CPU total %        CPU total     CPU time avg     CUDA total %       CUDA total    CUDA time avg  Number of Calls
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
conv2d                          26.12%        224.779us           26.12%        224.779us        224.779us           25.80%        281.568us        281.568us                1
convolution                     25.49%        219.349us           25.49%        219.349us        219.349us           25.33%        276.480us        276.480us                1
_convolution                    24.88%        214.119us           24.88%        214.119us        214.119us           24.86%        271.360us        271.360us                1
contiguous                       0.36%          3.120us            0.36%          3.120us          3.120us            0.28%          3.072us          3.072us                1
cudnn_convolution               23.16%        199.309us           23.16%        199.309us        199.309us           23.73%        259.072us        259.072us                1
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
Self CPU time total: 860.676us
CUDA time total: 1.092ms
```

## LICENSE
[MIT](LICENSE)
