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
# [(['AlexNet'], (3836.7989999999986, 13197.66349029541)), (['AlexNet', 'features'], (3527.07, 14528.191928863525)), (['AlexNet', 'features', '0'], (223.438, 1080.1919765472412))]

print(observer)
```
```text
Module         |  CPU Time | CUDA Time
---------------|-----------|----------
AlexNet        |   3.837ms |  13.198ms
├── features   |   3.527ms |  14.528ms
│  ├── 0       | 223.438us |   1.080ms
│  ├── 1       |  18.270us |  20.448us
│  ├── 2       |  29.030us |  52.224us
│  ├── 3       |  76.570us |   1.108ms
│  ├── 4       |  17.480us |  17.600us
│  ├── 5       |  28.150us |  51.008us
│  ├── 6       |  83.519us | 475.840us
│  ├── 7       |  17.820us |  18.432us
│  ├── 8       |  83.370us | 541.664us
│  ├── 9       |  17.590us |  18.432us
│  ├── 10      |  82.769us | 425.920us
│  ├── 11      |  17.260us |  18.272us
│  └── 12      |  28.160us |  49.280us
├── avgpool    |  28.130us |  54.272us
└── classifier | 187.109us | 716.000us
   ├── 0       |  29.179us |  52.992us
   ├── 1       |  37.800us | 419.904us
   ├── 2       |  17.319us |  17.536us
   ├── 3       |  28.860us |  52.096us
   ├── 4       |  37.629us | 202.752us
   ├── 5       |  17.270us |  17.408us
   └── 6       |  37.520us |  75.648us
```

## LICENSE
[MIT](LICENSE)
