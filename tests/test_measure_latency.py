import unittest
import torch
import torchprof
import torchvision
import pprint

class TestMeasureLatency(unittest.TestCase):
    def test(self):
        model = torchvision.models.densenet121(pretrained=False)
        x = torch.rand([1, 3, 224, 224])

        l_observer = torchprof.LatencyObserver(model)
        observed_latency = l_observer.measure_latency(x)

        pprint.pprint(observed_latency)


if __name__ == "__main__":
    unittest.main()
