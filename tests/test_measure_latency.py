import unittest
import torch
import torchprof
import torchvision
import pprint

class TestMeasureLatency(unittest.TestCase):
    def test_cpu_latency(self):
        model = torchvision.models.densenet121(pretrained=False)
        x = torch.rand([1, 3, 224, 224])

        l_observer = torchprof.LatencyObserver(model)
        observed_latency = l_observer.measure_latency(x)

        pprint.pprint(observed_latency)


    def test_gpu_latency(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        model = torchvision.models.densenet121(pretrained=False).cuda()
        x = torch.rand([1, 3, 224, 224]).cuda()

        l_observer = torchprof.LatencyObserver(model, use_cuda=True)
        observed_latency = l_observer.measure_latency(x)

        pprint.pprint(observed_latency)

if __name__ == "__main__":
    unittest.main()
