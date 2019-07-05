import unittest
import torch
import torchprof
import torchvision
import pprint


class TestMeasureLatency(unittest.TestCase):
    def test_cpu_latency(self):
        model = torchvision.models.alexnet(pretrained=False)
        x = torch.rand([1, 3, 224, 224])

        l_observer = torchprof.LatencyObserver(model)
        observed_latency = l_observer.measure_latency(x)

        # Measured 24 modules within alexnet
        self.assertEqual(len(observed_latency), 24)
        human_readable = str(l_observer)
        for trace, measure in observed_latency:
            for module in trace:
                self.assertIn(module, human_readable)

    def test_gpu_latency(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        model = torchvision.models.densenet121(pretrained=False).cuda()
        x = torch.rand([1, 3, 224, 224]).cuda()

        l_observer = torchprof.LatencyObserver(model, use_cuda=True)
        observed_latency = l_observer.measure_latency(x)
        self.assertEqual(len(observed_latency), 433)


if __name__ == "__main__":
    unittest.main()
