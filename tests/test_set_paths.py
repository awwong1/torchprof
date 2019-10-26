import unittest
import torch
import torchprof
import torchvision


class TestSetPaths(unittest.TestCase):
    def test_cpu_profile_structure(self):
        model = torchvision.models.alexnet(pretrained=False)
        x = torch.rand([1, 3, 224, 224])

        paths = [("AlexNet", "features", "3"), ("AlexNet", "avgpool")]

        with torchprof.Profile(model, paths=paths) as prof:
            model(x)

        # print(prof)
        traces, event_dict = prof.raw()
        self.assertEqual(len(event_dict.keys()), 2)
        self.assertEqual(list(event_dict.keys()), paths)
