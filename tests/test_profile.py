import unittest
import torch
import torchprof
import torchvision
import pprint

class TestProfile(unittest.TestCase):
    def test_profile(self):
        model = torchvision.models.alexnet(pretrained=False)
        x = torch.rand([1, 3, 224, 224])

        with torchprof.Profile(model) as prof:
            model(x)

        traces, profile_events = prof.raw()

        # ensure traces match alexnet
        alexnet_trace_stubs = [
            (('AlexNet',), False),
            (('AlexNet', 'features'), False),
            (('AlexNet', 'features', '0'), True),
            (('AlexNet', 'features', '1'), True),
            (('AlexNet', 'features', '2'), True),
            (('AlexNet', 'features', '3'), True),
            (('AlexNet', 'features', '4'), True),
            (('AlexNet', 'features', '5'), True),
            (('AlexNet', 'features', '6'), True),
            (('AlexNet', 'features', '7'), True),
            (('AlexNet', 'features', '8'), True),
            (('AlexNet', 'features', '9'), True),
            (('AlexNet', 'features', '10'), True),
            (('AlexNet', 'features', '11'), True),
            (('AlexNet', 'features', '12'), True),
            (('AlexNet', 'avgpool'), True),
            (('AlexNet', 'classifier'), False),
            (('AlexNet', 'classifier', '0'), True),
            (('AlexNet', 'classifier', '1'), True),
            (('AlexNet', 'classifier', '2'), True),
            (('AlexNet', 'classifier', '3'), True),
            (('AlexNet', 'classifier', '4'), True),
            (('AlexNet', 'classifier', '5'), True),
            (('AlexNet', 'classifier', '6'), True)]

        for layer_idx, trace in enumerate(traces):
            (path, leaf, _) = trace
            self.assertEqual((path, leaf), alexnet_trace_stubs[layer_idx])
