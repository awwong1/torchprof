import unittest
import torch
import torchprof
import torchvision


class TestProfile(unittest.TestCase):
    # ensure traces match alexnet
    alexnet_traces = [
        (("AlexNet",), False),
        (("AlexNet", "features"), False),
        (("AlexNet", "features", "0"), True),
        (("AlexNet", "features", "1"), True),
        (("AlexNet", "features", "2"), True),
        (("AlexNet", "features", "3"), True),
        (("AlexNet", "features", "4"), True),
        (("AlexNet", "features", "5"), True),
        (("AlexNet", "features", "6"), True),
        (("AlexNet", "features", "7"), True),
        (("AlexNet", "features", "8"), True),
        (("AlexNet", "features", "9"), True),
        (("AlexNet", "features", "10"), True),
        (("AlexNet", "features", "11"), True),
        (("AlexNet", "features", "12"), True),
        (("AlexNet", "avgpool"), True),
        (("AlexNet", "classifier"), False),
        (("AlexNet", "classifier", "0"), True),
        (("AlexNet", "classifier", "1"), True),
        (("AlexNet", "classifier", "2"), True),
        (("AlexNet", "classifier", "3"), True),
        (("AlexNet", "classifier", "4"), True),
        (("AlexNet", "classifier", "5"), True),
        (("AlexNet", "classifier", "6"), True),
    ]

    alexnet_cpu_ops = [
        None,
        None,
        (
            "conv2d",
            "convolution",
            "_convolution",
            "contiguous",
            "contiguous",
            "contiguous",
            "mkldnn_convolution",
        ),
        ("relu_",),
        ("max_pool2d", "max_pool2d_with_indices"),
        (
            "conv2d",
            "convolution",
            "_convolution",
            "contiguous",
            "contiguous",
            "contiguous",
            "mkldnn_convolution",
        ),
        ("relu_",),
        ("max_pool2d", "max_pool2d_with_indices"),
        (
            "conv2d",
            "convolution",
            "_convolution",
            "contiguous",
            "contiguous",
            "contiguous",
            "mkldnn_convolution",
        ),
        ("relu_",),
        (
            "conv2d",
            "convolution",
            "_convolution",
            "contiguous",
            "contiguous",
            "contiguous",
            "mkldnn_convolution",
        ),
        ("relu_",),
        (
            "conv2d",
            "convolution",
            "_convolution",
            "contiguous",
            "contiguous",
            "contiguous",
            "mkldnn_convolution",
        ),
        ("relu_",),
        ("max_pool2d", "max_pool2d_with_indices"),
        ("adaptive_avg_pool2d", "_adaptive_avg_pool2d"),
        None,
        ("dropout", "empty_like", "empty", "bernoulli_", "div_", "mul"),
        ("unsigned short", "addmm"),
        ("relu_",),
        ("dropout", "empty_like", "empty", "bernoulli_", "div_", "mul"),
        ("unsigned short", "addmm"),
        ("relu_",),
        ("unsigned short", "addmm"),
    ]

    alexnet_gpu_ops = (
        None,
        None,
        ("conv2d", "convolution", "_convolution", "contiguous", "cudnn_convolution"),
        ("relu_",),
        ("max_pool2d", "max_pool2d_with_indices"),
        ("conv2d", "convolution", "_convolution", "contiguous", "cudnn_convolution"),
        ("relu_",),
        ("max_pool2d", "max_pool2d_with_indices"),
        ("conv2d", "convolution", "_convolution", "contiguous", "cudnn_convolution"),
        ("relu_",),
        ("conv2d", "convolution", "_convolution", "contiguous", "cudnn_convolution"),
        ("relu_",),
        ("conv2d", "convolution", "_convolution", "contiguous", "cudnn_convolution"),
        ("relu_",),
        ("max_pool2d", "max_pool2d_with_indices"),
        ("adaptive_avg_pool2d", "_adaptive_avg_pool2d"),
        None,
        ("dropout", "_fused_dropout"),
        ("unsigned short", "addmm"),
        ("relu_",),
        ("dropout", "_fused_dropout"),
        ("unsigned short", "addmm"),
        ("relu_",),
        ("unsigned short", "addmm"),
    )

    def test_cpu_profile_structure(self):
        model = torchvision.models.alexnet(pretrained=False)
        x = torch.rand([1, 3, 224, 224])
        self._profile_structure(model, x, alexnet_ops=self.alexnet_cpu_ops)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_profile_structure(self):
        model = torchvision.models.alexnet(pretrained=False).cuda()
        x = torch.rand([1, 3, 224, 224]).cuda()
        self._profile_structure(
            model, x, use_cuda=True, alexnet_ops=self.alexnet_gpu_ops
        )

    def _profile_structure(self, model, x, use_cuda=False, alexnet_ops=[]):
        with torchprof.Profile(model, use_cuda=use_cuda) as prof:
            model(x)

        traces, event_lists_dict = prof.raw()

        for layer_idx, trace in enumerate(traces):
            (path, leaf, _) = trace
            self.assertEqual((path, leaf), self.alexnet_traces[layer_idx])
            event_lists = event_lists_dict[path]
            if leaf:
                # model(x) called once, each layer should have one event_list
                self.assertEqual(len(event_lists), 1)
                event_names = tuple(e.name for e in event_lists[0])
                # profiler returned order is not deterministic
                self.assertEqual(sorted(event_names), sorted(alexnet_ops[layer_idx]))
            else:
                # non leaf nodes should not have event_list values
                self.assertEqual(len(event_lists), 0)

        pretty = prof.display()
        pretty_full = prof.display(show_events=True)
        self.assertIsInstance(pretty, str)
        self.assertIsInstance(pretty_full, str)

        # pprint.pprint(pretty)
