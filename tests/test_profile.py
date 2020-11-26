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
        None,  # 0
        None,  # 1
        (
            "aten::conv2d",
            "aten::convolution",
            "aten::_convolution",
            "aten::contiguous",
            "aten::contiguous",
            "aten::contiguous",
            "aten::mkldnn_convolution",
            "aten::as_strided_",
        ),  # 2
        ("aten::relu_", "aten::threshold_"),  # 3
        (
            "aten::max_pool2d",
            "aten::max_pool2d_with_indices",
            "aten::contiguous",
        ),  # 4
        (
            "aten::conv2d",
            "aten::convolution",
            "aten::_convolution",
            "aten::contiguous",
            "aten::contiguous",
            "aten::contiguous",
            "aten::mkldnn_convolution",
            "aten::as_strided_",
        ),  # 5
        ("aten::relu_", "aten::threshold_"),  # 6
        (
            "aten::max_pool2d",
            "aten::max_pool2d_with_indices",
            "aten::contiguous",
        ),  # 7
        (
            "aten::conv2d",
            "aten::convolution",
            "aten::_convolution",
            "aten::contiguous",
            "aten::contiguous",
            "aten::contiguous",
            "aten::mkldnn_convolution",
            "aten::as_strided_",
        ),  # 8
        ("aten::relu_", "aten::threshold_"),  # 9
        (
            "aten::conv2d",
            "aten::convolution",
            "aten::_convolution",
            "aten::contiguous",
            "aten::contiguous",
            "aten::contiguous",
            "aten::mkldnn_convolution",
            "aten::as_strided_",
        ),  # 10
        ("aten::relu_", "aten::threshold_"),  # 11
        (
            "aten::conv2d",
            "aten::convolution",
            "aten::_convolution",
            "aten::contiguous",
            "aten::contiguous",
            "aten::contiguous",
            "aten::mkldnn_convolution",
            "aten::as_strided_",
        ),  # 12
        ("aten::relu_", "aten::threshold_"),  # 13
        (
            "aten::max_pool2d",
            "aten::max_pool2d_with_indices",
            "aten::contiguous",
        ),  # 14
        (
            "aten::adaptive_avg_pool2d",
            "aten::_adaptive_avg_pool2d",
        ),  # 15
        None,  # 16
        (
            "aten::dropout",
            "aten::empty_like",
            "aten::bernoulli_",
            "aten::div_",
            "aten::to",
            "aten::empty_strided",
            "aten::mul",
        ),  # 17
        (
            "aten::t",
            "aten::transpose",
            "aten::as_strided",
            "aten::addmm",
            "aten::expand",
            "aten::as_strided",
        ),  # 18
        ("aten::relu_", "aten::threshold_"),  # 19
        (
            "aten::dropout",
            "aten::empty_like",
            "aten::bernoulli_",
            "aten::div_",
            "aten::to",
            "aten::empty_strided",
            "aten::mul",
        ),  # 20
        (
            "aten::t",
            "aten::transpose",
            "aten::as_strided",
            "aten::addmm",
            "aten::expand",
            "aten::as_strided",
        ),  # 21
        ("aten::relu_", "aten::threshold_"),  # 22
        (
            "aten::t",
            "aten::transpose",
            "aten::as_strided",
            "aten::addmm",
            "aten::expand",
            "aten::as_strided",
        ),  # 23
    ]

    alexnet_gpu_ops = (
        None,  # 0
        None,  # 1
        (
            "aten::conv2d",
            "aten::convolution",
            "aten::_convolution",
            "aten::contiguous",
            "aten::cudnn_convolution",
            "aten::contiguous",
            "aten::contiguous",
            "aten::reshape",
            "aten::view",
            "aten::add_",
        ),  # 2
        ("aten::relu_", "aten::threshold_"),  # 3
        (
            "aten::max_pool2d",
            "aten::max_pool2d_with_indices",
            "aten::contiguous",
        ),  # 4
        (
            "aten::conv2d",
            "aten::convolution",
            "aten::_convolution",
            "aten::contiguous",
            "aten::cudnn_convolution",
            "aten::contiguous",
            "aten::contiguous",
            "aten::reshape",
            "aten::view",
            "aten::add_",
        ),  # 5
        ("aten::relu_", "aten::threshold_"),  # 6
        (
            "aten::max_pool2d",
            "aten::max_pool2d_with_indices",
            "aten::contiguous",
        ),  # 7
        (
            "aten::conv2d",
            "aten::convolution",
            "aten::_convolution",
            "aten::contiguous",
            "aten::cudnn_convolution",
            "aten::contiguous",
            "aten::contiguous",
            "aten::reshape",
            "aten::view",
            "aten::add_",
        ),  # 8
        ("aten::relu_", "aten::threshold_"),  # 9
        (
            "aten::conv2d",
            "aten::convolution",
            "aten::_convolution",
            "aten::contiguous",
            "aten::cudnn_convolution",
            "aten::contiguous",
            "aten::contiguous",
            "aten::reshape",
            "aten::view",
            "aten::add_",
        ),  # 10
        ("aten::relu_", "aten::threshold_"),  # 11
        (
            "aten::conv2d",
            "aten::convolution",
            "aten::_convolution",
            "aten::contiguous",
            "aten::cudnn_convolution",
            "aten::contiguous",
            "aten::contiguous",
            "aten::reshape",
            "aten::view",
            "aten::add_",
        ),  # 12
        ("aten::relu_", "aten::threshold_"),  # 13
        (
            "aten::max_pool2d",
            "aten::max_pool2d_with_indices",
            "aten::contiguous",
        ),  # 14
        (
            "aten::adaptive_avg_pool2d",
            "aten::_adaptive_avg_pool2d",
            "aten::contiguous",
        ),  # 15
        None,  # 16
        (
            "aten::dropout",
            "aten::_fused_dropout",
            "aten::empty_like",
        ),  # 17
        (
            "aten::t",
            "aten::transpose",
            "aten::as_strided",
            "aten::addmm",
            "aten::expand",
            "aten::as_strided",
        ),  # 18
        ("aten::relu_", "aten::threshold_"),  # 19
        (
            "aten::dropout",
            "aten::_fused_dropout",
            "aten::empty_like",
        ),  # 20
        (
            "aten::t",
            "aten::transpose",
            "aten::as_strided",
            "aten::addmm",
            "aten::expand",
            "aten::as_strided",
        ),  # 21
        ("aten::relu_", "aten::threshold_"),  # 22
        (
            "aten::t",
            "aten::transpose",
            "aten::as_strided",
            "aten::addmm",
            "aten::expand",
            "aten::as_strided",
        ),  # 23
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
                try:
                    self.assertTrue(
                        all(event_name in event_names for event_name in alexnet_ops[layer_idx]),
                        f"Layer {layer_idx} received {event_names}, old {alexnet_ops[layer_idx]}",
                    )
                except IndexError:
                    self.assertTrue(False, f"Layer {layer_idx} received {event_names}")
            else:
                # non leaf nodes should not have event_list values
                self.assertEqual(len(event_lists), 0)

        pretty = prof.display()
        pretty_full = prof.display(show_events=True)
        self.assertIsInstance(pretty, str)
        self.assertIsInstance(pretty_full, str)

        # pprint.pprint(pretty)
