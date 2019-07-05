import torch
from collections import OrderedDict


class LatencyObserver:
    _module_inputs = {}
    _latency_measures = []
    use_cuda = False

    def __init__(self, module: torch.nn.Module, use_cuda: bool = False):
        self.module = module
        self._register_module_hooks(self.module)
        self.use_cuda = use_cuda

    def __str__(self):
        tree = self._measures_to_tree()
        format_lines = self._structure_pretty_format(tree)

        # get the maximum character lengths for each column
        heading = ["Module", "CPU Time", "CUDA Time"]
        max_lens = [max(map(len, col)) for col in zip(*([heading] + format_lines))]

        # create the heading
        disp = "{:<{}s}".format(heading[0], max_lens[0]) + " | "
        disp += "{:>{}s}".format(heading[1], max_lens[1]) + " | "
        disp += "{:>{}s}".format(heading[2], max_lens[2]) + "\n"
        disp += "-|-".join(["-" * mlen for mlen in max_lens]) + "\n"
        for line in format_lines:
            label, cpu_time, cuda_time = line
            disp += "{:<{}s}".format(label, max_lens[0]) + " | "
            disp += "{:>{}s}".format(cpu_time, max_lens[1]) + " | "
            disp += "{:>{}s}".format(cuda_time, max_lens[2]) + "\n"
        return disp

    def __repr__(self):
        return repr(self._measures_to_tree())

    @staticmethod
    def _structure_pretty_format(tree):
        pretty_lines = LatencyObserver._pretty_format(tree)
        format_lines = []
        for idx, pretty_line in enumerate(pretty_lines):
            depth, name, measures = pretty_line
            cpu_time, gpu_time = [
                torch.autograd.profiler.format_time(x) for x in measures
            ]
            pre = ""
            prev_depths = [pl[0] for pl in pretty_lines[:idx]]
            next_depths = [pl[0] for pl in pretty_lines[idx + 1 :]]
            current = True
            while depth:
                if current:
                    if depth in next_depths and next_depths[0] >= depth:
                        pre = "\u251c\u2500\u2500 "
                    else:
                        pre = "\u2514\u2500\u2500 "
                else:
                    if depth in next_depths:
                        pre = "\u2502  " + pre
                    else:
                        pre = "   " + pre
                depth -= 1
                current = False
            format_lines.append([pre + name, cpu_time, gpu_time])
        return format_lines

    @staticmethod
    def _pretty_format(tree, depth=0):
        pretty_lines = []
        for name, subtree in tree.items():
            measures = subtree.pop(None)
            pretty_lines.append([depth, name, measures])
            pretty_lines.extend(LatencyObserver._pretty_format(subtree, depth + 1))
        return pretty_lines

    @staticmethod
    def _trace_to_key(trace):
        return "|".join(trace)

    def _input_hook(self, key):
        def _save_input(_self, module_input):
            self._module_inputs[key] = module_input

        return _save_input

    def _register_module_hooks(self, module, name=None, ancestors=[]):
        if name is None:
            name = module._get_name()
        trace = ancestors + [name]
        module.register_forward_pre_hook(self._input_hook(self._trace_to_key(trace)))

        for child_name, child in module.named_children():
            self._register_module_hooks(child, name=child_name, ancestors=trace)

    def _measure_recursive_latency(self, child, name, ancestors):
        trace = ancestors + [name]
        child_input = self._module_inputs[self._trace_to_key(trace)]

        with torch.autograd.profiler.profile(use_cuda=self.use_cuda) as prof:
            child(*child_input)
        cpu_time_total = prof.self_cpu_time_total
        cuda_time_total = sum([event.cuda_time_total for event in prof.function_events])
        self._latency_measures.append((trace, (cpu_time_total, cuda_time_total)))

        # recurse into children to get layer specific profile metrics
        for gchild_name, gchild in child.named_children():
            self._measure_recursive_latency(gchild, name=gchild_name, ancestors=trace)

    def _measures_to_tree(self):
        tree = OrderedDict()
        for trace, measurements in self._latency_measures:
            current_tree = tree
            for depth, module in enumerate(trace, 1):
                if module not in current_tree:
                    current_tree[module] = OrderedDict()
                if depth == len(trace):
                    current_tree[module][None] = measurements
                current_tree = current_tree[module]
        return tree

    def measure_latency(self, module_input: torch.Tensor, name=None, ancestors=[]):
        """Return the layer by layer latency of running the module
        Torch profiler output is in microseconds (second 10^-6)
        """
        self._module_inputs = {}
        self._latency_measures = []

        if name is None:
            name = self.module._get_name()
        trace = ancestors + [name]

        # get overall module performance, seed module input values
        with torch.autograd.profiler.profile(use_cuda=self.use_cuda) as prof:
            self.module(module_input)

        cpu_time_total = prof.self_cpu_time_total
        cuda_time_total = sum([event.cuda_time_total for event in prof.function_events])
        self._latency_measures.append((trace, (cpu_time_total, cuda_time_total)))

        # recurse into children to get layer specific profile metrics
        for child_name, child in self.module.named_children():
            self._measure_recursive_latency(child, name=child_name, ancestors=trace)

        return self._latency_measures
