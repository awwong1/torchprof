import torch


class LatencyObserver:
    _module_inputs = {}
    _latency_measures = {}

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self._register_module_hooks(self.module)

    def _input_hook(self, key):
        def _print_input(_, module_input):
            # store all intermediate input values
            self._module_inputs[key] = module_input

        return _print_input

    def _register_module_hooks(self, module, name=None, ancestors=[]):
        if name is None:
            name = module._get_name()
        trace = ancestors + [name]
        key = "|".join(trace)
        module.register_forward_pre_hook(self._input_hook(key))

        for child_name, child in module.named_children():
            self._register_module_hooks(child, name=child_name, ancestors=trace)

    def _measure_recursive_latency(self, child, name, ancestors):
        trace = ancestors + [name]
        key = "|".join(trace)
        child_input = self._module_inputs[key]
        with torch.autograd.profiler.profile() as prof:
            child(*child_input)
        self._latency_measures[key] = prof.self_cpu_time_total

        # recurse into children to get layer specific profile metrics
        for gchild_name, gchild in child.named_children():
            self._measure_recursive_latency(gchild, name=gchild_name, ancestors=trace)

    def measure_latency(self, module_input: torch.Tensor, name=None, ancestors=[]):
        """Return the layer by layer latency of running the module
        Torch profiler output is in nanoseconds (second 10^-6)
        """
        self._module_inputs = {}
        self._latency_measures = {}

        if name is None:
            name = self.module._get_name()
        trace = ancestors + [name]
        key = "|".join(trace)

        # get overall module performance, seed module input values
        with torch.autograd.profiler.profile() as prof:
            self.module(module_input)
        self._latency_measures[key] = prof.self_cpu_time_total

        # recurse into children to get layer specific profile metrics
        for child_name, child in self.module.named_children():
            self._measure_recursive_latency(child, name=child_name, ancestors=trace)

        return self._latency_measures
