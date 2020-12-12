import functools
import warnings
from collections import OrderedDict, defaultdict, namedtuple

import torch.autograd.profiler as torch_profiler

from .display import traces_to_display

Trace = namedtuple("Trace", ["path", "leaf", "module"])


def walk_modules(module, name="", path=()):
    """Generator. Walks through a PyTorch Module and outputs Trace tuples"""
    if not name:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    path = path + (name,)
    yield Trace(path, len(named_children) == 0, module)
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from walk_modules(child_module, name=name, path=path)


class Profile(object):
    """Layer by layer profiling of PyTorch models, using the PyTorch autograd profiler."""

    def __init__(
        self, model, enabled=True, use_cuda=False, profile_memory=False, paths=None
    ):
        self._model = model
        self.enabled = enabled
        self.use_cuda = use_cuda
        self.profile_memory = profile_memory
        self.paths = paths

        self.entered = False
        self.exited = False
        self.traces = ()
        self.trace_profile_events = defaultdict(list)

    def __enter__(self):
        if not self.enabled:
            return self
        if self.entered:
            raise RuntimeError("torchprof profiler is not reentrant")
        self.entered = True
        self._forwards = {}  # store the original forward functions
        self.traces = tuple(map(self._hook_trace, walk_modules(self._model)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        tuple(map(self._remove_hook_trace, self.traces))
        del self._forwards  # remove unnecessary forwards
        self.exited = True

    def __str__(self):
        return self.display()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def _hook_trace(self, trace):
        [path, leaf, module] = trace
        if (self.paths is not None and path in self.paths) or (
            self.paths is None and leaf
        ):
            _forward = module.forward
            self._forwards[path] = _forward

            @functools.wraps(_forward)
            def wrap_forward(*args, **kwargs):
                try:
                    with torch_profiler.profile(
                        use_cuda=self.use_cuda, profile_memory=self.profile_memory
                    ) as prof:
                        res = _forward(*args, **kwargs)
                except TypeError:
                    if self.profile_memory:
                        warnings.warn(
                            "`profile_memory` is unsupported in torch < 1.6",
                            RuntimeWarning,
                        )
                        self.profile_memory = False
                    with torch_profiler.profile(use_cuda=self.use_cuda) as prof:
                        res = _forward(*args, **kwargs)

                event_list = prof.function_events
                event_list.populate_cpu_children()
                # each profile call should be contained in its own list
                self.trace_profile_events[path].append(event_list)
                return res

            module.forward = wrap_forward
        return trace

    def _remove_hook_trace(self, trace):
        [path, leaf, module] = trace
        if (self.paths is not None and path in self.paths) or (
            self.paths is None and leaf
        ):
            module.forward = self._forwards[path]

    def raw(self):
        if self.exited:
            return (self.traces, self.trace_profile_events)

    def display(self, show_events=False):
        if self.exited:
            return traces_to_display(
                self.traces,
                self.trace_profile_events,
                show_events=show_events,
                paths=self.paths,
                use_cuda=self.use_cuda,
                profile_memory=self.profile_memory,
            )
        return "<unfinished torchprof.profile>"
