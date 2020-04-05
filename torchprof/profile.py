import functools
import torch.autograd.profiler as tprofiler
from collections import namedtuple, defaultdict, OrderedDict

Trace = namedtuple("Trace", ["path", "leaf", "module"])
Measure = namedtuple("Measure", ["self_cpu_total", "cpu_total", "cuda_total", "occurrences"])


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
    """Layer by layer profiling of Pytorch models, using the Pytorch autograd profiler.
    """

    def __init__(self, model, enabled=True, use_cuda=False, paths=None):
        self._model = model
        self.enabled = enabled
        self.use_cuda = use_cuda
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
        if self.exited:
            return traces_to_display(
                self.traces, self.trace_profile_events, paths=self.paths
            )
        return "<unfinished torchprof.profile>"

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
                with tprofiler.profile(use_cuda=self.use_cuda) as prof:
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
            )
        return "<unfinished torchprof.profile>"


def flatten_tree(t, depth=0):
    flat = []
    for name, st in t.items():
        measures = st.pop(None, None)
        flat.append([depth, name, measures])
        flat.extend(flatten_tree(st, depth=depth + 1))
    return flat


def traces_to_display(traces, trace_events, show_events=False, paths=None):
    """Construct human readable output of the profiler traces and events.
    """
    tree = OrderedDict()

    for trace in traces:
        [path, leaf, module] = trace
        current_tree = tree
        # unwrap all of the events, in case model is called multiple times
        events = [te for tevents in trace_events[path] for te in tevents]
        for depth, name in enumerate(path, 1):
            if name not in current_tree:
                current_tree[name] = OrderedDict()
            if depth == len(path) and (
                (paths is None and leaf) or (paths is not None and path in paths)
            ):
                # tree measurements have key None, avoiding name conflict
                if show_events:
                    for event in events:
                        current_tree[name][event.name] = {
                            None: Measure(
                                sum([e.self_cpu_time_total for e in events if e.name == event.name]),
                                sum([e.cpu_time_total for e in events if e.name == event.name]),
                                sum([e.cuda_time_total for e in events if e.name == event.name]),
                                len([e for e in events if e.name == event.name])
                            )
                        }
                else:
                    current_tree[name][None] = Measure(
                        sum([e.self_cpu_time_total for e in events]),
                        sum([e.cpu_time_total for e in events]),
                        sum([e.cuda_time_total for e in events]),
                        len(trace_events[path])
                    )
            current_tree = current_tree[name]
    tree_lines = flatten_tree(tree)

    # dt = ('|', '|-- ', '+-- ', ' ') # ascii
    dt = ("\u2502", "\u251c\u2500\u2500 ", "\u2514\u2500\u2500 ", " ")  # ascii-ex
    format_lines = []
    for idx, tree_line in enumerate(tree_lines):
        depth, name, measures = tree_line
        self_cpu_time = ""
        cpu_time = ""
        cuda_time = ""
        occurrences = ""
        if measures:
            self_cpu_time = tprofiler.format_time(measures.self_cpu_total)
            cpu_time = tprofiler.format_time(measures.cpu_total)
            cuda_time = tprofiler.format_time(measures.cuda_total)
            occurrences = str(measures.occurrences)
        pre = ""
        next_depths = [pl[0] for pl in tree_lines[idx + 1 :]]
        current = True
        while depth:
            if current:
                if depth in next_depths and next_depths[0] >= depth:
                    pre = dt[1]
                else:
                    pre = dt[2]
            else:
                if depth in next_depths:
                    pre = dt[0] + pre
                else:
                    pre = dt[3] + pre
            depth -= 1
            current = False
        format_lines.append([pre + name, self_cpu_time, cpu_time, cuda_time, occurrences])

    # construct the table
    heading = ("Module", "Self CPU total", "CPU total", "CUDA total", "Occurrences")
    max_lens = [max(map(len, col)) for col in zip(*([heading] + format_lines))]
    # create the heading
    disp = "{:<{}s}".format(heading[0], max_lens[0]) + " | "
    disp += "{:>{}s}".format(heading[1], max_lens[1]) + " | "
    disp += "{:>{}s}".format(heading[2], max_lens[2]) + " | "
    disp += "{:>{}s}".format(heading[3], max_lens[3]) + " | "
    disp += "{:>{}s}".format(heading[4], max_lens[4]) + "\n"
    disp += "-|-".join(["-" * mlen for mlen in max_lens]) + "\n"
    for line in format_lines:
        label, self_cpu_time, cpu_time, cuda_time, occurrences = line
        disp += "{:<{}s}".format(label, max_lens[0]) + " | "
        disp += "{:>{}s}".format(self_cpu_time, max_lens[1]) + " | "
        disp += "{:>{}s}".format(cpu_time, max_lens[2]) + " | "
        disp += "{:>{}s}".format(cuda_time, max_lens[3]) + " | "
        disp += "{:>{}s}".format(occurrences, max_lens[4]) + "\n"

    return disp
