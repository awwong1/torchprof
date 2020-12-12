from collections import OrderedDict, namedtuple

import torch.autograd.profiler as torch_profiler

Measure = namedtuple(
    "Measure",
    [  # when attr value is None, profiler unsupported
        "self_cpu_total",
        "cpu_total",
        "self_cuda_total",
        "cuda_total",
        "cpu_memory",
        "self_cpu_memory",
        "cuda_memory",
        "self_cuda_memory",
        "occurrences",
    ],
)


def _flatten_tree(t, depth=0):
    flat = []
    for name, st in t.items():
        measures = st.pop(None, None)
        flat.append([depth, name, measures])
        flat.extend(_flatten_tree(st, depth=depth + 1))
    return flat


def _build_measure_tuple(events, occurrences):
    # Events may have missing attributes depending on the PyTorch version used.

    # memory profiling supported in torch >= 1.6
    self_cpu_memory = None
    has_self_cpu_memory = any(hasattr(e, "self_cpu_memory_usage") for e in events)
    if has_self_cpu_memory:
        self_cpu_memory = sum([getattr(e, "self_cpu_memory_usage", 0) for e in events])
    cpu_memory = None
    has_cpu_memory = any(hasattr(e, "cpu_memory_usage") for e in events)
    if has_cpu_memory:
        cpu_memory = sum([getattr(e, "cpu_memory_usage", 0) for e in events])
    self_cuda_memory = None
    has_self_cuda_memory = any(hasattr(e, "self_cuda_memory_usage") for e in events)
    if has_self_cuda_memory:
        self_cuda_memory = sum(
            [getattr(e, "self_cuda_memory_usage", 0) for e in events]
        )
    cuda_memory = None
    has_cuda_memory = any(hasattr(e, "cuda_memory_usage") for e in events)
    if has_cuda_memory:
        cuda_memory = sum([getattr(e, "cuda_memory_usage", 0) for e in events])

    # self CUDA time supported in torch >= 1.7
    self_cuda_total = None
    has_self_cuda_time = any(hasattr(e, "self_cuda_time_total") for e in events)
    if has_self_cuda_time:
        self_cuda_total = sum([getattr(e, "self_cuda_time_total", 0) for e in events])

    return Measure(
        self_cpu_total=sum([e.self_cpu_time_total for e in events]),
        cpu_total=sum([e.cpu_time_total for e in events]),
        self_cuda_total=self_cuda_total,
        cuda_total=sum([e.cuda_time_total for e in events]),
        self_cpu_memory=self_cpu_memory,
        cpu_memory=cpu_memory,
        self_cuda_memory=self_cuda_memory,
        cuda_memory=cuda_memory,
        occurrences=occurrences,
    )


def _format_measure_tuple(measure):
    format_memory = getattr(torch_profiler, "format_memory", lambda _: "N/A")

    self_cpu_total = (
        torch_profiler.format_time(measure.self_cpu_total) if measure else ""
    )
    cpu_total = torch_profiler.format_time(measure.cpu_total) if measure else ""
    self_cuda_total = (
        torch_profiler.format_time(measure.self_cuda_total)
        if measure and measure.self_cuda_total is not None
        else ""
    )
    cuda_total = torch_profiler.format_time(measure.cuda_total) if measure else ""
    self_cpu_memory = (
        format_memory(measure.self_cpu_memory)
        if measure and measure.self_cpu_memory is not None
        else ""
    )
    cpu_memory = (
        format_memory(measure.cpu_memory)
        if measure and measure.cpu_memory is not None
        else ""
    )
    self_cuda_memory = (
        format_memory(measure.self_cuda_memory)
        if measure and measure.self_cuda_memory is not None
        else ""
    )
    cuda_memory = (
        format_memory(measure.cuda_memory)
        if measure and measure.cuda_memory is not None
        else ""
    )
    occurrences = str(measure.occurrences) if measure else ""

    return Measure(
        self_cpu_total=self_cpu_total,
        cpu_total=cpu_total,
        self_cuda_total=self_cuda_total,
        cuda_total=cuda_total,
        self_cpu_memory=self_cpu_memory,
        cpu_memory=cpu_memory,
        self_cuda_memory=self_cuda_memory,
        cuda_memory=cuda_memory,
        occurrences=occurrences,
    )


def group_by(events, keyfn):
    event_groups = OrderedDict()
    for event in events:
        key = keyfn(event)
        key_events = event_groups.get(key, [])
        key_events.append(event)
        event_groups[key] = key_events
    return event_groups.items()


def traces_to_display(
    traces,
    trace_events,
    show_events=False,
    paths=None,
    use_cuda=False,
    profile_memory=False,
    # dt = ('|', '|-- ', '+-- ', ' ') # ascii
    dt=("\u2502", "\u251c\u2500\u2500 ", "\u2514\u2500\u2500 ", " "),  # ascii-ex
):
    """Construct human readable output of the profiler traces and events."""
    tree = OrderedDict()

    for trace in traces:
        [path, leaf, module] = trace
        current_tree = tree
        # unwrap all of the events, in case model is called multiple times
        events = [te for t_events in trace_events[path] for te in t_events]
        for depth, name in enumerate(path, 1):
            if name not in current_tree:
                current_tree[name] = OrderedDict()
            if depth == len(path) and (
                (paths is None and leaf) or (paths is not None and path in paths)
            ):
                # tree measurements have key None, avoiding name conflict
                if show_events:
                    for event_name, event_group in group_by(events, lambda e: e.name):
                        event_group = list(event_group)
                        current_tree[name][event_name] = {
                            None: _build_measure_tuple(event_group, len(event_group))
                        }
                else:
                    current_tree[name][None] = _build_measure_tuple(
                        events, len(trace_events[path])
                    )
            current_tree = current_tree[name]
    tree_lines = _flatten_tree(tree)

    format_lines = []
    has_self_cuda_total = False
    has_self_cpu_memory = False
    has_cpu_memory = False
    has_self_cuda_memory = False
    has_cuda_memory = False

    for idx, tree_line in enumerate(tree_lines):
        depth, name, measures = tree_line

        next_depths = [pl[0] for pl in tree_lines[idx + 1 :]]
        pre = ""
        if depth > 0:
            pre = dt[1] if depth in next_depths and next_depths[0] >= depth else dt[2]
            depth -= 1
        while depth > 0:
            pre = (dt[0] + pre) if depth in next_depths else (dt[3] + pre)
            depth -= 1
        format_lines.append([pre + name, *_format_measure_tuple(measures)])
        if measures:
            has_self_cuda_total = (
                has_self_cuda_total or measures.self_cuda_total is not None
            )
            has_self_cpu_memory = (
                has_self_cpu_memory or measures.self_cpu_memory is not None
            )
            has_cpu_memory = has_cpu_memory or measures.cpu_memory is not None
            has_self_cuda_memory = (
                has_self_cuda_memory or measures.self_cuda_memory is not None
            )
            has_cuda_memory = has_cuda_memory or measures.cuda_memory is not None

    # construct the table (this is pretty ugly and can probably be optimized)
    heading = (
        "Module",
        "Self CPU total",
        "CPU total",
        "Self CUDA total",
        "CUDA total",
        "Self CPU Mem",
        "CPU Mem",
        "Self CUDA Mem",
        "CUDA Mem",
        "Number of Calls",
    )
    max_lens = [max(map(len, col)) for col in zip(*([heading] + format_lines))]

    # not all columns should be displayed, specify kept indexes
    keep_indexes = [0, 1, 2, 9]
    if profile_memory:
        if has_self_cpu_memory:
            keep_indexes.append(5)
        if has_cpu_memory:
            keep_indexes.append(6)
    if use_cuda:
        if has_self_cuda_total:
            keep_indexes.append(3)
        keep_indexes.append(4)
        if profile_memory:
            if has_self_cuda_memory:
                keep_indexes.append(7)
            if has_cuda_memory:
                keep_indexes.append(8)
    keep_indexes = tuple(sorted(keep_indexes))

    display = (  # table heading
        " | ".join(
            [
                "{:<{}s}".format(heading[keep_index], max_lens[keep_index])
                for keep_index in keep_indexes
            ]
        )
        + "\n"
    )
    display += (  # separator
        "-|-".join(
            [
                "-" * max_len
                for val_idx, max_len in enumerate(max_lens)
                if val_idx in keep_indexes
            ]
        )
        + "\n"
    )
    for format_line in format_lines:  # body
        display += (
            " | ".join(
                [
                    "{:<{}s}".format(value, max_lens[val_idx])
                    for val_idx, value in enumerate(format_line)
                    if val_idx in keep_indexes
                ]
            )
            + "\n"
        )

    return display
