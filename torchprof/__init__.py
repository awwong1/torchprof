name = "torchprof"

from torchprof.latency_observer import LatencyObserver
from torchprof.profile import Profile

__all__ = ["LatencyObserver", "Profile"]
__version__ = "0.3.0"