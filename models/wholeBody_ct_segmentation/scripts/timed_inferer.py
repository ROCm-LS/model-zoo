import time

import torch
from monai.inferers import Inferer

WARMUP = 5
MEASURE = 20


class TimedInferer(Inferer):
    """Wraps another Inferer and records CUDA-synchronized wall-clock time.

    Methodology: 5 warmup calls (discarded) + 20 measured calls, median reported.
    Matches the §1 H100 TensorRT measurement protocol in 02-RESULTS.md.
    """

    def __init__(self, inferer: Inferer, **kwargs) -> None:
        self.inferer = inferer
        self.times: list[float] = []
        self._call_count: int = 0
        # Absorb extra config keys (e.g. sw_batch_size) so TRT overlay refs resolve.
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, inputs, network, *args, **kwargs):
        self._call_count += 1
        is_warmup = self._call_count <= WARMUP

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        output = self.inferer(inputs, network, *args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tag = "warmup" if is_warmup else "measured"
        print(f"[TimedInferer] call #{self._call_count} ({tag}): {elapsed * 1000:.2f} ms")
        if not is_warmup:
            self.times.append(elapsed)
        return output

    def summary(self) -> None:
        if not self.times:
            print("[TimedInferer] no measured calls recorded")
            return
        sorted_t = sorted(self.times)
        n = len(sorted_t)
        median = sorted_t[n // 2] * 1000
        mean = sum(sorted_t) / n * 1000
        print(
            f"[TimedInferer] {WARMUP} warmup + {n} measured | "
            f"median {median:.2f} ms | mean {mean:.2f} ms"
        )
