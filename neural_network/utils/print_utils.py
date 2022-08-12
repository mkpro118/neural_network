import sys
from .typesafety import type_safe


class ProgressBar:

    BACKSPACE = '\b'

    @type_safe
    def __init__(self, n_steps, prefix="", size=20, out=sys.stdout, dynamic_display: bool = True):
        self.n_steps = n_steps
        self.prefix = prefix
        self.size = size
        self.out = out
        self.dynamic_display = dynamic_display

    def __enter__(self):
        self.curr = 1
        return self

    def __exit__(self, *args):
        print("\n", flush=True, file=self.out)

    def update(self):
        ProgressBar.show(
            self.curr,
            self.n_steps,
            self.prefix,
            self.dynamic_display,
            self.size,
            self.out
        )
        self.curr += 1

    @staticmethod
    def show(step, n_steps, prefix, dynamic_display, size, out):
        x = int(size * step / n_steps)
        print(
            f"{prefix} "
            f"{step}/{n_steps} "
            f"[{'=' * (x)}"
            f"{ProgressBar.BACKSPACE + '>' if step != n_steps else ''}"
            f"{('.' * (size-x))}]",
            end='\r' if dynamic_display else '\n', file=out, flush=True
        )
