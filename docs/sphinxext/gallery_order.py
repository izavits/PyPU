"""Order of the examples in the gallery: the order in which examples/README.rst lists them.

The README groups the model examples by family and lists the getting-started examples in
reading order, and a test checks that it lists every script, so it is the one place to
change the order.
"""

import re
from pathlib import Path

README = Path(__file__).parents[2] / "examples" / "README.rst"


class ReadmeOrder:
    """Sort key for sphinx-gallery's ``within_subsection_order``.

    Parameters
    ----------
    src_dir : str or Path
        The gallery section being sorted (unused: the README covers every section).
    """

    def __init__(self, src_dir):
        self.src_dir = src_dir
        names = re.findall(r"plot_\w+\.py", README.read_text(encoding="utf-8"))
        self.position = {name: i for i, name in reversed(list(enumerate(names)))}

    def __call__(self, filename):
        """Return the position of ``filename`` in the README; unlisted files go last."""
        name = Path(filename).name
        return (self.position.get(name, len(self.position)), name)

    def __repr__(self):
        return f"{type(self).__name__}()"
