"""Run the code of the user guide (docs/user_guide/*.md) and check its printed output.

The guide's code is in ```` ```pycon ```` blocks, written as doctests: ``>>>`` lines and
the output they print. The blocks of a page run in order, in one namespace, like the
examples in a docstring. A page whose code needs an optional module says so in a MyST
comment, ``% test-requires: cvxopt``, and is skipped without it.
"""

import doctest
import re
from pathlib import Path

import pytest

GUIDE = Path(__file__).parents[1] / "docs" / "user_guide"
PAGES = sorted(GUIDE.glob("*.md"))
BLOCK = re.compile(r"^```pycon\n(.*?)^```", re.MULTILINE | re.DOTALL)
REQUIRES = re.compile(r"^% test-requires: (.+)$", re.MULTILINE)
OPTIONS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.stem)
def test_user_guide_page(page):
    text = page.read_text(encoding="utf-8")
    for line in REQUIRES.findall(text):
        for module in line.split(","):
            pytest.importorskip(module.strip())
    blocks = BLOCK.findall(text)
    test = doctest.DocTestParser().get_doctest("\n".join(blocks), {}, page.stem, str(page), 0)
    report = []
    runner = doctest.DocTestRunner(optionflags=OPTIONS)
    runner.run(test, out=report.append)
    assert runner.failures == 0, "".join(report)
