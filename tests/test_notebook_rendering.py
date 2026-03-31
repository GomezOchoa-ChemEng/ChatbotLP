import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from IPython.display import Markdown

from src.notebook_rendering import build_notebook_display


def test_build_notebook_display_keeps_markdown_latex_unchanged():
    rendered = build_notebook_display(
        {
            "response": "**Theorem 1.** Example.\n\n$$z_P^* = z_D^*$$",
            "render_mode": "markdown_latex",
        }
    )

    assert isinstance(rendered, Markdown)
    assert "$$z_P^* = z_D^*$$" in rendered.data


def test_build_notebook_display_wraps_math_fragment_in_markdown_display_math():
    rendered = build_notebook_display(
        {
            "response": "\\begin{aligned}x &= y + z\\end{aligned}",
            "render_mode": "math_fragment",
        }
    )

    assert isinstance(rendered, Markdown)
    assert rendered.data == "$$\n\\begin{aligned}x &= y + z\\end{aligned}\n$$"


def test_build_notebook_display_auto_detects_math_like_response():
    rendered = build_notebook_display(
        {
            "response": "\\begin{aligned}x &= y + z\\end{aligned}",
        }
    )

    assert isinstance(rendered, Markdown)
    assert rendered.data == "\\begin{aligned}x &= y + z\\end{aligned}"


def test_build_notebook_display_keeps_multiline_text_as_markdown():
    rendered = build_notebook_display(
        {
            "response": "Line one.\n\nLine two.",
        }
    )

    assert isinstance(rendered, Markdown)
    assert rendered.data == "Line one.\n\nLine two."
