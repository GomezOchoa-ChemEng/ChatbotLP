"""Small helpers for rendering chatbot outputs in Colab/Jupyter."""

from __future__ import annotations

from typing import Any, Mapping


def build_notebook_display(result: Mapping[str, Any]) -> Any:
    """Return an IPython display object for a chatbot result when possible."""

    from IPython.display import Markdown

    response = str(result.get("response", ""))
    render_mode = result.get("render_mode")
    math_markers = ("$$", "\\[", "\\begin{aligned}")

    if render_mode == "math_fragment":
        return Markdown(f"$$\n{response}\n$$")
    if render_mode == "markdown_latex":
        return Markdown(response)
    if any(marker in response for marker in math_markers):
        return Markdown(response)
    return Markdown(response)


def render_chatbot_result(result: Mapping[str, Any]) -> Any:
    """Display a chatbot result in Colab/Jupyter and return the display handle."""

    from IPython.display import display

    rendered = build_notebook_display(result)
    display(rendered)
    return rendered
