"""Modal sandbox service for secure code execution.

Runs user/LLM-generated Python code in an isolated Modal sandbox.
Used for feature engineering and visualization pipelines.

Authentication: set MODAL_TOKEN_ID + MODAL_TOKEN_SECRET in .env (or export as env vars).
The Modal SDK reads these automatically from the process environment.
"""

import asyncio
import logging
import os
from collections.abc import Callable

import modal

from autoviz.config import settings

logger = logging.getLogger(__name__)

INPUT_CSV_PATH = "/home/user/input.csv"
OUTPUT_CSV_PATH = "/home/user/output.csv"
OUTPUT_PNG_PATH = "/home/user/output.png"
_SCRIPT_PATH = "/tmp/script.py"

CODE_PREFIX = f"""import pandas as pd

df = pd.read_csv('{INPUT_CSV_PATH}')

"""

# Font config for CJK (Chinese/Japanese/Korean) - always applied for viz to avoid boxes.
# Strategy: find font files via glob, load via addfont(), read the ACTUAL family name from the
# file itself (FontProperties.get_name), then inject that into rcParams. This avoids any
# guesswork about font family names and bypasses the stale matplotlib font cache.
VIZ_FONT_PREFIX = """
import glob as _glob
import matplotlib
import matplotlib.font_manager as _fm

# Load CJK fonts via glob (bypasses stale font cache; findSystemFonts skips .ttc in old versions)
_cjk_candidates = (
    _glob.glob('/usr/share/fonts/opentype/noto/*CJK*.ttc') +
    _glob.glob('/usr/share/fonts/opentype/noto/*CJK*.otf') +
    _glob.glob('/usr/share/fonts/**/*CJK*.ttc', recursive=True) +
    _glob.glob('/usr/share/fonts/**/*CJK*.ttf', recursive=True) +
    _glob.glob('/usr/share/fonts/**/*wqy*', recursive=True) +
    _glob.glob('/usr/share/fonts/**/*WenQuanYi*', recursive=True)
)
_cjk_names = []
for _f in _cjk_candidates:
    try:
        _fm.fontManager.addfont(_f)
        _n = _fm.FontProperties(fname=_f).get_name()
        if _n and _n not in _cjk_names:
            _cjk_names.append(_n)
    except Exception:
        pass
def _apply_cjk_font():
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = (
        _cjk_names + ['DejaVu Sans']
    ) if _cjk_names else ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

_apply_cjk_font()

# sns.set_theme() and plt.style.use() reset rcParams and wipe our font settings.
# Monkey-patch both so CJK fonts are always re-applied after any style change.
try:
    import seaborn as _sns
    _orig_sns_set_theme = _sns.set_theme
    def _patched_set_theme(*_a, **_kw):
        _orig_sns_set_theme(*_a, **_kw)
        _apply_cjk_font()
    _sns.set_theme = _patched_set_theme
    _sns.set = _patched_set_theme
except ImportError:
    pass

try:
    import matplotlib.style as _mpl_style
    _orig_style_use = _mpl_style.use
    def _patched_style_use(*_a, **_kw):
        _orig_style_use(*_a, **_kw)
        _apply_cjk_font()
    _mpl_style.use = _patched_style_use
    import matplotlib.pyplot as _plt_pre
    _plt_pre.style.use = _patched_style_use
except Exception:
    pass

# Generated code often overrides rcParams with Windows-only fonts (SimHei etc.).
# Intercept rcParams assignments so the CJK list survives any later override.
try:
    import matplotlib as _mpl
    _orig_rc_setitem = _mpl.RcParams.__setitem__

    def _patched_rc_setitem(self, key, val):
        if key == 'font.sans-serif':
            requested = list(val) if isinstance(val, (list, tuple)) else [val]
            if _cjk_names:
                merged = list(_cjk_names)
                for _name in requested:
                    if _name not in merged and _name != 'DejaVu Sans':
                        merged.append(_name)
                merged.append('DejaVu Sans')
                val = merged
        elif key == 'font.family':
            val = 'sans-serif'
        elif key == 'axes.unicode_minus':
            val = False
        return _orig_rc_setitem(self, key, val)

    _mpl.RcParams.__setitem__ = _patched_rc_setitem
except Exception:
    pass

# Re-apply the CJK font list immediately before rendering, when the figure is
# actually drawn, so the final PNG never falls back to tofu boxes.
try:
    import matplotlib.figure as _mpl_figure
    import matplotlib.pyplot as _plt
    _orig_plt_savefig = _plt.savefig
    _orig_fig_savefig = _mpl_figure.Figure.savefig

    def _patched_plt_savefig(*_a, **_kw):
        _apply_cjk_font()
        return _orig_plt_savefig(*_a, **_kw)

    def _patched_fig_savefig(self, *_a, **_kw):
        _apply_cjk_font()
        return _orig_fig_savefig(self, *_a, **_kw)

    _plt.savefig = _patched_plt_savefig
    _mpl_figure.Figure.savefig = _patched_fig_savefig
except Exception:
    pass

"""

# Modal image with all required packages and CJK fonts.
# Modal builds and caches this image on first use; subsequent sandbox creations are fast.
_MODAL_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("fonts-noto-cjk", "fonts-wqy-zenhei")
    .pip_install(
        "pandas",
        "matplotlib",
        "seaborn",
        "numpy",
        "scipy",
        "plotly",
        "kaleido",
        "squarify"
    )
)


def _ensure_modal_credentials() -> None:
    """Inject Modal token from settings into the process environment if not already set.

    Modal SDK reads MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from os.environ automatically.
    pydantic-settings populates Python attributes but does not write to os.environ,
    so we bridge them here on first use.
    """
    if settings.MODAL_TOKEN_ID:
        os.environ.setdefault("MODAL_TOKEN_ID", settings.MODAL_TOKEN_ID)
    if settings.MODAL_TOKEN_SECRET:
        os.environ.setdefault("MODAL_TOKEN_SECRET", settings.MODAL_TOKEN_SECRET)

    if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
        raise ValueError(
            "Modal credentials not configured. "
            "Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env (or as environment variables)."
        )


async def _create_sandbox(timeout: float) -> modal.Sandbox:
    """Create a Modal sandbox using Modal's native async interface."""
    app = await modal.App.lookup.aio(settings.MODAL_APP_NAME, create_if_missing=True)
    return await modal.Sandbox.create.aio(
        image=_MODAL_IMAGE,
        app=app,
        timeout=int(timeout) + 60,  # sandbox lifetime = exec timeout + 60 s buffer
    )


async def _exec_script(
    sb: modal.Sandbox,
    full_code: str,
    input_files: dict[str, bytes | str],
    exec_timeout: float,
    on_stdout: Callable[[str], None] | None = None,
    on_stderr: Callable[[str], None] | None = None,
) -> None:
    """Write files into the sandbox, execute the script, stream output, check for errors."""
    # Upload input files using Modal's native async interface
    for path, content in input_files.items():
        if isinstance(content, str):
            await sb.filesystem.write_text.aio(content, path)
        else:
            await sb.filesystem.write_bytes.aio(content, path)

    await sb.filesystem.write_text.aio(full_code, _SCRIPT_PATH)

    # exec.aio() returns an async ContainerProcess whose stdout/stderr support async iteration
    process = await sb.exec.aio("python", _SCRIPT_PATH, timeout=int(exec_timeout))

    # Drain stdout and stderr concurrently to avoid pipe buffer deadlocks
    stderr_lines: list[str] = []

    async def _drain_stdout() -> None:
        async for line in process.stdout:
            stripped = line.rstrip("\n")
            if on_stdout and stripped:
                on_stdout(stripped)
            elif stripped:
                logger.info("Modal sandbox stdout: %s", stripped)

    async def _drain_stderr() -> None:
        async for line in process.stderr:
            stripped = line.rstrip("\n")
            stderr_lines.append(stripped)
            if on_stderr and stripped:
                on_stderr(stripped)
            elif stripped:
                logger.warning("Modal sandbox stderr: %s", stripped)

    await asyncio.gather(_drain_stdout(), _drain_stderr())

    returncode = await process.wait.aio()

    if returncode != 0:
        stderr_text = "\n".join(stderr_lines) or "(no stderr output)"
        logger.error(
            "Modal sandbox execution failed (exit %d):\n%s", returncode, stderr_text
        )
        raise RuntimeError(
            f"Sandbox execution failed (exit {returncode}):\n{stderr_text}"
        )


async def run_feature_engineering_code(
    preprocessed_csv: str,
    generated_code: str,
    *,
    on_stdout: Callable[[str], None] | None = None,
    on_stderr: Callable[[str], None] | None = None,
    timeout: float = 60.0,
) -> str:
    """Run feature engineering code in a Modal sandbox.

    Writes the preprocessed CSV to the input path, prepends df loading to generated
    code, executes in sandbox, and returns the output CSV content.

    Args:
        preprocessed_csv: CSV string (preprocessed dataframe).
        generated_code: Python code from LLM (operates on df, must save to output path).
        on_stdout: Optional sync callback for stdout lines (for WebSocket progress).
        on_stderr: Optional sync callback for stderr lines.
        timeout: Execution timeout in seconds.

    Returns:
        Output CSV string (enhanced dataframe).

    Raises:
        ValueError: If Modal credentials are not configured.
        RuntimeError: If code execution fails or output file is missing.
    """
    _ensure_modal_credentials()

    full_code = CODE_PREFIX + generated_code

    # Guarantee the save call is present even if LLM forgets
    if "to_csv" not in generated_code or OUTPUT_CSV_PATH not in generated_code:
        full_code = full_code.rstrip()
        if not full_code.endswith("\n"):
            full_code += "\n"
        full_code += f"\ndf.to_csv('{OUTPUT_CSV_PATH}', index=False)\n"

    sb = await _create_sandbox(timeout)
    try:
        await _exec_script(
            sb,
            full_code,
            {INPUT_CSV_PATH: preprocessed_csv.encode("utf-8")},
            timeout,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )
        output_bytes = await sb.filesystem.read_bytes.aio(OUTPUT_CSV_PATH)
        return output_bytes.decode("utf-8")
    finally:
        await sb.terminate.aio()
        await sb.detach.aio()


async def run_visualization_code(
    enhanced_csv: str,
    generated_code: str,
    *,
    timeout: float = 60.0,
) -> bytes:
    """Run visualization code in a Modal sandbox and return PNG bytes.

    Writes the enhanced CSV to the input path, prepends df loading to the generated
    code, executes it, and reads back the saved PNG file.

    Args:
        enhanced_csv: CSV string (feature-engineered dataframe).
        generated_code: Python code from LLM that must save a plot to OUTPUT_PNG_PATH.
        timeout: Execution timeout in seconds.

    Returns:
        PNG image as bytes.

    Raises:
        ValueError: If Modal credentials are not configured.
        RuntimeError: If code execution fails or output PNG is missing.
    """
    _ensure_modal_credentials()

    full_code = CODE_PREFIX + VIZ_FONT_PREFIX + generated_code

    # Guarantee the savefig call is present even if LLM forgets
    if OUTPUT_PNG_PATH not in generated_code:
        full_code = full_code.rstrip()
        if not full_code.endswith("\n"):
            full_code += "\n"
        full_code += (
            f"\nimport matplotlib.pyplot as plt\n"
            f"plt.savefig('{OUTPUT_PNG_PATH}', dpi=150, bbox_inches='tight')\n"
            f"plt.close()\n"
        )

    sb = await _create_sandbox(timeout)
    try:
        await _exec_script(
            sb,
            full_code,
            {INPUT_CSV_PATH: enhanced_csv.encode("utf-8")},
            timeout,
        )
        return await sb.filesystem.read_bytes.aio(OUTPUT_PNG_PATH)
    finally:
        await sb.terminate.aio()
        await sb.detach.aio()
