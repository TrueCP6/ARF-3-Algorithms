"""Pull the figures, animations and printed results out of the exported notebooks in docs/.

`marimo export html` stores every cell's output (and console output) in a JSON session embedded in the
page. Reading the README's figures back out of that session guarantees they come from exactly the same
run as the HTML they are shown next to, rather than from a separate (and, for RL, different) rerun.

    python scripts/extract_figures.py

writes docs/img/<Notebook>_<name>.{png,gif} and prints each notebook's console output.
"""
import base64
import json
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
IMG = DOCS / "img"

# Names given to the PNG figures of each notebook, in the order they appear on the page
PNG_NAMES = {
    "Acrobot_A2C": ["training"],
    "DQN": ["training"],
    "PPO_v2": ["training"],
}

DATA_URL = re.compile(r"data:image/(png|gif|jpeg);base64,([A-Za-z0-9+/=]+)")


def load_session(html_path):
    html = html_path.read_text()
    start = html.index("{", html.index('"session":'))
    session, _ = json.JSONDecoder().raw_decode(html[start:])
    return session


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    for notebook, png_names in PNG_NAMES.items():
        session = load_session(DOCS / f"{notebook}.html")
        png_names = list(png_names)
        print(f"===== {notebook}")
        for cell in session["cells"]:
            for output in cell["outputs"]:
                data = output.get("data")
                if not isinstance(data, dict):
                    continue
                for payload in data.values():
                    text = payload if isinstance(payload, str) else json.dumps(payload)
                    for kind, b64 in DATA_URL.findall(text):
                        # The Cart-Pole photo in DQN's last cell is a jpeg, not a result
                        if kind == "jpeg":
                            continue
                        name = "policy" if kind == "gif" else png_names.pop(0)
                        path = IMG / f"{notebook}_{name}.{kind}"
                        path.write_bytes(base64.b64decode(b64))
                        print(f"wrote {path.relative_to(DOCS.parent)} ({path.stat().st_size / 1e3:.0f} kB)")
            for line in cell["console"]:
                if line.get("name") == "stdout":
                    print("  stdout:", line["text"].rstrip().replace("\n", "\n          "))


if __name__ == "__main__":
    main()
