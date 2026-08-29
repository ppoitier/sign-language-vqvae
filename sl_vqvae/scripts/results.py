from pathlib import Path

import numpy as np


def save_results(results: dict, filepath: str) -> None:
    file_path = Path(filepath)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, results, allow_pickle=True)
