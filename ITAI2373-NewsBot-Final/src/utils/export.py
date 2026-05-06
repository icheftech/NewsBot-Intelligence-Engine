"""
Export utilities — save results to CSV and generate summary reports.
"""

import os
import json
import pandas as pd
from datetime import datetime


def save_dataframe(df: pd.DataFrame, path: str, fmt: str = 'csv'):
    """Save a DataFrame to disk.

    Args:
        df: DataFrame to save.
        path: Output file path.
        fmt: 'csv' or 'json'.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == 'csv':
        df.to_csv(path, index=False)
    elif fmt == 'json':
        df.to_json(path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported format '{fmt}'. Use 'csv' or 'json'.")
    print(f"Saved {len(df)} rows to {path}")


def export_topic_report(topic_modeler, output_path: str):
    """Write a plain-text topic report for LDA or NMF results.

    Args:
        topic_modeler: Fitted TopicModeler instance.
        output_path: Path for the .txt output file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    method = topic_modeler.method.upper()
    lines = [
        f"NewsBot 2.0 — {method} Topic Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Topics: {topic_modeler.n_topics}",
        "-" * 50,
    ]
    for i in range(topic_modeler.n_topics):
        words = topic_modeler.get_topic_words(i, n_words=10)
        lines.append(f"Topic {i+1:2d}: {' | '.join(words)}")
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Topic report saved to {output_path}")


def export_system_stats(stats_cache: dict, output_path: str):
    """Save the system stats cache as a JSON file.

    Args:
        stats_cache: Dict of pre-computed system statistics.
        output_path: Destination .json path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats_cache, f, indent=2, default=str)
    print(f"Stats exported to {output_path}")
