"""
Visualization utilities — reusable plotting functions for NewsBot 2.0.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config.settings import CATEGORY_PALETTE


def plot_category_distribution(df: pd.DataFrame, save_path: str = None):
    """Bar chart of article counts per category."""
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = df['category'].value_counts()
    ax.bar(counts.index, counts.values,
           color=[CATEGORY_PALETTE.get(c, '#888') for c in counts.index])
    ax.set_title('Article Count by Category', fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sentiment_by_category(df: pd.DataFrame, save_path: str = None):
    """Grouped bar chart of positive / negative / neutral sentiment by category."""
    sent_cols = ['sent_pos', 'sent_neg', 'sent_neu']
    available = [c for c in sent_cols if c in df.columns]
    means = df.groupby('category')[available].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    means.plot(kind='bar', ax=ax, color=['#2ca02c', '#d62728', '#aec7e8'][:len(available)])
    ax.set_title('Sentiment Dimensions by Category', fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Mean Score')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(['Positive', 'Negative', 'Neutral'][:len(available)])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_topic_distribution(df: pd.DataFrame, save_path: str = None):
    """Stacked bar of dominant topic distribution per category."""
    if 'dominant_topic' not in df.columns:
        raise ValueError("DataFrame must have a 'dominant_topic' column.")
    topic_cat = df.groupby(['category', 'dominant_topic']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    topic_cat.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', legend=False)
    ax.set_title('Topic Distribution by Category', fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Article Count')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_compression_stats(summary_df: pd.DataFrame, save_path: str = None):
    """Bar chart of compression percentage by category."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary_df['category'], summary_df['compression_pct'],
           color=[CATEGORY_PALETTE.get(c, '#888') for c in summary_df['category']])
    ax.set_title('Summarization Compression % by Category', fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Compression %')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_language_distribution(lang_counts: dict, save_path: str = None):
    """Donut chart of detected language distribution."""
    top = list(lang_counts.items())[:6]
    labels = [f'{l} ({v})' for l, v in top]
    values = [v for _, v in top]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct='%1.1f%%',
           startangle=90, colors=plt.cm.Set2.colors[:len(top)])
    ax.set_title('Language Detection Distribution', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
