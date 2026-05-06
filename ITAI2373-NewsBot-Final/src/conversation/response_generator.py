"""
Response generation — formats structured responses for each intent.
"""

from src.conversation.intent_classifier import classify_intent, extract_content


def format_classify_response(category: str, proba_pairs: list, sentiment: dict) -> str:
    sent_label = (
        'Positive 😊' if sentiment['sent_compound'] >= 0.05
        else 'Negative 😞' if sentiment['sent_compound'] <= -0.05
        else 'Neutral 😐'
    )
    lines = ['**Classification Result**\n',
             f'📁 **Category:** {category.upper()}',
             '📊 **Confidence scores:**']
    for cls, prob in proba_pairs[:3]:
        bar = '█' * int(prob * 20)
        lines.append(f'  {cls:15s} {bar} {prob:.1%}')
    lines.append(f'\n💬 **Sentiment:** {sent_label} (score: {sentiment["sent_compound"]:.3f})')
    return '\n'.join(lines)


def format_summary_response(stats: dict) -> str:
    return (
        f'**Summary** ({stats["original_words"]} → {stats["summary_words"]} words, '
        f'{stats["compression_pct"]}% compression)\n\n{stats["summary"]}'
    )


def format_search_response(query: str, results: list) -> str:
    lines = [f'**Semantic Search Results** for: "{query}"\n']
    for i, r in enumerate(results, 1):
        lines.append(f'**{i}.** [{r["category"].upper()}] — similarity: {r["score"]}')
        lines.append(f'{r["snippet"][:180]}...\n')
    return '\n'.join(lines)


def format_translate_response(result: dict) -> str:
    return (
        f'**🌐 Translation Result**\n\n'
        f'- **Detected language:** {result["source_lang"]}\n'
        f'- **English translation:** {result["translated"][:300]}\n'
        f'- **Predicted category:** {result.get("predicted_category", "N/A").upper()}\n'
        f'- **Sentiment:** {result.get("sentiment_compound", 0):.3f}'
    )


def format_stats_response(stats_cache: dict) -> str:
    lines = ['**📊 NewsBot 2.0 — System Statistics**\n',
             f'- **Total articles indexed:** {stats_cache["total_articles"]:,}',
             f'- **Classification accuracy:** {stats_cache["accuracy"]:.1%}\n',
             '**Category Breakdown:**']
    for cat, cnt in stats_cache['category_counts'].items():
        sent = stats_cache['avg_sentiment'].get(cat, 0)
        words = stats_cache['avg_words'].get(cat, 0)
        icon = '📈' if sent >= 0.05 else ('📉' if sent <= -0.05 else '➡️')
        lines.append(
            f'  - **{cat.title()}:** {cnt} articles | avg sentiment {sent:.3f} {icon} | avg {words} words'
        )
    return '\n'.join(lines)


def format_topics_response(top_topics: dict) -> str:
    lines = ['**🔑 Top Keywords by Category:**\n']
    for cat, words in top_topics.items():
        lines.append(f'**{cat.title()}:** {" · ".join(words)}')
    return '\n'.join(lines)


HELP_TEXT = (
    '**🤖 NewsBot 2.0 — What I can do:**\n\n'
    '1. **classify:** [article text] → Predicts category + sentiment\n'
    '2. **summarize:** [article text] → Abstractive summary\n'
    '3. **search:** [topic or question] → Finds semantically similar articles\n'
    '4. **translate:** [foreign text] → Translates + classifies\n'
    '5. **stats** → System performance & category breakdown\n'
    '6. **topics** → Top keywords per category\n\n'
    'Try: `classify: The Prime Minister announced new policies today...`'
)
