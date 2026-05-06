# NewsBot Intelligence Engine 2.0 — API Reference

## src.data_processing.text_preprocessor

### `preprocess(text, return_tokens=False)`
Clean, tokenize, and lemmatize a single text string.

**Args:**
- `text` (str): Raw article text
- `return_tokens` (bool): If True, return list of tokens instead of joined string

**Returns:** `str` or `list[str]`

### `preprocess_corpus(texts, return_tokens=False)`
Apply `preprocess()` to an iterable of texts.

**Returns:** `list[str]` or `list[list[str]]`

---

## src.data_processing.feature_extractor

### `fit_tfidf(clean_texts)`
Fit a TF-IDF vectorizer and transform the corpus.

**Returns:** `(TfidfVectorizer, sparse matrix)`

### `fit_count(clean_texts)`
Fit a count vectorizer for topic modeling.

**Returns:** `(CountVectorizer, sparse DTM)`

---

## src.data_processing.data_validator

### `validate_dataframe(df)`
Run quality checks on a BBC news DataFrame.

**Returns:** `dict` with keys `passed` (bool), `issues` (list), `stats` (dict)

### `clean_dataframe(df)`
Drop nulls, filter to valid categories, normalise column types.

**Returns:** cleaned `pd.DataFrame`

---

## src.analysis.classifier.NewsClassifier

### `fit(clean_texts, labels, vectorizer=None)`
Fit the Logistic Regression classifier.

### `predict(texts)`
Predict category labels. **Returns:** `np.ndarray`

### `predict_proba(texts)`
Probability distribution over categories. **Returns:** 2-D `np.ndarray`

### `evaluate(clean_texts, labels)`
Evaluate on a held-out set. **Returns:** `dict` with `accuracy` and `report`

### `train_test_evaluate(tfidf_matrix, labels, test_size=0.2)`
Split, fit, and evaluate in one call. **Returns:** `dict`

---

## src.analysis.sentiment_analyzer

### `score_text(text)`
Return VADER compound, pos, neg, neu scores and label.

**Returns:** `dict` with keys `sent_compound`, `sent_pos`, `sent_neg`, `sent_neu`, `sent_label`

### `score_corpus(texts)`
Apply `score_text()` to an iterable.

**Returns:** `pd.DataFrame`

### `sentiment_by_category(df)`
Mean sentiment scores grouped by category.

**Returns:** `pd.DataFrame`

---

## src.analysis.topic_modeler.TopicModeler

### `__init__(n_topics=10, method='lda')`
- `method`: `'lda'` or `'nmf'`

### `fit_transform(dtm, vectorizer)`
Train the topic model. **Returns:** `(n_docs, n_topics)` array

### `get_topic_words(topic_id, n_words=10)`
Top words for a topic. **Returns:** `list[str]`

### `assign_dominant_topics(topic_dist)`
Add dominant topic columns. **Returns:** `pd.DataFrame`

### `visualize_topics()`
Render pyLDAvis (LDA only). **Returns:** pyLDAvis PreparedData

### `compare_methods(dtm, vectorizer)`
Fit both LDA and NMF and return top words. **Returns:** `dict`

---

## src.language_models.summarizer

### `summarize(text, max_length=130, min_length=30)`
DistilBART abstractive summary. **Returns:** `str`

### `summarize_with_stats(text)`
Summary + compression statistics. **Returns:** `dict`

---

## src.language_models.embeddings.SemanticSearchIndex

### `build(df)`
Encode articles into the embedding index. **Returns:** `self`

### `search(query, top_k=5)`
Cosine similarity retrieval. **Returns:** `list[dict]`

---

## src.multilingual.language_detector

### `detect_language(text)`
ISO 639-1 language code. **Returns:** `str` (e.g. `'en'`, `'es'`, `'unknown'`)

### `language_distribution(texts)`
Language frequency count across corpus. **Returns:** `dict`

---

## src.multilingual.translator

### `translate_to_english(text)`
Translate to English. **Returns:** `str`

### `translate_and_detect(text)`
Detect + translate. **Returns:** `dict` with `source_lang`, `translated`, `error`

---

## src.conversation.query_processor.QueryProcessor

### `process(user_input, history=None)`
Route query through intent pipeline. **Returns:** formatted markdown `str`

---

## src.conversation.intent_classifier

### `classify_intent(query)`
Detect intent from query. **Returns:** `str` — one of `classify`, `summarize`, `search`, `translate`, `stats`, `topics`, `help`, `fallback`

### `extract_content(query, intent)`
Extract article/query content after command prefix. **Returns:** `str`

---

## src.utils.evaluation

### `full_classification_report(y_true, y_pred, labels=None)`
Accuracy + sklearn report. **Returns:** `dict`

### `cross_validate_classifier(model, X, y, cv=5)`
Stratified k-fold CV. **Returns:** `dict` with `mean_accuracy`, `std`, `scores`

### `compare_classifiers(classifiers, X_train, X_test, y_train, y_test)`
Fit and compare multiple models. **Returns:** `pd.DataFrame`
