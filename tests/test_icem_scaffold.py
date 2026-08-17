from scripts.icem import tokenize_with_offsets
from scripts.icem.classifier import CachedHsdClassifier, LexiconHsdClassifier, _output_activation, _positive_label_index
from scripts.icem.context_rules import cue_token_indexes
from scripts.icem.datasets import _civil_comments_targets
from scripts.icem.importance import occlusion_importance, select_anchor_indexes
from scripts.icem.metrics import aggregate_variant_metrics
from scripts.icem.release_policy import release_icem_context, release_pii_mask, release_variant
from scripts.icem.schema import ImportanceScore, Row, Span
from scripts.icem.spans import detect_identifier_spans
from scripts.icem.synthetic_pii import inject_synthetic_pii


def test_tokenizer_preserves_offsets():
    text = "@mira said Muslims should be banned."
    tokens = tokenize_with_offsets(text)

    assert [token.text for token in tokens[:4]] == ["@mira", "said", "Muslims", "should"]
    assert text[tokens[0].start : tokens[0].end] == "@mira"


def test_synthetic_pii_records_gold_spans():
    injected = inject_synthetic_pii("immigrants should leave.", seed=17)

    assert injected.spans
    for span in injected.spans:
        assert injected.text[span.start : span.end]
        assert span.replacement


def test_icem_context_release_keeps_classifier_anchor_context():
    text = "I do not agree with people saying Muslims should be banned."
    tokens = tokenize_with_offsets(text)
    classifier = LexiconHsdClassifier(cues=("banned",))
    scores = occlusion_importance(text, tokens, classifier)
    anchors = select_anchor_indexes(scores, top_k=1, min_delta=0.01)

    result = release_icem_context(
        row_id="1",
        text=text,
        tokens=tokens,
        anchor_indexes=anchors,
        window_radius=2,
    )

    assert result.variant == "icem_context"
    assert "banned" in result.released_text
    assert "do not agree" in result.released_text


def test_anchor_selection_falls_back_to_best_non_pii_evidence_below_threshold():
    scores = (
        ImportanceScore(token_index=2, token_text="policy", delta=0.004, baseline_score=0.7, perturbed_score=0.696),
        ImportanceScore(token_index=5, token_text="slur", delta=0.009, baseline_score=0.7, perturbed_score=0.691),
    )

    anchors = select_anchor_indexes(scores, top_k=1, min_delta=0.02, token_count=12)

    assert anchors == (5,)


def test_fallback_cue_inventory_includes_obvious_slurs():
    text = "the kike billionaire violated that understanding"
    tokens = tokenize_with_offsets(text)

    anchors = cue_token_indexes(tokens, {"HARM_CUE"}, target_groups=("jewish",))

    assert tokens[anchors[0]].text == "kike"


def test_direct_and_quasi_identifier_masking():
    text = "Alex Mercer from Riverton High wrote: immigrants should be banned."
    spans = detect_identifier_spans(text)

    direct = release_pii_mask(row_id="1", text=text, identifier_spans=spans)
    quasi = release_pii_mask(row_id="1", text=text, identifier_spans=spans, include_quasi=True)

    assert "[PERSON]" in direct.released_text
    assert "Riverton High" in direct.released_text
    assert "[SCHOOL]" in quasi.released_text
    assert "immigrants should be banned" in quasi.released_text


def test_importance_variants_mask_overlapping_identifier_context():
    text = "Contact me at mira.example@test.invalid, those people should leave."
    tokens = tokenize_with_offsets(text)
    spans = detect_identifier_spans(text)
    classifier = LexiconHsdClassifier(cues=("leave",))
    scores = occlusion_importance(text, tokens, classifier, excluded_spans=spans)
    anchors = select_anchor_indexes(scores, top_k=1, min_delta=0.01, token_count=len(tokens))

    result = release_variant(
        variant="importance_window",
        row_id="1",
        text=text,
        tokens=tokens,
        anchor_indexes=anchors,
        identifier_spans=spans,
        window_radius=10,
        importance_scores=scores,
    )

    assert "mira.example@test.invalid" not in result.released_text
    assert "[EMAIL]" in result.released_text
    assert "leave" in result.released_text


def test_window_radius_controls_surrounding_context_without_unmasking_pii():
    text = "Alex Mercer wrote: those people should leave immediately."
    tokens = tokenize_with_offsets(text)
    spans = detect_identifier_spans(text)
    leave_index = next(token.index for token in tokens if token.text == "leave")

    narrow = release_variant(
        variant="importance_window",
        row_id="1",
        text=text,
        tokens=tokens,
        anchor_indexes=(leave_index,),
        identifier_spans=spans,
        window_radius=1,
    )
    wide = release_variant(
        variant="importance_window",
        row_id="1",
        text=text,
        tokens=tokens,
        anchor_indexes=(leave_index,),
        identifier_spans=spans,
        window_radius=3,
    )

    assert "should leave immediately" in narrow.released_text
    assert "those people should leave immediately" in wide.released_text
    assert "Alex Mercer" not in wide.released_text


def test_icem_context_expansion_masks_overlapping_identifier_context():
    text = "Alex Mercer said those people should leave immediately."
    tokens = tokenize_with_offsets(text)
    spans = detect_identifier_spans(text)
    leave_index = next(token.index for token in tokens if token.text == "leave")

    result = release_variant(
        variant="icem_context",
        row_id="1",
        text=text,
        tokens=tokens,
        anchor_indexes=(leave_index,),
        identifier_spans=spans,
        window_radius=5,
    )

    assert "Alex Mercer" not in result.released_text
    assert "[PERSON]" in result.released_text
    assert "leave" in result.released_text


def test_metrics_report_reduction_and_gold_pii_residuals():
    row = Row(
        row_id="1",
        dataset="synthetic",
        split="test",
        text="Alex Mercer said immigrants should leave.",
        label="hateful",
        binary_label=1,
        metadata={
            "synthetic_pii_spans": [
                {
                    "start": 0,
                    "end": len("Alex Mercer"),
                    "label": "PERSON",
                    "source": "synthetic",
                    "replacement": "[PERSON]",
                }
            ]
        },
    )
    raw = release_variant(
        variant="raw",
        row_id=row.row_id,
        text=row.text,
        tokens=tokenize_with_offsets(row.text),
        anchor_indexes=(4,),
        identifier_spans=(Span(0, len("Alex Mercer"), "PERSON", "synthetic", replacement="[PERSON]"),),
    )
    masked = release_variant(
        variant="pii_mask",
        row_id=row.row_id,
        text=row.text,
        tokens=tokenize_with_offsets(row.text),
        anchor_indexes=(4,),
        identifier_spans=(Span(0, len("Alex Mercer"), "PERSON", "synthetic", replacement="[PERSON]"),),
    )

    metrics = aggregate_variant_metrics(
        [row],
        [masked],
        released_scores=[0.9],
        raw_scores=[0.9],
    )

    assert raw.released_text == row.text
    assert metrics["direct_pii_residual_rate"] == 0.0
    assert metrics["retained_token_pct"] == 100.0


def test_cached_classifier_deduplicates_repeated_texts():
    class CountingClassifier:
        def __init__(self):
            self.calls = []

        def predict_proba(self, texts):
            self.calls.append(tuple(texts))
            return [0.9 if "hate" in text else 0.1 for text in texts]

    backend = CountingClassifier()
    classifier = CachedHsdClassifier(backend)

    assert classifier.predict_proba(["hate text", "clean text", "hate text"]) == [0.9, 0.1, 0.9]
    assert classifier.predict_proba(["clean text", "new hate text"]) == [0.1, 0.9]
    assert backend.calls == [("hate text", "clean text"), ("new hate text",)]


def test_civil_comments_lexical_target_detection():
    targets = _civil_comments_targets("Muslims, women, and Jewish neighbors discussed the policy.")

    assert targets == ("jewish", "muslim", "women")


def test_toxicity_multilabel_classifier_uses_sigmoid_toxicity_output():
    id2label = {0: "toxicity", 1: "severe_toxicity", 2: "obscene", 3: "identity_attack", 4: "insult"}

    assert _positive_label_index(id2label) == 0
    assert _output_activation(id2label, None) == "sigmoid"
