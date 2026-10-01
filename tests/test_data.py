import pandas as pd

from txt2onto.data import build_text, clean_field, is_placeholder, normalize, text_hash

TEXT_CFG = {
    "fields": ["host_comments", "isolation_comments", "comments"],
    "prefixes": {"host_comments": "host", "isolation_comments": "isolation", "comments": "comments"},
    "separator": " | ",
    "placeholder_tokens": ["missing", "not collected", "not applicable"],
}
PH = {"missing", "not collected", "not applicable"}


def test_clean_field_drops_placeholders_and_blanks():
    assert clean_field("ATCC:27758; missing", PH) == "ATCC:27758"
    assert clean_field("Missing; ; not collected", PH) == ""
    assert clean_field(float("nan"), PH) == ""
    assert clean_field(None, PH) == ""


def test_build_text_skips_empty_fields():
    df = pd.DataFrame({
        "host_comments": [None, "adult male"],
        "isolation_comments": ["blood; missing", "stool"],
        "comments": ["not applicable", None],
    })
    assert build_text(df, TEXT_CFG).tolist() == ["isolation: blood", "host: adult male | isolation: stool"]


def test_normalize_and_hash_ignore_case_and_whitespace():
    assert normalize("  Blood   Culture ") == "blood culture"
    assert text_hash(normalize("Blood  culture")) == text_hash(normalize("blood culture"))


def test_is_placeholder():
    assert is_placeholder("Not Applicable", PH)
    assert is_placeholder(float("nan"), PH)
    assert not is_placeholder("Bos taurus", PH)
