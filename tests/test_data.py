from ampred.data import BASE_URL, COMMIT, Split, is_valid_peptide, load_split, parse_fasta


def test_parse_single_record():
    assert parse_fasta(">a\nACDE\n") == [("a", "ACDE")]


def test_parse_multiline_and_lowercase():
    recs = parse_fasta(">x desc here\nacd\nEFG\n>y\nKK\n")
    assert recs == [("x", "ACDEFG"), ("y", "KK")]


def test_parse_ignores_blank_lines():
    assert parse_fasta("\n>a\n\nAC\n\n") == [("a", "AC")]


def test_valid_peptide():
    assert is_valid_peptide("ACDEFGHIKLMNPQRSTVWY")
    assert not is_valid_peptide("ACDX")
    assert not is_valid_peptide("")


def test_url_is_pinned_to_commit():
    assert COMMIT in BASE_URL and len(COMMIT) == 40


def test_load_split_labels(tmp_path):
    (tmp_path / "AMP.tr.fa").write_text(">p1\nKKLLKK\n>p2\nGIGKFL\n")
    (tmp_path / "DECOY.tr.fa").write_text(">d1\nMSTEDA\n>bad\nMSXB\n")
    s = load_split("train", tmp_path)
    assert isinstance(s, Split)
    assert s.labels == [1, 1, 0]  # invalid decoy dropped
    assert s.ids == ["p1", "p2", "d1"]
    assert len(s) == 3
