import json
import random

from ampred.benchmark import run


def _write(path, prefix, seqs):
    path.write_text("".join(f">{prefix}{i}\n{s}\n" for i, s in enumerate(seqs)))


def test_benchmark_runs_end_to_end_without_esm(tmp_path):
    rng = random.Random(0)
    def amp():
        return "".join(rng.choice("KRLLWIF") for _ in range(rng.randint(12, 30)))
    def dec():
        return "".join(rng.choice("DESTNQAGP") for _ in range(rng.randint(12, 30)))
    for short, n in (("tr", 40), ("eval", 20), ("te", 30)):
        _write(tmp_path / f"AMP.{short}.fa", "a", [amp() for _ in range(n)])
        _write(tmp_path / f"DECOY.{short}.fa", "d", [dec() for _ in range(n)])
    res = run(str(tmp_path), str(tmp_path / "out"), use_esm=False)
    assert res["n"] == {"train": 80, "eval": 40, "test": 60}
    # an easy synthetic task: composition models must be near-perfect
    assert res["models"]["logreg[aac]"]["auc"] > 0.95
    assert (tmp_path / "out" / "metrics.json").exists()
    json.loads((tmp_path / "out" / "metrics.json").read_text())
