import json

from crelat.experiment import RunContext


def test_run_context_writes_complete_manifest(tmp_path):
    run = RunContext("smoke", {"seed": 1}, run_root=tmp_path)
    (run.path / "tables" / "value.txt").write_text("ok", encoding="utf-8")
    run.complete({"rows": 1})
    manifest = json.loads((run.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["metadata"]["rows"] == 1
    assert "tables/value.txt" in manifest["artifacts"]
