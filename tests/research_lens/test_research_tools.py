import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "skills" / "research-question-interlocutor" / "scripts"


def test_lexical_index_and_diversified_search(tmp_path):
    researcher = tmp_path / "researcher"
    index = researcher / "index"
    index.mkdir(parents=True)
    records = [
        {"chunk_id": "a:1", "work_id": "a", "title": "First", "page": 1, "text": "character networks and literary evidence"},
        {"chunk_id": "b:1", "work_id": "b", "title": "Second", "page": 4, "text": "character modeling requires explicit evidence"},
    ]
    (index / "chunks.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    subprocess.run([sys.executable, str(SCRIPTS / "build_index.py"), str(researcher)], check=True)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "search_corpus.py"), str(researcher), "character evidence"],
        check=True,
        text=True,
        capture_output=True,
    )
    matches = json.loads(result.stdout)
    assert {match["work_id"] for match in matches} == {"a", "b"}


def test_evidence_audit_rejects_uncited_documented_claim(tmp_path):
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps({"evidence_ledger": [{"label": "documented", "claim": "x"}]}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "audit_evidence.py"), str(session)],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "missing work_id" in result.stderr
