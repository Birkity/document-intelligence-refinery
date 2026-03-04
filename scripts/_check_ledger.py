import json
from pathlib import Path

ledger = Path(".refinery/extraction_ledger.jsonl")
entries = [json.loads(line) for line in ledger.read_text().strip().split("\n")]
for e in entries:
    did = e["document_id"]
    strat = e["strategy_used"]
    conf = e["confidence_score"]
    esc = e["escalation_triggered"]
    print(f"  {did}  {strat:20s}  conf={conf:.2f}  escalated={esc}")
print(f"\nTotal: {len(entries)}")
