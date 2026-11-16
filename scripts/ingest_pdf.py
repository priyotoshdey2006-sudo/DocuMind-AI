import json
import sys
from pathlib import Path

# Allow running this script directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest import ingest_pdf


if len(sys.argv) != 2:
    raise SystemExit("Usage: python scripts/ingest_pdf.py data/your-paper.pdf")

records = ingest_pdf(sys.argv[1])

output_path = Path("data/chunks.json")
output_path.write_text(
    json.dumps(records, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Created {len(records)} chunks: {output_path}")
print("Sample:", records[0])