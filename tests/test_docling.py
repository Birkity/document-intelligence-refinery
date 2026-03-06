"""Quick test to verify Docling integration."""
import logging
from src.strategies.layout import LayoutExtractor

logging.basicConfig(level=logging.INFO)

# Initialize the extractor
extractor = LayoutExtractor()

# Check if docling is detected
print("=" * 60)
print(f"Docling installed and detected: {extractor._has_docling}")
print(f"Strategy name: {extractor.strategy_name}")
print("=" * 60)

if extractor._has_docling:
    print("\n✓ Docling is available and will be used for layout extraction!")
    print("  This provides:")
    print("  - Better table detection and parsing")
    print("  - Multi-column layout handling")
    print("  - Figure and caption extraction")
    print("  - Reading order preservation")
else:
    print("\n✗ Docling not available - using pdfplumber fallback")

print("\nTo test a full extraction with Docling, run:")
print('  python -m src.cli run "data/CBE ANNUAL REPORT 2023-24.pdf" --sample-pages 3')
