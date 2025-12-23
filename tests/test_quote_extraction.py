import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
for path in (PROJECT_PARENT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from framing_analyzer import AnalyzerConfig, TextProcessor, StructureZoneExtractor, FragmentGenerator


def test_quotes_detection_and_fragment_generation():
    config = AnalyzerConfig()
    processor = TextProcessor(config)
    zone_extractor = StructureZoneExtractor(config)
    fragment_generator = FragmentGenerator(config)

    article = (
        'He said, “Curly quote example.” Another line without quotes. '
        'She added, "Straight quote is here." End of story.'
    )

    processed = processor.process_article(article, title="Title")
    article_with_zones = zone_extractor.divide_into_zones(processed)

    quotes = article_with_zones.zones.get("quotes", [])
    assert len(quotes) >= 2  # 弯引号和直引号至少各有一次被识别

    fragments = fragment_generator.create_fragments(article_with_zones)
    quote_fragments = [f for f in fragments if f["zone"] == "quotes"]
    assert len(quote_fragments) == len(quotes)
