"""Build the offline report from its editable template and supplied assets.

Usage: python build.py --font /path/to/NotoSansCJKsc-Regular.otf \
    --triptych /path/to/rendered-triptych.png

Requires fonttools and Pillow. The triptych is rendered from existing saved
predictions with tools/render_validation_poses.py; see README for provenance.
"""

import argparse
import base64
from io import BytesIO
from pathlib import Path
import re

from fontTools import subset
from fontTools.ttLib import TTFont
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, required=True)
    parser.add_argument('--triptych', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    template = (root / 'report.template.html').read_text(encoding='utf-8')
    # Include all visible glyphs, punctuation and dynamically generated labels.
    text = re.sub(r'<[^>]+>', '', template)
    options = subset.Options()
    options.flavor = 'woff'
    options.name_IDs = ['*']
    options.name_legacy = True
    options.name_languages = ['*']
    font = TTFont(args.font)
    sub = subset.Subsetter(options=options)
    sub.populate(text=text)
    sub.subset(font)
    font.flavor = 'woff'
    encoded_font = BytesIO()
    font.save(encoded_font)
    # Lossless PNG, keeping the original renderer's text and arrows readable.
    image = Image.open(args.triptych).convert('RGB')
    # Remove only the renderer's metadata and repeated explanatory footnotes.
    # Keep all three view titles, geometry, platform, arrows and per-view notes.
    image = image.crop((0, 82, image.width, image.height - 54))
    encoded_image = BytesIO()
    image.save(encoded_image, format='PNG', optimize=True)
    page = template.replace('__REPORT_FONT__', base64.b64encode(encoded_font.getvalue()).decode())
    page = page.replace('__TRIPTYCH_IMAGE__', base64.b64encode(encoded_image.getvalue()).decode())
    if '__REPORT_FONT__' in page or '__TRIPTYCH_IMAGE__' in page:
        raise RuntimeError('Unresolved embedded asset')
    (root / 'index.html').write_text(page, encoding='utf-8')
    print(f'Built {root / "index.html"}: {len(page.encode()):,} bytes')


if __name__ == '__main__':
    main()
