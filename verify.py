#!/usr/bin/env python3
"""Check original code preservation, local assets/anchors, and embed JavaScript syntax."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import html
import json
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
WEB = ROOT / 'webapp'


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids, self.links = set(), []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids, f'Duplicate id: {attrs["id"]}'
            self.ids.add(attrs['id'])
        for attr in ('src', 'href'):
            if attrs.get(attr):
                self.links.append(attrs[attr])


def main():
    pages = {p: Page(p.read_text()) for p in WEB.rglob('*.html')}
    checked = 0
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            assert target.exists(), f'Missing file: {path.name}: {link}'
            if url.fragment and target in pages:
                assert unquote(url.fragment) in pages[target].ids, f'Missing anchor: {path.name}: {link}'
            checked += 1
    originals = '\n'.join(p.read_text() for p in (ROOT / 'src/app/src/content/chapters').glob('*.mdx'))
    code = re.findall(r'^```[^\n]*\n([\s\S]*?)^```', originals, re.M)
    rendered = ''.join(p.read_text() for p in (WEB / 'chapters').glob('*.html'))
    rendered_code = [html.unescape(re.sub('<[^>]+>', '', x)) for x in re.findall(r'<pre><code[^>]*>([\s\S]*?)</code></pre>', rendered)]
    assert len(code) == len(rendered_code) == 8
    assert sorted(code) == sorted(rendered_code), 'Original code differs from rendered code'
    assert all(token not in rendered for token in ('<Embed ', '<Code ', '<SourceImage', '<HarborVisualizer', '[@'))
    scripts_checked = 0
    with tempfile.TemporaryDirectory() as temporary:
        script = Path(temporary) / 'check.js'
        for path in (WEB / 'embeds').glob('*.html'):
            for js in re.findall(r'<script>([\s\S]*?)</script>', path.read_text()):
                script.write_text(js)
                result = subprocess.run(['node', '--check', str(script)], capture_output=True, text=True)
                assert result.returncode == 0, f'Invalid JavaScript: {path.name}: {result.stderr}'
                scripts_checked += 1
    result = dict(html_pages=len(pages), local_links_and_assets=checked, exact_code_blocks=len(code), embed_scripts=scripts_checked)
    (ROOT / 'qa/verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
