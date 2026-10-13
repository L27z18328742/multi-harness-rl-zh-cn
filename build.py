#!/usr/bin/env python3
"""Build the Chinese reading site from editable Markdown and a pinned source snapshot."""
from pathlib import Path
import html
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src/app/src/content'
WEB = ROOT / 'webapp'
SOURCE_URL = 'https://huggingface.co/spaces/FineEnvs/multi-harness-rl'
CHAPTERS = [
    ('introduction', '导论', '导论', 'Introduction', '模型相同，harness 不同，结果为何不同？'),
    ('basics', '第 1 章', '基础知识', 'The basics', 'harness、agentic RL、白盒与黑盒，以及精确 token。'),
    ('why-multi-harness', '第 2 章', '为什么要跨 harness 训练', 'Why train across harnesses', '分数反转、接口过拟合与前沿模型的训练实践。'),
    ('openenv', '第 3 章', '框架如何运作', 'How the framework works', 'OpenEnv 采集代理、Harbor 沙箱和 TRL 训练器。'),
    ('training', '第 4 章', '跨 harness 训练小模型', 'Training small models across harnesses', 'LFM 与 Qwen 实验、工具效率奖励，以及 SFT 对比。'),
    ('conclusions', '结语', '结论', 'Conclusions', '结果、实验局限与后续方向。'),
]
AUTHORS = 'Adithya S Kolavi、Joel Niklaus、Sergio Paniego Blanco、Leonie Monigatti、Amine Dirhoussi、Ben Burtenshaw、Lewis Tunstall、Leandro von Werra'
UI = dict(line.split('\t', 1) for line in (ROOT / 'content/ui.zh.tsv').read_text().splitlines() if line.strip())
FOOT = f'非官方简体中文译本 · 原作：<a href="{SOURCE_URL}">Hugging Face / FineEnvs</a>（Hugging Face、Liquid AI） · <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a> · 本译本修改了语言与阅读布局；第三方图像及标志沿用原署名。'


def slug(title):
    return re.sub(r'[^a-z0-9 -]', '', title.lower()).replace(' ', '-')


def bib_entries():
    entries = {}
    for block in re.split(r'(?=^@)', (SRC / 'bibliography.bib').read_text(), flags=re.M):
        match = re.match(r'@\w+\{([^,]+),', block)
        if not match:
            continue
        def field(name):
            start = re.search(r'\b' + name + r'\s*=\s*\{', block)
            if not start:
                return ''
            depth, end = 1, start.end()
            while depth and end < len(block):
                depth += (block[end] == '{') - (block[end] == '}')
                end += 1
            return re.sub(r'\s+', ' ', block[start.end():end - 1]).replace('{', '').replace('}', '')
        url = field('url')
        if not url:
            link = re.search(r'https?://[^}\s]+', block)
            url = link[0] if link else ''
        entries[match[1]] = dict(title=field('title'), author=field('author'), year=field('year'), url=url)
    return entries


def render_markdown(md):
    return subprocess.run(
        ['node', '-e', 'const fs=require("fs"),m=require(process.argv[1]);process.stdout.write(m.parse(fs.readFileSync(0,"utf8")));', str(WEB / 'assets/marked.min.js')],
        input=md, text=True, check=True, capture_output=True).stdout


def make_embed(name, config, output):
    fragment = (SRC / 'embeds' / name).read_text()
    # Translate complete display sentences before SVG word wrapping splits them.
    # Short technical keys stay untouched; runtime localization handles short labels.
    for en, zh in UI.items():
        if len(en) < 30 or en.count(' ') < 3:
            continue
        for quote in ('"', "'", '`'):
            escaped = en.replace('\\', '\\\\').replace(quote, '\\' + quote)
            translated = zh.replace('\\', '\\\\').replace(quote, '\\' + quote)
            fragment = fragment.replace(quote + escaped + quote, quote + translated + quote)
    # Static prose inside HTML template strings must also be translated before wrapping.
    for en, zh in sorted(UI.items(), key=lambda item: -len(item[0])):
        if len(en) >= 25 and en.count(' ') >= 3 and not re.search(r'''["'`]''', en + zh):
            fragment = fragment.replace(en, zh)
    # Relative paths keep the site usable under a subdirectory as well as at localhost.
    fragment = re.sub(r'''(["'`])/(data|logos|papers)/''', r'\1../assets/\2/', fragment)
    # The first fallback already resolves; remove the original relative fallbacks.
    fragment = fragment.replace("'assets/data/'+f, '../assets/data/'+f", "'../assets/data/'+f")
    cfg = html.escape(json.dumps(config, ensure_ascii=False), quote=True)
    wrapper = f'''<!doctype html><html lang="zh-Hans"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="../assets/style.css"><link rel="stylesheet" href="../assets/reading.css">
<style>html{{scroll-behavior:auto}}body{{background:var(--panel);margin:0;font-size:14px;line-height:1.6;padding:0 2px}}img{{max-width:100%}}svg{{font-family:var(--sans)}}[data-config]{{width:100%;display:flow-root}}</style>
<script src="../assets/color-palettes.js"></script><script src="../assets/i18n.js"></script></head>
<body><div data-config="{cfg}">{fragment}</div><script src="../assets/embed-size.js"></script></body></html>'''
    (WEB / 'embeds' / output).write_text(wrapper)


def expand(chapter, md, report):
    source = (SRC / 'chapters' / f'{chapter}.mdx').read_text()
    code = re.findall(r'^```[^\n]*\n[\s\S]*?^```', source, re.M)
    original_embeds = re.findall(r'<HtmlEmbed\b[\s\S]*?/>', source)
    src_heads = re.findall(r'^(#{2,4}) (.+)$', source, re.M)
    dst_heads = re.findall(r'^(#{2,4}) (.+)$', md, re.M)
    assert [x[0] for x in src_heads] == [x[0] for x in dst_heads], f'Heading coverage: {chapter}'
    codes_seen, embeds_seen = [], []
    def code_replace(match):
        n = int(match[1]); codes_seen.append(n)
        return code[n]
    md = re.sub(r'<Code n="(\d+)"\s*/>', code_replace, md)
    def embed_replace(match):
        tag = match[0]
        n = int(re.search(r'n="(\d+)"', tag)[1]); embeds_seen.append(n)
        original = original_embeds[n]
        name = re.search(r'src="([^"]+)"', original)[1]
        config_match = re.search(r'config=\{(\{[\s\S]*?\})\}', original)
        config = {}
        if config_match:
            config_text = re.sub(r'([{,]\s*)(\w+)\s*:', r'\1"\2":', config_match[1])
            config = json.loads(config_text)
        if 'note' in config:
            config['note'] = '上图：全部 1,000 个测试单元。下图：各 harness。独立 OpenCode 实验有单独测得的 15.9% 基线。'
        output = f'{chapter}-{n:02d}.html'
        make_embed(name, config, output)
        title_match = re.search(r'title="([^"]*)"', tag)
        desc_match = re.search(r'desc="([^"]*)"', tag)
        title = title_match[1] if title_match else '交互图示'
        desc = desc_match[1] if desc_match else ''
        eid = name.removesuffix('.html')
        if any(re.search(r'src="([^"]+)"', earlier)[1] == name for earlier in original_embeds[:n]):
            eid += f'-{n}'
        return f'<figure id="{eid}"><div class="fig-title">{html.escape(title)}</div><iframe class="viz-frame" src="../embeds/{output}" title="{html.escape(title)}" loading="lazy"></iframe><figcaption>{html.escape(desc)}</figcaption></figure>\n'
    md = re.sub(r'<Embed\b[^>]*?/>', embed_replace, md)
    assert codes_seen == list(range(len(code))), f'Code coverage: {chapter}'
    assert embeds_seen == list(range(len(original_embeds))), f'Embed coverage: {chapter}'
    md = md.replace('<SourceImage />', '<figure><img src="../assets/images/joel-harness-pairing.png" alt="GLM-5.2 与 Gemma 4 26B-A4B 在 SWE-bench Pro 上使用不同编码 harness 时，pass@1 与每任务成本的比较"><figcaption>SWE-bench Pro 上，两个模型在不同编码 harness 下的 pass@1 与每任务成本。图作者：<a href="https://x.com/joelniklaus/status/2085725862142623875">Joel Niklaus</a>。图中标签保留原文。</figcaption></figure>')
    md = md.replace('<HarborVisualizer />', '<iframe class="external-demo" src="https://huggingfaceh4-harbor-visualiser.hf.space/?dataset=FineEnvs/SmolDataEnvs-harbor-train" title="Harbor 可视化工具：SmolDataEnvs 训练集" loading="lazy"></iframe>\n\n<p class="footnote">此在线任务浏览器需要网络；<a href="https://huggingface.co/spaces/HuggingFaceH4/harbor-visualiser">单独打开</a>。</p>')
    report[chapter] = dict(headings=len(src_heads), code_blocks=len(code), embeds=len(original_embeds))
    return md, [(slug(en), zh) for (_, en), (_, zh) in zip(src_heads, dst_heads)]


def toc(heads):
    return '<ol>' + ''.join(f'<li><a href="#{sid}">{html.escape(title)}</a></li>' for sid, title in heads) + '</ol>'


def page(title, body, heads, prefix='../', nav='', pager=''):
    return f'''<!doctype html><html lang="zh-Hans"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} — 多 harness RL 中文版</title><link rel="stylesheet" href="{prefix}assets/style.css"><link rel="stylesheet" href="{prefix}assets/reading.css"></head><body>
<header class="topbar"><a class="brand" href="{prefix}index.html">多 harness RL 中文版<span>在真实智能体框架中训练模型</span></a><nav class="chapnav">{nav}<a href="{prefix}full.html">全文</a></nav></header>
<div class="layout"><aside class="sidebar"><nav class="toc"><h2>阅读目录</h2>{toc(heads)}</nav></aside><main class="main"><details class="mobile-toc"><summary>展开阅读目录</summary>{toc(heads)}</details><div class="chapter-source">原文发布：2026 年 10 月 1 日 · 非官方译本 · <a href="{SOURCE_URL}">查看原文</a></div><article class="prose" id="content">{body}</article><nav class="pager">{pager}</nav><footer class="pagefoot">{FOOT}</footer></main></div><script src="{prefix}assets/reading.js"></script></body></html>'''


def main():
    for folder in ('chapters', 'embeds', 'assets/data', 'assets/images', 'assets/logos', 'assets/papers'):
        (WEB / folder).mkdir(parents=True, exist_ok=True)
    for source, target in [(SRC / 'assets/data', WEB / 'assets/data'), (SRC / 'assets/image', WEB / 'assets/images'), (ROOT / 'src/app/public/logos', WEB / 'assets/logos'), (ROOT / 'src/app/public/papers', WEB / 'assets/papers')]:
        shutil.copytree(source, target, dirs_exist_ok=True)
    shutil.copy(ROOT / 'src/app/public/scripts/color-palettes.js', WEB / 'assets/color-palettes.js')
    dictionary = dict(line.split('\t', 1) for line in (ROOT / 'content/ui.zh.tsv').read_text().splitlines() if line.strip())
    (WEB / 'assets/i18n.js').write_text((ROOT / 'i18n-template.js').read_text().replace('__DICTIONARY__', json.dumps(dictionary, ensure_ascii=False)))
    entries = bib_entries()
    cited = []
    for cid, *_ in CHAPTERS:
        for key in re.findall(r'@([a-zA-Z][\w-]+)', (ROOT / 'content' / f'{cid}.md').read_text()):
            if key not in cited:
                cited.append(key)
    missing = set(cited) - entries.keys()
    assert not missing, f'Missing citations: {missing}'
    references = '<h2 id="references">参考文献</h2><ol>' + ''.join(
        f'<li id="ref-{key}"><a href="{html.escape(entries[key]["url"])}">{html.escape(entries[key]["title"])}</a> · {html.escape(entries[key]["author"])} · {entries[key]["year"]}</li>' for key in cited) + '</ol>'
    (WEB / 'chapters/references.html').write_text(page('参考文献', references, [('references', '参考文献')]))
    anchor_chapter = {}
    for cid, *_ in CHAPTERS:
        for heading in re.findall(r'^#{2,4} (.+)$', (SRC / 'chapters' / f'{cid}.mdx').read_text(), re.M):
            anchor_chapter[slug(heading)] = cid
        for fname in re.findall(r'src="([^"/]+)\.html"', (SRC / 'chapters' / f'{cid}.mdx').read_text()):
            anchor_chapter[fname] = cid
    report, expanded, all_heads, full_parts = {}, [], [], []
    for i, (cid, no, title, _, _) in enumerate(CHAPTERS):
        md, heads = expand(cid, (ROOT / 'content' / f'{cid}.md').read_text(), report)
        md_heads = iter(heads)
        expanded.append(re.sub(r'^(#{2,4} .+)$', lambda m: f'<a id="{next(md_heads)[0]}"></a>\n\n{m[0]}', md, flags=re.M))
        # Citations retain stable keys in HTML and link to a shared references page.
        cited_md = re.sub(r'\[@([\w-]+)\]', lambda m: f'<a class="citation" href="references.html#ref-{m[1]}">[{cited.index(m[1]) + 1}]</a>', md)
        cited_md = re.sub(r'(?<![\w/])@([a-zA-Z][\w-]+)', lambda m: f'<a class="citation" href="references.html#ref-{m[1]}">[{cited.index(m[1]) + 1}]</a>', cited_md)
        body = render_markdown(cited_md)
        hiter = iter(heads)
        body = re.sub(r'<h([234])>(.*?)</h\1>', lambda m: f'<h{m[1]} id="{next(hiter)[0]}">{m[2]}</h{m[1]}>', body)
        full_parts.append(body.replace('href="references.html#', 'href="#').replace('src="../', 'src="'))
        all_heads.extend(heads)
        body = re.sub(r'href="#([^\"]+)"', lambda m: f'href="{anchor_chapter[m[1]]}.html#{m[1]}"' if m[1] in anchor_chapter and anchor_chapter[m[1]] != cid else m[0], body)
        body = body.replace('href="/data/', 'href="../assets/data/')
        prev = CHAPTERS[i - 1] if i else None
        nxt = CHAPTERS[i + 1] if i + 1 < len(CHAPTERS) else None
        nav = (f'<a href="{prev[0]}.html">← 上一章</a>' if prev else '') + f'<span class="chapnav-current">{no}</span>' + (f'<a href="{nxt[0]}.html">下一章 →</a>' if nxt else '')
        pager = (f'<a href="{prev[0]}.html"><span class="dir">← 上一章</span><div class="ttl">{prev[2]}</div></a>' if prev else '<span></span>') + (f'<a class="next" href="{nxt[0]}.html"><span class="dir">下一章 →</span><div class="ttl">{nxt[2]}</div></a>' if nxt else '')
        (WEB / 'chapters' / f'{cid}.html').write_text(page(title, body, heads, nav=nav, pager=pager))
    full_body = '\n'.join(full_parts).replace('href="/data/', 'href="assets/data/') + references
    (WEB / 'full.html').write_text(page('全文阅读', full_body, all_heads + [('references', '参考文献')], prefix=''))
    cards = ''.join(f'<a class="card" href="chapters/{cid}.html"><span class="no">{no}</span><span class="zh">{title}</span><span class="en">{en}</span><span class="desc">{desc}</span></a>' for cid, no, title, en, desc in CHAPTERS)
    embeds = sum(c['embeds'] for c in report.values())
    index = f'''<!doctype html><html lang="zh-Hans"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多 harness RL 终极指南 — 简体中文版</title><link rel="stylesheet" href="assets/style.css"><link rel="stylesheet" href="assets/reading.css"></head><body>
<section class="hero"><div class="eyebrow">简体中文全译本 · 交互阅读版</div><h1>多 harness RL 终极指南<br>The ultimate guide to multi-harness RL</h1><p class="sub">同一个模型，在不同智能体运行框架中表现不同。现在，你可以直接在 Claude Code、Codex、OpenCode 等真实 harness 内，用强化学习训练开放模型，让它在多种框架中同时改善任务表现与工具效率。</p><p class="sub" style="font-size:.82rem">作者：{AUTHORS}<br>Hugging Face · Liquid AI · 原文发布于 2026 年 10 月 1 日</p><div class="meta"><div><b>6</b>正文章节</div><div><b>{embeds}</b>交互图表实例</div><div><b>8</b>原文代码块</div><div><b>CC BY 4.0</b>内容许可</div></div><nav class="reading-actions"><a href="chapters/introduction.html">开始阅读 →</a><a href="full.html">连续阅读全文</a><a href="chapters/references.html">参考文献</a><a href="{SOURCE_URL}">英文原文 ↗</a></nav></section><div class="grid-wrap"><h2>阅读章节</h2><div class="chapter-grid">{cards}</div></div><footer class="foot">{FOOT}<br>阅读布局参考 <a href="https://github.com/ai-twinkle/ultrascale-playbook-zh-tw">Ultra-Scale Playbook 中文版</a>；译文源稿与原文快照随目录保存。</footer></body></html>'''
    (WEB / 'index.html').write_text(index)
    exported_md = '\n\n'.join(expanded).replace('../assets/', 'webapp/assets/').replace('../embeds/', 'webapp/embeds/').replace('](/data/', '](webapp/assets/data/')
    exported_md = re.sub(r'\[@([\w-]+)\]', lambda m: f'[{cited.index(m[1])+1}](#ref-{m[1]})', exported_md)
    exported_md = re.sub(r'(?<![\w/])@([a-zA-Z][\w-]+)', lambda m: f'[{cited.index(m[1])+1}](#ref-{m[1]})', exported_md)
    (ROOT / 'multi_harness_rl_zh.md').write_text('# 多 harness RL 终极指南\n\n' + AUTHORS + '\n\n原文：' + SOURCE_URL + '\n\n非官方简体中文译本，正文采用 CC BY 4.0；语言和布局已改编。\n\n' + exported_md + '\n\n' + references + '\n')
    (ROOT / 'qa').mkdir(exist_ok=True)
    report['totals'] = dict(headings=sum(x['headings'] for x in report.values()), code_blocks=sum(x['code_blocks'] for x in report.values()), embeds=embeds, citations=len(cited))
    (ROOT / 'qa/coverage.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Built 6 chapters + references + index + full page; {report["totals"]}')


if __name__ == '__main__':
    main()
