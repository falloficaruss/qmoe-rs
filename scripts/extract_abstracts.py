import re


def extract_field(body, field):
    match = re.search(rf'{field}\s*=\s*\{{', body)
    if not match:
        return None
    start = match.end()
    depth = 1
    i = start
    while i < len(body) and depth > 0:
        if body[i] == '{':
            depth += 1
        elif body[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return body[start:i-1]
    return None


with open('Connected_Papers.bib') as f:
    content = f.read()

entries = re.findall(r'@article\{([^,]+),\s*(.*?)\n\}', content, re.DOTALL)

with open('abstracts.md', 'w') as out:
    out.write('# Extracted Abstracts\n\n')
    for key, body in entries:
        title = extract_field(body, 'title')
        abstract = extract_field(body, 'abstract')
        if not title or not abstract:
            continue
        abstract = abstract.strip()
        if abstract in ('null', '', ',', '{,}'):
            continue
        out.write(f'## {title}\n\n{abstract}\n\n')

count = 0
with open('abstracts.md') as f:
    for line in f:
        if line.startswith('## '):
            count += 1
print(f'Done. Extracted {count} abstracts.')
