import json
from pathlib import Path

input_path = Path('data/raw/naturalnews_science_links_tags.jsonl')
output_path = Path('data/raw/naturalnews_science_unique_tags.jsonl')

tags_set = set()

with input_path.open('r', encoding='utf-8') as infile:
    for line in infile:
        try:
            record = json.loads(line)
            tags = record.get('tags', [])
            for tag in tags:
                tags_set.add(tag)
        except Exception:
            continue

with output_path.open('w', encoding='utf-8') as outfile:
    for tag in sorted(tags_set):
        json.dump({'tag': tag}, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f'Extracted {len(tags_set)} unique tags to {output_path}')
