import json
import re
from collections import Counter
from typing import List, Dict, Any


def is_printable(s: str) -> bool:
    """Check if all characters in the string are printable (including whitespace)."""
    return all((c.isprintable() or c in '\t\n\r') for c in s)


def find_non_printable(s: str) -> List[str]:
    """Return a list of unique non-printable characters in the string."""
    return sorted(set(c for c in s if not (c.isprintable() or c in '\t\n\r')))


def scan_jsonl_file(filepath: str, max_records: int = 0) -> Dict[str, Any]:
    """
    Scan a JSONL file for encoding issues, non-printable characters, and malformed fields.
    Returns a summary and examples.
    """
    issues = []
    field_stats = Counter()
    non_printable_examples = []
    malformed_lines = []
    total_records = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if max_records and i > max_records:
                break
            line = line.rstrip('\n')
            try:
                obj = json.loads(line)
            except Exception as e:
                malformed_lines.append({'line_number': i, 'error': str(e), 'line': line[:200]})
                continue
            total_records += 1
            for field, value in obj.items():
                field_stats[field] += 1
                if isinstance(value, str):
                    non_print = find_non_printable(value)
                    if non_print:
                        non_printable_examples.append({
                            'line_number': i,
                            'field': field,
                            'non_printable': non_print,
                            'snippet': value[:100]
                        })
    summary = {
        'total_records': total_records,
        'malformed_lines': malformed_lines,
        'non_printable_examples': non_printable_examples[:10],
        'field_stats': dict(field_stats),
    }
    return summary


def main():
    filepath = "data/processed/doaj_vaccines_structured.jsonl"
    result = scan_jsonl_file(filepath)
    with open("data/processed/doaj_vaccines_structured_scan_report.json", "w", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print("Scan complete. See data/processed/doaj_vaccines_structured_scan_report.json for results.")


if __name__ == "__main__":
    main()
