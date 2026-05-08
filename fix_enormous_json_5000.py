import json
from collections import Counter
from tqdm.auto import tqdm

GIANT    = "/home/temari/god please no diploma/restore_punct/data/rare_punct_fiction.json"
FILTERED = "/home/temari/god please no diploma/restore_punct/data/rare_punct_5000.json"

TARGET_IDS = frozenset({6, 12, 13, 16, 17, 19, 20, 21, 22, 24, 25})

MAX_EXAMPLES_PER_CLASS = 455
MAX_TOTAL              = 5_000

class_counts = Counter()

def want_example(ner_tags):
    hits = TARGET_IDS.intersection(ner_tags)
    if not hits:
        return False
    return any(class_counts[t] < MAX_EXAMPLES_PER_CLASS for t in hits)

kept = 0
first = True
with open(GIANT, "r", encoding="utf-8") as fin, \
     open(FILTERED, "w", encoding="utf-8") as fout:
    fout.write("[\n")
    for line in tqdm(fin, desc="Filtering 151 GB (5k subset)"):
        line = line.strip()
        if line in ("[", "]") or not line:
            continue
        if line.endswith(","):
            line = line[:-1]
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        tags = item.get("ner_tags", [])
        if not want_example(tags):
            continue
        for t in TARGET_IDS.intersection(tags):
            class_counts[t] += 1
        if not first:
            fout.write(",\n")
        json.dump(item, fout, ensure_ascii=False)
        first = False
        kept += 1
        if kept >= MAX_TOTAL:
            break
        if all(class_counts[t] >= MAX_EXAMPLES_PER_CLASS for t in TARGET_IDS):
            print("All classes saturated")
            break
    fout.write("\n]\n")

print(f"\nKept {kept} examples")
for t in sorted(TARGET_IDS):
    print(f"  class {t:2d}: {class_counts[t]:>6d} examples")
