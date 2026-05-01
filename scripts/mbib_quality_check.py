from datasets import load_dataset
import statistics

ds = load_dataset("mediabiasgroup/mbib-base")
fake = ds["fake_news"]
print(fake.column_names)

# sprawdź rozkład długości
lengths = [len(r["text"]) for r in fake]
print(f"median: {statistics.median(lengths)}, mean: {statistics.mean(lengths):.0f}")
print(f"pod 500 znaków: {sum(1 for l in lengths if l < 500)}/{len(lengths)}")
