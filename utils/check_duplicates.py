# utils/check_duplicates.py
import os

def get_all_filenames(folder):
    names = set()
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                names.add(f)
    return names

train_files = get_all_filenames("data/train_new")
valid_files = get_all_filenames("data/valid_new")
test_files  = get_all_filenames("data/test_new")

train_valid_overlap = train_files & valid_files
train_test_overlap  = train_files & test_files
valid_test_overlap  = valid_files & test_files

print(f"Train-Valid overlap: {len(train_valid_overlap)}")
print(f"Train-Test overlap: {len(train_test_overlap)}")
print(f"Valid-Test overlap: {len(valid_test_overlap)}")

if not any([train_valid_overlap, train_test_overlap, valid_test_overlap]):
    print("✅ No duplicate filenames found across splits!")
else:
    print("⚠️ Some duplicates detected — check dataset integrity.")
