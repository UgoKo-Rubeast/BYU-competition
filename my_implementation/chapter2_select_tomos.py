import random


def run_chapter2_select_tomos(input_root, num_tomos, seed):
    all_tomos = sorted([
        p for p in input_root.iterdir()
        if p.is_dir() and p.name.startswith("tomo_")
    ])

    if len(all_tomos) < num_tomos:
        raise ValueError(f"tomo フォルダ数が不足しています: found={len(all_tomos)}, required={num_tomos}")

    rng = random.Random(seed)
    selected_tomos = rng.sample(all_tomos, k=num_tomos)

    assert len(selected_tomos) == num_tomos, "selected_tomos length mismatch"
    assert len(set(p.name for p in selected_tomos)) == num_tomos, "selected_tomos contains duplicates"

    print("[INFO] selected_tomos:")
    for p in selected_tomos:
        print(" -", p.name)

    return all_tomos, selected_tomos
