#!/usr/bin/env python3
import os
import sys
import json
import pickle
import argparse
from collections import defaultdict
from PIL import Image
import copy
import math

def round_to_base(x, base):
    return max(base, int(round(x / base)) * base)

def generate_candidates(base, max_side=2048):
    base_widths = [3840, 2560, 1920, 1600, 1360, 1280, 1024, 896, 768,720, 640, 512, 384]
    ARs = [16/9, 4/3, 1/1, 3/4, 9/16,2/3,3/2] 
    cands = set()
    for w0 in base_widths:
        if w0 > max_side:
            continue
        for ar in ARs:
            w = round_to_base(w0, base)
            h = round_to_base(w0 / ar, base)
            if w <= max_side and h <= max_side:
                cands.add((w, h))

            h_alt = round_to_base(w0, base)
            w_alt = round_to_base(w0 * ar, base)
            if w_alt <= max_side and h_alt <= max_side:
                cands.add((w_alt, h_alt))

    smalls = [(base, base), (base*2, base), (base*2, base*2), (1280,720), (1024,768), (720,1280)]
    for s in smalls:
        if s[0] <= max_side and s[1] <= max_side:
            cands.add(s)
    return sorted(list(cands), key=lambda x: x[0]*x[1])

def choose_best_resolution(w, h, SPATIAL_DOWNSAMPLE=8, divisible_by=8, max_side=2048, allow_upscale=False):
    base = SPATIAL_DOWNSAMPLE * divisible_by
    ar_in = w / h
    area_in = w * h

    candidates = generate_candidates(base, max_side=max_side)
    if not candidates:

        tw = min(round_to_base(w, base), max_side)
        th = min(round_to_base(h, base), max_side)
        return tw, th

    best = None
    best_score = float('inf')
    for cw, ch in candidates:
        ar_c = cw / ch
        area_c = cw * ch
        aspect_diff = abs(ar_in - ar_c) / max(ar_in, ar_c)
        area_ratio = area_c / area_in

        if not allow_upscale and area_ratio > 1.0:

            area_penalty = abs(math.log(area_ratio)) * 1.0 + 1.0
        else:
            area_penalty = abs(math.log(max(area_ratio, 1e-8))) * 0.5

        score = aspect_diff * 1.0 + area_penalty * 0.6


        if area_ratio > 4.0 or area_ratio < 0.1:
            continue

        if score < best_score:
            best_score = score
            best = (cw, ch)

    if best is None:

        tw = min(round_to_base(w, base), max_side)
        th = min(round_to_base(h, base), max_side)
        return tw, th

    return best
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_image_size(path):
    try:
        with Image.open(path) as im:
            w, h = im.size
            w, h = choose_best_resolution(w, h, SPATIAL_DOWNSAMPLE=8, divisible_by=8, max_side=1024, allow_upscale=False)
            return (w,h)
    except Exception:
        return None
def find_bucket_for_size(buckets, size, tol):
    """
    buckets: list of dicts with keys: 'rep'=(w,h), 'items'=[...]
    size: (w,h)
    tol: int tolerance (absolute)
    Return index of bucket that matches within tol, or None if none.
    Choose bucket with minimal max(abs(dw), abs(dh)) among those within tol.
    """
    best_idx = None
    best_score = None
    w, h = size
    for i, b in enumerate(buckets):
        rep = b["rep"]
        if rep is None:
            continue
        dw = abs(rep[0] - w)
        dh = abs(rep[1] - h)
        if dw <= tol and dh <= tol:
            score = max(dw, dh)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = i
    return best_idx

def main():
    parser = argparse.ArgumentParser(description="Reorder JSON by (similar) image-size batches with tolerance; full batches first")
    parser.add_argument("--input_json", required=True, help="Input JSON (list of items, each has 'caption')")
    parser.add_argument("--pkl", required=True, help="PKL file mapping caption -> list of ABSOLUTE image paths")
    parser.add_argument("--output_json", required=True, help="Output reordered JSON path")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size (default 48)")
    parser.add_argument("--tol", type=int, default=0, help="Tolerance in pixels for w and h to group into same bucket (default 0)")
    parser.add_argument("--max_warn", type=int, default=20, help="Max number of warnings to print")
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        print(f"Input JSON not found: {args.input_json}", file=sys.stderr); sys.exit(1)
    if not os.path.exists(args.pkl):
        print(f"PKL not found: {args.pkl}", file=sys.stderr); sys.exit(1)

    json_list = load_json(args.input_json)
    cap2paths = load_pkl(args.pkl)

    # normalize mapping to lists
    for k, v in list(cap2paths.items()):
        if v is None:
            cap2paths[k] = []
        elif not isinstance(v, (list, tuple)):
            cap2paths[k] = [v]
        else:
            cap2paths[k] = list(v)

    # buckets stored as list to preserve order of first-seen representative sizes
    # each bucket: {"rep": (w,h) or None, "items": [item, ...]}
    buckets = []
    # We will keep one special None-size bucket if needed (rep=None)
    warn = 0

    for idx, item in enumerate(json_list):
        if "caption" not in item:
            if warn < args.max_warn:
                print(f"Warning: item {idx} missing 'caption'; placed into None-size bucket", file=sys.stderr)
            warn += 1
            none_idx = next((i for i,b in enumerate(buckets) if b["rep"] is None), None)
            if none_idx is None:
                buckets.append({"rep": None, "items": [item]})
            else:
                buckets[none_idx]["items"].append(item)
            continue

        cap = item["caption"]
        paths = cap2paths.get(cap) or []
        size = None
        if paths:
            # pick the first readable image path
            for p in paths:
                s = get_image_size(p)
                if s is not None:
                    size = s
                    break
            if size is None:
                if warn < args.max_warn:
                    print(f"Warning: no readable images for caption '{cap}' (item {idx}); placed into None-size bucket", file=sys.stderr)
                warn += 1
                none_idx = next((i for i,b in enumerate(buckets) if b["rep"] is None), None)
                if none_idx is None:
                    buckets.append({"rep": None, "items": [item]})
                else:
                    buckets[none_idx]["items"].append(item)
            else:
                match_idx = find_bucket_for_size(buckets, size, args.tol)
                if match_idx is not None:
                    buckets[match_idx]["items"].append(item)
                else:
                    buckets.append({"rep": (size[0], size[1]), "items": [item]})
        else:
        
            if warn < args.max_warn:
                print(f"Warning: no image paths for caption '{cap}' (item {idx}); placed into None-size bucket", file=sys.stderr)
            warn += 1
            none_idx = next((i for i,b in enumerate(buckets) if b["rep"] is None), None)

    if warn > args.max_warn:
        print(f"(Total warnings: {warn}; only first {args.max_warn} shown)", file=sys.stderr)

    # Collect full batches across buckets first, keep leftovers to be padded later.
    full_batches = []    # list of lists (each list length == batch_size)
    leftover_chunks = []  # list of lists (each list length in (0, batch_size))

    for b in buckets:
        items = b["items"]
        if not items:
            continue
        L = len(items)
        n_full = L // args.batch_size
        # emit all full batches in order
        for i in range(n_full):
            start = i * args.batch_size
            batch = items[start:start + args.batch_size]
            # ensure we don't accidentally keep references when padding later; but full batches are original items
            full_batches.append(batch)
        # leftover
        rem = L % args.batch_size
        if rem > 0:
            leftover = items[n_full * args.batch_size : n_full * args.batch_size + rem]
            leftover_chunks.append(leftover)

    # Now build final reordered list: all full batches first (preserving inter-batch order),
    # then padded batches created from leftover_chunks (preserve leftover chunk order).
    reordered = []
    for batch in full_batches:
        reordered.extend(batch)

    for chunk in leftover_chunks:
        # pad each chunk by repeating its first item until batch_size
        if len(chunk) == 0:
            continue
        padded = list(chunk)
        first = chunk[0]
        while len(padded) < args.batch_size:
            padded.append(copy.deepcopy(first))
        reordered.extend(padded)

    # Save reordered JSON
    out_dir = os.path.dirname(args.output_json)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(reordered, f, ensure_ascii=False, indent=4)

    print(f"Saved reordered json to {args.output_json}. Original items: {len(json_list)}, output items: {len(reordered)}")

if __name__ == "__main__":
    main()
