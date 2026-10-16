import os
from PIL import Image
from collections import Counter, defaultdict

def main():
    print("==================================================")
    print("       ExDark Dataset & Split Verification        ")
    print("==================================================")

    # 1. Inspect imageclasslist.txt
    print("\n[1] Inspecting Groundtruth/imageclasslist.txt...")
    split_counts = Counter()
    class_split_counts = defaultdict(Counter)
    light_counts = Counter()
    in_out_counts = Counter()

    split_file = os.path.join("Groundtruth", "imageclasslist.txt")
    with open(split_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0]
    print(f"Header: {header}")
    data_lines = lines[1:]
    print(f"Total entries: {len(data_lines)}")

    class_id_to_name = {
        "1": "Bicycle", "2": "Boat", "3": "Bottle", "4": "Bus",
        "5": "Car", "6": "Cat", "7": "Chair", "8": "Cup",
        "9": "Dog", "10": "Motorbike", "11": "People", "12": "Table"
    }

    for line in data_lines:
        parts = [p.strip() for p in line.split()]
        name, cls, light, in_out, split = parts[0], parts[1], parts[2], parts[3], parts[4]
        split_name = {"1": "Train", "2": "Val", "3": "Test"}.get(split, split)
        split_counts[split_name] += 1
        class_split_counts[cls][split_name] += 1
        light_counts[light] += 1
        in_out_counts[in_out] += 1

    print(f"Overall Split Distribution: {dict(split_counts)}")
    print("\nPer-class splits (Class ID / Name: Train / Val / Test / Total):")
    for cls, counts in sorted(class_split_counts.items(), key=lambda x: int(x[0])):
        cname = class_id_to_name.get(cls, f"Class_{cls}")
        print(f"  {cls:>2} ({cname:<9}): Train={counts['Train']:>4}, Val={counts['Val']:>4}, Test={counts['Test']:>4}, Total={sum(counts.values()):>4}")

    # 2. Inspect Annotations & Images
    print("\n[2] Inspecting Image Files & Annotation Geometry...")
    class_names = sorted(os.listdir("ExDark"))
    total_boxes = 0
    boxes_per_image = Counter()
    clipped_boxes = 0
    img_modes = Counter()
    img_exts = Counter()
    widths, heights = [], []

    for cls in class_names:
        img_dir = os.path.join("ExDark", cls)
        anno_dir = os.path.join("ExDark_Annno", cls)
        if not os.path.isdir(img_dir):
            continue

        for fname in os.listdir(img_dir):
            ext = os.path.splitext(fname)[1].lower()
            img_exts[ext] += 1

            img_path = os.path.join(img_dir, fname)
            with Image.open(img_path) as im:
                img_modes[im.mode] += 1
                w, h = im.size
                widths.append(w)
                heights.append(h)

            anno_file = fname + ".txt"
            anno_path = os.path.join(anno_dir, anno_file)

            img_boxes = 0
            if os.path.exists(anno_path):
                with open(anno_path, "r") as af:
                    alines = af.readlines()
                for aline in alines:
                    aline = aline.strip()
                    if not aline or aline.startswith("%"):
                        continue
                    tokens = aline.split()
                    if len(tokens) >= 5:
                        try:
                            l, t, bw, bh = float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4])
                            if bw <= 0 or bh <= 0 or l < 0 or t < 0 or (l + bw) > w or (t + bh) > h:
                                clipped_boxes += 1
                            img_boxes += 1
                        except ValueError:
                            pass
            boxes_per_image[img_boxes] += 1
            total_boxes += img_boxes

    print(f"\nTotal Bounding Boxes across all classes: {total_boxes}")
    print(f"Boxes per image distribution: {dict(sorted(boxes_per_image.items()))}")
    print(f"Boxes requiring coordinate boundary clipping [0, W/H]: {clipped_boxes}")
    print(f"Image Color Modes in Dataset: {dict(img_modes)}")
    print(f"Image File Extensions: {dict(img_exts)}")
    print(f"Image Dimensions: Min=({min(widths)}x{min(heights)}), Max=({max(widths)}x{max(heights)}), Mean=({sum(widths)//len(widths)}x{sum(heights)//len(heights)})")
    print("==================================================")
    print("            M1 Inspection Finished!               ")
    print("==================================================")

if __name__ == "__main__":
    main()
