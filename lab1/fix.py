import json, os

JSON_DIR = r"C:\Users\nguye\Documents\3RD YEAR\TT_nhung\lab1\segmentation_dataset\images"

for f in os.listdir(JSON_DIR):
    if not f.endswith(".json"): continue
    
    with open(os.path.join(JSON_DIR, f)) as fp:
        data = json.load(fp)
    
    print(f"=== {f} ===")
    print(f"imageWidth:  {data.get('imageWidth')}")
    print(f"imageHeight: {data.get('imageHeight')}")
    
    for shape in data.get("shapes", []):
        pts = shape.get("points", [])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        print(f"label: {shape['label']}")
        print(f"x range: {min(xs):.1f} → {max(xs):.1f}")
        print(f"y range: {min(ys):.1f} → {max(ys):.1f}")
    break
