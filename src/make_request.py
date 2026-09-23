import argparse
import base64
import io
import json
from PIL import Image

parser = argparse.ArgumentParser(); parser.add_argument("--image", required=True); parser.add_argument("--out", default="sample-request.json")
args = parser.parse_args()
image = Image.open(args.image).convert("RGB")
image.thumbnail((512, 512))
buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=85)
Path = __import__("pathlib").Path
Path(args.out).write_text(json.dumps({"image": base64.b64encode(buffer.getvalue()).decode("ascii")}), encoding="utf-8")
print(f"Wrote {args.out} ({len(buffer.getvalue())} bytes)")