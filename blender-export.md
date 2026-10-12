# Blender Grease Pencil export

`blender.py` (repo root) converts the Twemoji SVGs in `data/svg/` into Blender **Grease Pencil** objects and works out which shapes are the **eyes, mouth, brows, …** so they can be animated later.

It is independent of the Three.js pipeline: it reads the SVGs directly, so the dev server is **not** needed.

## Requirements

- Python 3.8+ (standard library only, no `pip install`)
- **Blender 4.3 or newer** (needs the new Grease Pencil data model). On Ubuntu:
  ```bash
  sudo snap install blender --classic
  ```
  Ubuntu's apt `blender` package is usually older than 4.3 and will not work.

`blender.py` looks for Blender via `--blender PATH`, then `$BLENDER`, then `PATH`, then `/snap/bin`, `/opt/blender*`, `~/blender*`.

## Usage

```bash
# one or a few emoji (shortname, unicode code, or a mix)
python3 blender.py --emojis joy,1f600,heart_eyes --output ./output/blender/

# everything (~5 min; one Blender process handles the whole batch)
python3 blender.py --emojis all --output ./output/blender/

# just work out the face map: no Blender needed
python3 blender.py --emojis all --facemap-only --debug-svg
```

| Flag | Default | |
|------|---------|--|
| `--emojis` | required | shortnames, unicode codes, a mix, or `all` |
| `--output` | `./output/blender` | output directory (already git-ignored) |
| `--naming` | `unicode` | `unicode` (`1f602`) or `shortname` (`joy`) file names |
| `--size` | `2.0` | emoji size in Blender units |
| `--curve-segments` | `8` | line segments per Bezier curve, same meaning as the TS CLI |
| `--blender` | auto | path to the Blender executable |
| `--facemap-only` | off | write the face-map JSON only, skip Blender |
| `--debug-svg` | off | write `<name>.facemap.svg`, the shapes outlined and labelled |
| `--verbose` | off | show all Blender output |

## Output

For each emoji:

| File | |
|------|--|
| `<name>.blend` | Grease Pencil object, one layer per face part |
| `<name>.facemap.json` | which shapes were identified as what |
| `<name>.facemap.svg` | (`--debug-svg`) a picture of that mapping, open it in a browser |

The face map is also stored inside the `.blend` as the custom property `face_map` (JSON string) on both the object and its Grease Pencil data, plus `emoji_code` on the object.

### Layers

Each face part is its own **layer**, and the layer's origin (pivot) is placed at the centre of its shapes. Scaling an eye layer therefore scales around the eye, and rotating a brow layer pivots around the brow.

| Layer | Meaning |
|-------|---------|
| `face` | the head shape |
| `eye_left`, `eye_right` | eyes |
| `pupil_*`, `highlight_*` | shapes inside an eye (darker → pupil, lighter → highlight) |
| `brow_left`, `brow_right` | eyebrows |
| `mouth`, `teeth`, `tongue`, `mouth_detail` | mouth and what is inside it |
| `tear_*`, `blush_*` | tears and cheek blush |
| `other_01`, `other_02`, … | everything else, split into runs so the original SVG paint order is kept (e.g. the hand in 🤔 stays on top of the mouth) |

> **`left`/`right` are screen-side**: as seen by the viewer, `left` has the smaller X. They are *not* the character's own left/right.

Layer order is the SVG paint order, so the original picture is reproduced exactly.

### `facemap.json`

```jsonc
{
  "code": "1f600",
  "is_face": true,       // a head plus eyes and/or a mouth were found (deliberately permissive)
  "complete": true,      // both eyes AND a mouth were found: use this when you need to be sure
  "groups": { "eyes": ["eye_left", "eye_right"], "mouth": ["mouth", "teeth"] },
  "warnings": [],
  "parts": {
    "eye_left": {
      "layer": "eye_left",
      "strokes": 1,
      "center": [-0.333, 0.25],            // Blender X, Z (also the layer's origin)
      "bbox": [-0.472, 0.056, -0.194, 0.444],
      "svg_bbox": [9.5, 10.0, 14.5, 17.0], // original SVG coordinates
      "colors": ["#664500"],
      "rule": "eye-pair",                  // which heuristic matched it
      "confidence": 1.0                    // 1.0 strong pair … 0.4 single-eye guess
    }
  }
}
```

## Animating the parts

Layers carry their own transform, so basic facial animation is just keyframing layers. A blink, from a Blender Python console or script:

```python
import bpy, json

obj = bpy.data.objects["1f602"]
gp = obj.data
face_map = json.loads(obj["face_map"])

for part in face_map["groups"]["eyes"]:
    name = face_map["parts"][part]["layer"]
    for frame, squash in ((1, 1.0), (6, 0.1), (11, 1.0)):
        gp.layers[name].scale = (1.0, 1.0, squash)   # Z is vertical in front view
        gp.keyframe_insert(data_path=f'layers["{name}"].scale', frame=frame)
```

(Verified on Blender 5.2: the eye squashes to 10 % at frame 6 and returns by frame 11.)

## How the face detection works

Twemoji SVGs have no semantic names, and often draw both eyes, both brows *and* the mouth in a single `<path>`. So `blender.py` splits every path into its sub-paths and classifies each polygon geometrically:

1. **Head**: a round shape covering most of the canvas (this is what rules out flags and button icons).
2. **Eyes**: the best mirror-symmetric pair in the upper half. When two symmetric pairs exist (eyes *and* brows), the lower one wins.
3. **Mouth**: the biggest non-skin, non-blue shape in the lower centre. Shapes inside it become `teeth` / `tongue` / `mouth_detail` by colour.
4. **Pupils/highlights** inside eyes, **brows** (thin, dark, above an eye), **tears** (blue) and **blush** (soft pink circles).

Measured over all 4,009 bundled emoji against Unicode's own face categories (face-*, cat/monkey faces): **96 % of real faces are detected, 90 % with both eyes and a mouth.** Expect some misses (😎 sunglasses hide the eyes, 🙃 is upside-down, 🤯 👾 🤖 are unusual shapes) and some extra detections: people and animal faces are detected too (usually correct), and a few objects with symmetric blobs (🎳, 🌮) are labelled as faces. Check `complete`, `confidence` and `--debug-svg` if it matters.

## Known limitations

- **Blender version**: developed and tested against **Blender 5.2.2**. The code also targets 4.3–4.5 (it falls back to the older material flags and `grease_pencils_v3`), but that path is **untested**. Older than 4.3 is refused with a clear message.
- **Fills only.** Twemoji is flat colour, so there are no strokes, gradients, clip paths or masks (14 SVGs use a clip path, which is ignored with a warning in `facemap.json`).
- **Holes** (rings, the "O" in a button) cannot be cut out of a single Grease Pencil fill, so the hole is painted as an extra stroke in the colour of whatever is behind it. It looks right, but it is not transparent.
- **Face detection is heuristic**, see above.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The parsing and face-detection tests need nothing else. The Blender-side tests run only if the `bpy` module is installed (`pip install bpy` in a venv; the real `blender` executable is not needed) and are skipped otherwise.
