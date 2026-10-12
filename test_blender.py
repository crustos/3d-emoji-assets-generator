"""Tests for blender.py.

Run from the repo root:

    python3 -m unittest discover -s tests -v

The parsing and face-detection tests need only the standard library. The
Blender-side tests run only if the `bpy` module is importable (for example
`pip install bpy` in a venv); otherwise they are skipped. The real `blender`
executable is not needed.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import blender as B  # noqa: E402

HAS_BPY = importlib.util.find_spec("bpy") is not None


def model_for(code, size=2.0, segments=8):
    model, _ = B.build_model(code, code, ROOT / "data" / "svg" / f"{code}.svg", size, segments)
    return model


def write_svg(tmp: Path, body: str) -> Path:
    p = tmp / "t.svg"
    p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36">{body}</svg>')
    return p


class PathFlattening(unittest.TestCase):
    def test_relative_square(self):
        (sp,) = B.flatten_path("M0 0h10v10H0z")
        self.assertEqual(sp[:3], [(0, 0), (10, 0), (10, 10)])
        self.assertAlmostEqual(abs(B.signed_area(B._clean_polygon(sp))), 100)

    def test_implicit_lineto_after_moveto(self):
        (sp,) = B.flatten_path("M0 0 10 0 10 10")
        self.assertEqual(sp, [(0, 0), (10, 0), (10, 10)])

    def test_arc_with_packed_flags(self):
        # "a5 5 0 1010 0": the two flags are written without separators.
        (sp,) = B.flatten_path("M0 0a5 5 0 1010 0", 8)
        self.assertEqual(sp[-1], (10, 0))
        for x, y in sp:
            self.assertAlmostEqual(((x - 5) ** 2 + y ** 2) ** 0.5, 5, places=6)

    def test_smooth_curve_continues(self):
        (sp,) = B.flatten_path("M0 0C0 10 10 10 10 0S20 -10 20 0", 8)
        self.assertEqual(len(sp), 1 + 8 + 8)
        self.assertEqual(sp[-1], (20, 0))

    def test_every_bundled_svg_parses(self):
        for svg in sorted((ROOT / "data" / "svg").glob("*.svg"))[::25]:
            B.parse_svg(svg)  # must not raise or hang


class Holes(unittest.TestCase):
    RING_ON_BLUE = ('<path fill="#3b88c3" d="M0 0h36v36H0z"/>'
                    '<path fill="#fff" fill-rule="evenodd" d="M4 4h28v28H4z M12 12h12v12H12z"/>')

    def parse(self, body):
        with tempfile.TemporaryDirectory() as t:
            return B.parse_svg(write_svg(Path(t), body))

    def test_ring_hole_takes_underlying_colour(self):
        parsed = self.parse(self.RING_ON_BLUE)
        holes = [u for u in parsed.units if u.hole]
        self.assertEqual(len(holes), 1)
        self.assertEqual(holes[0].color, "#3b88c3")
        self.assertIsNotNone(holes[0].parent)
        # hole is painted after its parent, so it shows through
        self.assertGreater(holes[0].order, holes[0].parent.order)

    def test_ring_does_not_hang_or_flag_the_outer_as_hole(self):
        # Regression: the outer polygon's interior point lies inside the hole,
        # which used to make the two sub-paths "contain" each other forever.
        parsed = self.parse('<path fill="#000" fill-rule="evenodd" d="M4 4h28v28H4z M12 12h12v12H12z"/>')
        self.assertEqual(sum(u.hole for u in parsed.units), 0)  # nothing underneath: hole dropped
        self.assertEqual(len([u for u in parsed.units if not u.hole]), 1)

    def test_same_direction_nesting_is_not_a_hole_under_nonzero(self):
        parsed = self.parse('<path fill="#000" d="M4 4h28v28H4z M12 12h12v12H12z"/>')
        self.assertEqual(sum(u.hole for u in parsed.units), 0)
        self.assertEqual(len(parsed.units), 2)


class FaceDetection(unittest.TestCase):
    def parts(self, code):
        return model_for(code)["facemap"]["parts"]

    def test_laughing_face(self):
        p = self.parts("1f602")
        for name in ("face", "eye_left", "eye_right", "brow_left", "brow_right", "mouth", "teeth",
                     "tear_left", "tear_right"):
            self.assertIn(name, p)
        # screen-side naming: left has the smaller x
        self.assertLess(p["eye_left"]["center"][0], p["eye_right"]["center"][0])

    def test_eyes_are_below_brows(self):
        # Regression: the brow pair is also symmetric and used to win (😭, 🐶).
        for code in ("1f602", "1f62d"):
            p = self.parts(code)
            self.assertLess(p["eye_left"]["center"][1], p["brow_left"]["center"][1], code)  # Z is up

    def test_crying_face_mouth_is_not_the_tear_puddle(self):
        p = self.parts("1f62d")
        self.assertIn("tongue", p)
        width = p["mouth"]["bbox"][2] - p["mouth"]["bbox"][0]
        self.assertLess(width, 0.6)  # the puddle spans the whole 2.0-wide face
        self.assertNotIn("#5dadec", p["mouth"]["colors"])

    def test_dog_eyes_are_the_dark_ovals(self):
        p = self.parts("1f436")
        self.assertEqual(p["eye_left"]["colors"], ["#292f33"])

    def test_pupils_and_blush(self):
        p = self.parts("1f633")
        for name in ("pupil_left", "pupil_right", "blush_left", "blush_right", "brow_left"):
            self.assertIn(name, p)

    def test_hands_and_hearts_are_not_blush(self):
        for code in ("1f631", "1f970"):
            self.assertFalse([k for k in self.parts(code) if k.startswith("blush")], code)

    def test_heart_eyes(self):
        p = self.parts("1f60d")
        self.assertEqual(p["eye_left"]["colors"], ["#dd2e44"])

    def test_non_faces(self):
        for code in ("1f680", "1f1e6-1f1eb", "1f1f4"):  # rocket, Afghan flag, letter button
            fm = model_for(code)["facemap"]
            self.assertFalse(fm["is_face"], code)
            self.assertEqual(fm["parts"], {})

    def test_complete_flag(self):
        self.assertTrue(model_for("1f602")["facemap"]["complete"])
        self.assertFalse(model_for("1f680")["facemap"]["complete"])


class Layers(unittest.TestCase):
    def test_paint_order_survives_layer_split(self):
        # 🤔: the hand is drawn after (on top of) the mouth.
        m = model_for("1f914")
        names = [l["name"] for l in m["layers"]]
        self.assertLess(names.index("mouth"), names.index("other_02"))
        self.assertEqual(m["facemap"]["warnings"], [])

    def test_layer_origin_is_part_centre(self):
        m = model_for("1f602")
        for layer in m["layers"]:
            xs = [p[0] for s in layer["strokes"] for p in s["points"]]
            zs = [p[2] for s in layer["strokes"] for p in s["points"]]
            self.assertAlmostEqual((min(xs) + max(xs)) / 2, 0, places=6)
            self.assertAlmostEqual((min(zs) + max(zs)) / 2, 0, places=6)

    def test_size_is_respected(self):
        m = model_for("1f600", size=4.0)
        face = m["layers"][0]
        xs = [p[0] for p in face["strokes"][0]["points"]]
        self.assertAlmostEqual(max(xs) - min(xs), 4.0, places=2)


class Cli(unittest.TestCase):
    def test_resolve_codes(self):
        short, _ = B.load_emoji_tables()
        self.assertEqual(B.resolve_codes("joy", short), ["1f602"])
        self.assertEqual(B.resolve_codes(":joy:,1F602,1f600", short), ["1f602", "1f600"])  # deduped
        with self.assertRaises(SystemExit):
            B.resolve_codes("definitely_not_an_emoji", short)

    def test_all_resolves_to_every_svg(self):
        short, _ = B.load_emoji_tables()
        self.assertEqual(len(B.resolve_codes("all", short)), len(list((ROOT / "data" / "svg").glob("*.svg"))))

    def test_facemap_only_needs_no_blender(self):
        with tempfile.TemporaryDirectory() as t:
            rc = B.host_main(["--emojis", "1f602", "--output", t, "--facemap-only", "--debug-svg"])
            self.assertEqual(rc, 0)
            fm = json.loads((Path(t) / "1f602.facemap.json").read_text())
            self.assertIn("eye_left", fm["parts"])
            self.assertTrue((Path(t) / "1f602.facemap.svg").exists())


@unittest.skipUnless(HAS_BPY, "bpy module not installed")
class BlenderSide(unittest.TestCase):
    def build(self, code):
        import bpy
        out = tempfile.mkdtemp() + f"/{code}.blend"
        B._build_emoji_in_blender(bpy, model_for(code), out)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.open_mainfile(filepath=out)
        return bpy.data.objects[code]

    def test_layers_and_pivots(self):
        obj = self.build("1f602")
        layers = obj.data.layers
        self.assertEqual([l.name for l in layers][0], "face")
        self.assertLess(layers["eye_left"].translation[0], 0)
        self.assertGreater(layers["eye_right"].translation[0], 0)
        self.assertAlmostEqual(layers["eye_left"].translation[0], -layers["eye_right"].translation[0], places=4)

    def test_every_stroke_is_filled_and_outline_hidden(self):
        # Regression: Blender 5 only fills strokes with a non-zero `fill_id`;
        # without it the file contained all the shapes but nothing filled.
        obj = self.build("1f602")
        for layer in obj.data.layers:
            drawing = layer.frames[0].drawing
            self.assertTrue(all(v.value > 0 for v in drawing.attributes["fill_id"].data), layer.name)
            self.assertTrue(all(v.value for v in drawing.attributes["hide_stroke"].data), layer.name)
            ids = [v.value for v in drawing.attributes["fill_id"].data]
            self.assertEqual(len(ids), len(set(ids)), layer.name)

    def test_colour_is_converted_to_linear(self):
        obj = self.build("1f602")
        mat = next(m for m in obj.data.materials if m.name == "emoji_ffcc4d")
        r, g, b, a = mat.grease_pencil.fill_color
        self.assertAlmostEqual(r, 1.0, places=3)
        self.assertAlmostEqual(g, 0.604, places=3)
        self.assertAlmostEqual(b, 0.074, places=3)

    def test_face_map_is_stored_on_the_object(self):
        obj = self.build("1f602")
        fm = json.loads(obj["face_map"])
        self.assertIn("mouth", fm["parts"])
        self.assertEqual(obj["emoji_code"], "1f602")


if __name__ == "__main__":
    unittest.main()
