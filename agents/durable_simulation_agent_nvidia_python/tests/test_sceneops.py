"""The OpenUSD side, on the host: the helper API writes one sublayer per step, the lint rejects what Kit must never open,
the trusted judge sees the sandbox's violations, one call fixes a layout, a rebuilt sandbox recomposes the same layer
stack hash, and the Workflow's site constants match the site layer."""

from __future__ import annotations

from pathlib import Path
from textwrap import indent

import pytest
from pxr import Gf, Usd, UsdGeom

from activities.openshell import FakeSandboxes
from paths import CATALOG, SITE_LAYER
from sceneops.catalog import S3_ROOT, load_catalog, merge_hits
from sceneops.hashing import layer_stack_hash, sha256_file
from site_facts import BUILD_AREA, CAMERAS, DOCKS, SITE_VOLUMES
from validate.judge import judge_layers, lane_check
from validate.lint import lint_layer
from validate.plan import render_plan

CODE_PLACE = ("s1 = h.place_asset('packing_table', -3.0, 2.0, 90, name='station_1')\n"
              "s2 = h.place_asset('packing_table', -3.0, -1.0, 90, name='station_2')\n"
              "p = h.place_asset('pallet', 1.7, -3.0, 0, name='staging_1')\n"
              "h.add_floor_marking('cell_edge', [(-4.5, -4.0), (-4.5, 4.0)])\n"
              "h.add_light('task_light', -3.0, 0.5)\n"
              "print('violations', len(h.check_clearance()))\n")
CODE_FIX = "moves = h.fix_all()\nprint(moves)\n"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("SANDBOX_BACKEND", "fake")
    return FakeSandboxes()


async def test_apply_lint_judge_fix_rehydrate_and_plan(fake, tmp_path):
    await fake.ensure("x1", "exec", [])
    env = (await fake.apply("x1", 1, CODE_PLACE, {"model": "test"}, 0.6)).envelope
    assert env["ok"] and env["cached"] is False and "violations 1" in env["data"]["stdout_tail"]
    assert [(v["asset"], v["volume"], v["status"]) for v in env["data"]["precheck_violations"]] == [("/World/Agent/staging_1", "L2", "too_close")]
    layer1 = tmp_path / "001.usda"
    await fake.download_layer("x1", 1, layer1)
    assert sha256_file(layer1) == env["data"]["sha256"]
    assert lint_layer(layer1, catalog=CATALOG, asset_root="/opt/assets").ok
    text = layer1.read_text()
    assert "dictionary agent" in text and "code_sha256" in text, "the layer carries its provenance in customLayerData"

    verdict = judge_layers([layer1], site_layer=SITE_LAYER, catalog=CATALOG, min_clearance_m=0.6)
    assert [v["asset"] for v in verdict["violations"]] == ["/World/Agent/staging_1"] and not verdict["passed"]
    assert sorted(a["catalog_id"] for a in verdict["placed_assets"]) == ["packing_table", "packing_table", "pallet"]
    assert sorted(p["kind"] for p in verdict["agent_prims"]) == ["asset", "asset", "asset", "floor_marking", "light"]
    lane = lane_check([layer1], site_layer=SITE_LAYER, catalog=CATALOG, lane="L2")
    assert lane["passed"] is False and lane["contacts"] == ["/World/Agent/staging_1"] and lane["reached"] is True and lane["method"] == "geometry"
    assert 0 < lane["min_clearance_m"] < 0.6

    # the same code for the same step again (a retried Activity): the cached result, no second run
    assert (await fake.apply("x1", 1, CODE_PLACE, {"model": "test"}, 0.6)).envelope["cached"] is True

    env2 = (await fake.apply("x1", 2, CODE_FIX, {}, 0.6)).envelope
    assert env2["ok"] and env2["data"]["precheck_violations"] == [] and "'moved': True" in env2["data"]["stdout_tail"]
    layer2 = tmp_path / "002.usda"
    await fake.download_layer("x1", 2, layer2)
    v2 = judge_layers([layer1, layer2], site_layer=SITE_LAYER, catalog=CATALOG, min_clearance_m=0.6)
    assert v2["passed"] and lane_check([layer1, layer2], site_layer=SITE_LAYER, catalog=CATALOG, lane="L2")["passed"]
    moved = next(a for a in v2["placed_assets"] if a["path"].endswith("staging_1"))
    assert moved["x"] < 0.3, "nudged back inside the build area, away from the dock lane"
    pairs = [(1, sha256_file(layer1)), (2, sha256_file(layer2))]
    assert env2["data"]["layer_stack_hash"] == layer_stack_hash(pairs)

    # a rebuilt sandbox recomposes the same stack from the recorded layers
    await fake.ensure("x2", "exec", [])
    await fake.upload_layer("x2", layer1, 1)
    await fake.upload_layer("x2", layer2, 2)
    rr = (await fake.rehydrate("x2", pairs)).envelope
    assert rr["ok"] and rr["data"]["layer_stack_hash"] == layer_stack_hash(pairs) and [s["step"] for s in rr["data"]["layers_applied"]] == [1, 2]
    bad = (await fake.rehydrate("x2", [(1, "0" * 64), (2, pairs[1][1])])).envelope
    assert bad["ok"] is False and bad["error"]["type"] == "RehydrateMismatch"

    out = render_plan([layer1, layer2], site_layer=SITE_LAYER, catalog=CATALOG, out=tmp_path / "plan.png", violations=verdict["violations"])
    assert out.stat().st_size > 2000


async def test_code_mistakes_come_back_as_validation_errors(fake):
    await fake.ensure("x3", "exec", [])
    r = await fake.apply("x3", 1, "h.place_asset('/some/path/not_in_catalog.usd', 0, 0)\n", {}, 0.6)
    assert r.exit_code == 2 and r.envelope["ok"] is False and r.envelope["error"]["type"] == "ValueError" and "only catalog ids" in r.envelope["error"]["msg"]
    r = await fake.apply("x3", 1, "h.place_asset('no_such_asset', 0, 0)\n", {}, 0.6)
    assert r.envelope["error"]["type"] == "ValueError" and "nearest" in r.envelope["error"]["msg"]
    r = await fake.apply("x3", 1, "print(\n", {}, 0.6)
    assert r.envelope["error"]["type"] == "SyntaxError"
    r = await fake.apply("x3", 1, "stage.GetRootLayer().subLayerPaths.append('x.usda')\nh.place_asset('pallet', 0, 0)\n", {}, 0.6)
    assert r.envelope["error"]["type"] == "EditEscaped"


def test_lint_rejects_what_kit_must_never_open(tmp_path):
    bad = tmp_path / "bad.usda"
    bad.write_text('#usda 1.0\n(\n    subLayers = [@../other.usda@]\n)\n\ndef Xform "World"\n{\n    def Xform "Agent"\n    {\n'
                   '        def Xform "a" (references = @http://evil.example/x.usd@)\n        {\n            custom string omni:scripting:scripts = "x"\n        }\n'
                   '        def Camera "c"\n        {\n        }\n    }\n    def Xform "Elsewhere"\n    {\n    }\n}\n')
    rep = lint_layer(bad, catalog=CATALOG, asset_root="/opt/assets")
    msgs = " | ".join(rep.errors)
    assert not rep.ok
    for needle in ("sublayers", "is not a catalog asset", "forbidden property", "/World/Elsewhere", "prim type 'Camera'"):
        assert needle in msgs, msgs
    assert lint_layer(SITE_LAYER, catalog=CATALOG, asset_root="/opt/assets").ok is False, "the site layer itself is not an agent layer"


def test_site_constants_match_the_layer():
    stage = Usd.Stage.Open(str(SITE_LAYER))
    assert UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])

    def aabb(path: str) -> list[float]:
        r = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        return [round(v, 3) for v in (r.GetMin()[0], r.GetMin()[1], r.GetMax()[0], r.GetMax()[1])]

    for n, (x, y) in DOCKS.items():
        t = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(f"/World/Site/Dock_{n}")).GetXformVectors(Usd.TimeCode.Default())[0]
        assert [round(t[0], 3), round(t[1], 3)] == [x, y], n
    for v in SITE_VOLUMES:
        assert aabb(f"/World/Site/{'Lanes' if v['kind'] == 'lane' else 'KeepClear'}/{v['name']}") == v["aabb"], v["name"]
    assert aabb("/World/Site/Candidates/BuildArea") == BUILD_AREA
    for cam in CAMERAS:
        prim = stage.GetPrimAtPath(f"/World/Cameras/{cam}")
        assert prim and prim.GetTypeName() == "Camera", cam
        if cam != "TopDown":
            fwd = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).TransformDir(Gf.Vec3d(0, 0, -1))
            assert fwd[1] > 0.5, f"{cam} should look north toward the racks"


def test_search_hits_map_to_catalog_ids_and_the_catalog_fills_in():
    catalog = load_catalog(CATALOG)
    hits = [{"url": S3_ROOT + "Isaac/Props/PackingTable/packing_table.usd", "score": 0.91}, {"url": S3_ROOT + "Somewhere/unknown_thing.usd", "score": 0.5}]
    out = merge_hits(hits, catalog, "pallet staging")
    assert out[0]["id"] == "packing_table" and out[0]["source"] == "usd_search" and out[0]["size_m"] == catalog["packing_table"]["size_m"]
    assert out[1]["source"] == "catalog" and "pallet" in [o["id"] for o in out[1:]]
    assert merge_hits(hits, catalog, "unmatched term")[1]["id"] is None
    unmapped_hits = [{"url": S3_ROOT + f"Unlisted/asset_{i}.usd"} for i in range(8)]
    assert "pallet" in [r["id"] for r in merge_hits(unmapped_hits, catalog, "pallet")]


def _layer(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text("#usda 1.0\n\n" + body)
    return p


def _agent_layer(tmp_path: Path, body: str) -> Path:
    return _layer(tmp_path, "agent.usda", 'over "World"\n{\n    over "Agent"\n    {\n' + indent(body, "        ") + '\n    }\n}\n')


def test_lint_refuses_payloads_variants_and_asset_paths_anywhere(tmp_path):
    cases = {
        "a payload on the /World prefix": ('over "World" (\n    prepend payload = @http://evil.example/x.usd@\n)\n{\n    over "Agent"\n    {\n'
                                           '        def Xform "ok"\n        {\n        }\n    }\n}\n', "/World"),
        "a reference inside a variant": ('over "World"\n{\n    over "Agent"\n    {\n        def Xform "v" (\n            variants = {\n                string look = "a"\n'
                                         '            }\n            prepend variantSets = "look"\n        )\n        {\n            variantSet "look" = {\n'
                                         '                "a" (\n                    prepend references = @/etc/evil.usda@\n                ) {\n                }\n'
                                         '            }\n        }\n    }\n}\n', "variants are not allowed"),
        "an inherit": ('over "World"\n{\n    over "Agent"\n    {\n        def Xform "i" (\n            inherits = </World/Site/Lanes/L2>\n        )\n'
                       '        {\n        }\n    }\n}\n', "metadata 'inheritPaths' not allowed"),
        "a time-sampled asset array outside the root": ('over "World"\n{\n    over "Agent"\n    {\n        def Xform "a"\n        {\n'
                                                        '            asset[] files.timeSamples = {\n                1: [@/opt/assets/ok.usd@, @/etc/passwd@],\n'
                                                        '            }\n        }\n    }\n}\n', "time samples and splines are not allowed"),
    }
    for i, (what, (body, needle)) in enumerate(cases.items()):
        rep = lint_layer(_layer(tmp_path, f"case{i}.usda", body), catalog=CATALOG, asset_root="/opt/assets")
        assert not rep.ok and any(needle in e for e in rep.errors), (what, rep.errors)


def _author(tmp_path: Path, asset_root: Path, steps: list[str]) -> list[Path]:
    """Author one layer per step the way sceneops.apply does: the new layer is the edit target over the older ones."""
    from pxr import Sdf

    from sceneops.helpers import SceneHelpers

    paths: list[Path] = []
    for i, code in enumerate(steps, 1):
        path = tmp_path / f"{i:03d}.usda"
        layer = Sdf.Layer.CreateNew(str(path))
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().subLayerPaths = [str(path), *[str(p) for p in reversed(paths)], str(SITE_LAYER)]
        stage.SetEditTarget(Usd.EditTarget(layer))
        ns = {"h": SceneHelpers(stage, load_catalog(CATALOG), asset_root=str(asset_root)), "stage": stage, "UsdGeom": UsdGeom, "Gf": Gf}  # what sceneops.apply provides
        exec(code, ns)  # noqa: S102 — test code
        layer.Save()
        paths.append(path)
    return paths


def test_a_re_placed_name_keeps_only_its_new_asset(tmp_path):
    assets = tmp_path / "assets"
    for rel, mesh in (("Isaac/Props/PackingTable/packing_table.usd", "PackingTableMesh"), ("Isaac/Props/Pallet/pallet.usd", "PalletMesh")):
        f = assets / rel
        f.parent.mkdir(parents=True)
        f.write_text(f'#usda 1.0\n(\n    defaultPrim = "Root"\n)\n\ndef Xform "Root"\n{{\n    def Cube "{mesh}"\n    {{\n    }}\n}}\n')
    layers = _author(tmp_path, assets, ["h.place_asset('packing_table', -3.0, 2.0, name='station_1')",
                                        "h.place_asset('pallet', -3.0, 2.0, name='station_1')",
                                        "h.add_floor_marking('station_2', [(-4.0, -4.0), (-4.0, 0.0)])\nh.place_asset('pallet', -2.0, -2.0, name='station_2')",
                                        "h.add_floor_marking('station_2', [(-4.5, -4.0), (-4.5, 0.0)])"])
    assert all(lint_layer(layer, catalog=CATALOG, asset_root=str(assets)).ok for layer in layers)

    def compose(n: int) -> Usd.Stage:
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().subLayerPaths = [str(p) for p in reversed(layers[:n])] + [str(SITE_LAYER)]
        return stage

    st = compose(2)
    assert [c.GetName() for c in st.GetPrimAtPath("/World/Agent/station_1").GetChildren()] == ["PalletMesh"], "the older reference is gone"
    st = compose(4)
    marking = st.GetPrimAtPath("/World/Agent/station_2")
    assert marking.GetTypeName() == "Mesh" and not marking.GetChildren() and marking.GetCustomDataByKey("agent:catalog_id") == ""
    assert UsdGeom.Xformable(marking).GetOrderedXformOps() == [], "no transform inherited from the pallet it replaced"
    verdict = judge_layers(layers, site_layer=SITE_LAYER, catalog=CATALOG, asset_root=str(assets))
    assert [a["path"] for a in verdict["placed_assets"]] == ["/World/Agent/station_1"]
    assert {p["path"]: p["kind"] for p in verdict["agent_prims"]}["/World/Agent/station_2"] == "floor_marking"


def test_the_judge_measures_what_is_really_there(tmp_path):
    """Assets found by their references and measured with full transforms; geometry counts; overhead items do not."""
    crate = load_catalog(CATALOG)["crate_plastic_a"]
    x = 1.5 - crate["local_bbox"][2]  # unscaled, the crate ends 0.9 m short of L2 (x 2.4)
    s = (3.0 - x) / crate["local_bbox"][2]  # scaled, it reaches 0.6 m into L2
    layers = _author(tmp_path, Path("/opt/assets"), [
        f"p = h.place_asset('crate_plastic_a', {x}, 2.0, name='crate')\nUsdGeom.XformCommonAPI(stage.GetPrimAtPath(p)).SetScale(Gf.Vec3f({s}, {s}, 1.0))",
        "q = stage.DefinePrim('/World/Agent/untagged', 'Xform')\nq.GetReferences().AddReference('/opt/assets/Isaac/Props/Pallet/pallet.usd')\n"
        "UsdGeom.XformCommonAPI(q).SetTranslate(Gf.Vec3d(3.4, -8.0, 0.0))",
        "c = UsdGeom.Cube.Define(stage, '/World/Agent/wall')\nc.CreateSizeAttr(1.0)\nUsdGeom.XformCommonAPI(c).SetTranslate(Gf.Vec3d(3.4, -15.0, 0.5))",
        "h.place_asset('lamp_ceiling', 3.4, -5.0, name='lamp')",
    ])
    verdict = judge_layers(layers, site_layer=SITE_LAYER, catalog=CATALOG, min_clearance_m=0.6)
    flagged = {(v["asset"], v["volume"]) for v in verdict["violations"]}
    assert ("/World/Agent/crate", "L2") in flagged, "a scale op does not hide the crate"
    assert ("/World/Agent/untagged", "L2") in flagged, "an asset without the helper's custom data is still an asset"
    assert ("/World/Agent/wall", "L2") in flagged, "hand-built geometry is an obstacle too"
    assert not any(a == "/World/Agent/lamp" for a, _ in flagged), "a ceiling lamp over the lane blocks nothing"
    assert {a["path"] for a in verdict["placed_assets"]} >= {"/World/Agent/crate", "/World/Agent/untagged", "/World/Agent/lamp"}
    lane = lane_check(layers, site_layer=SITE_LAYER, catalog=CATALOG, lane="L2")
    assert lane["passed"] is False and lane["reached"] is False and lane["min_clearance_m"] == 0.0
    assert set(lane["contacts"]) == {"/World/Agent/crate", "/World/Agent/untagged", "/World/Agent/wall"}


def test_lint_allows_only_catalog_references_on_direct_children(tmp_path):
    def lint(body: str):
        return lint_layer(_layer(tmp_path, "l.usda", 'over "World"\n{\n    over "Agent"\n    {\n' + body + '    }\n}\n'), catalog=CATALOG, asset_root="/opt/assets")

    assert lint('        def Xform "ok" (\n            prepend references = @/opt/assets/Isaac/Props/Pallet/pallet.usd@\n        )\n        {\n        }\n').ok
    graph = lint('        def Xform "g" (\n            prepend references = @/opt/assets/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd@\n'
                 '        )\n        {\n        }\n')
    assert not graph.ok and "is not a catalog asset" in graph.errors[0], "under the asset root is not enough: that sample carries OmniGraph nodes"
    internal = lint('        def Xform "i" (\n            prepend references = </World/Site/Lanes/L2>\n        )\n        {\n        }\n')
    assert not internal.ok and "internal references are not allowed" in internal.errors[0]
    nested = lint('        def Xform "group"\n        {\n            def Cube "hidden"\n            {\n            }\n        }\n')
    assert not nested.ok and "only direct children" in nested.errors[0]


@pytest.mark.parametrize("prim", ["over", "def", "def Cube"])
def test_catalog_reference_requires_a_defining_xform(tmp_path, prim):
    asset = load_catalog(CATALOG)["pallet"]["usd"]
    layer = _agent_layer(tmp_path, f'{prim} "Pallet" (\n    references = @/opt/assets/{asset}@\n)\n{{\n}}')
    report = lint_layer(layer, catalog=CATALOG)
    assert not report.ok and any("requires a defined Xform" in e for e in report.errors)


@pytest.mark.parametrize(("attr", "reason"), [
    ("double size.timeSamples = {0: 1, 1: 2}", "time samples"),
    ("float3[] extent = [(-1, -1, -1), (1, 1, 1)]", "forbidden property"),
    ('uniform token purpose = "proxy"', "forbidden property"),
    ("asset[] source.timeSamples = {1: None}", "time samples"),
])
def test_agent_geometry_uses_static_computed_bounds(tmp_path, attr, reason):
    layer = _agent_layer(tmp_path, f'def Cube "Cube"\n{{\n    {attr}\n}}')
    report = lint_layer(layer, catalog=CATALOG)
    assert not report.ok and any(reason in e for e in report.errors)


def test_lint_accepts_value_blocked_asset_defaults(tmp_path):
    layer = _agent_layer(tmp_path, 'def Xform "Asset"\n{\n    asset file = None\n    asset[] files = None\n}')
    assert lint_layer(layer, catalog=CATALOG).ok


async def test_a_damaged_cache_record_or_layer_file_is_redone_not_fatal(fake):
    await fake.ensure("x4", "exec", [])
    code = "h.place_asset('pallet', -2.0, -2.0, name='p1')\n"
    assert (await fake.apply("x4", 1, code, {}, 0.6)).envelope["ok"]
    layers = fake.root / "x4" / "work" / "layers"
    record = layers / "001.result.json"
    record.write_text(record.read_text()[:40])  # a killed attempt left half a record
    again = (await fake.apply("x4", 1, code, {}, 0.6)).envelope
    assert again["ok"] and again["cached"] is False, "a damaged record is a cache miss"
    (layers / "001.usda").write_text("#usda 1.0\\n(\\n  this is not usda")  # and half a layer
    record.unlink()
    other = (await fake.apply("x4", 1, "h.place_asset('pallet', -3.0, -2.0, name='p2')\n", {}, 0.6)).envelope
    assert other["ok"] and [a["path"] for a in other["data"]["placed_assets"]] == ["/World/Agent/p2"]
    assert not list(layers.glob("*.part")), "results are renamed into place"
