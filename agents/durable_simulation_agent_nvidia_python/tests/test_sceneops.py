"""The OpenUSD side, on the host: the helper API writes one sublayer per step, the lint rejects what Kit must never open,
the trusted judge sees the sandbox's violations, one call fixes a layout, a rebuilt sandbox recomposes the same layer
stack hash, and the Workflow's site constants match the site layer."""

from __future__ import annotations

from pathlib import Path

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
    assert lint_layer(layer1, asset_root="/opt/assets").ok
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
    rep = lint_layer(bad, asset_root="/opt/assets")
    msgs = " | ".join(rep.errors)
    assert not rep.ok
    for needle in ("sublayers", "outside /opt/assets", "forbidden property", "/World/Elsewhere", "prim type 'Camera'"):
        assert needle in msgs, msgs
    assert lint_layer(SITE_LAYER, asset_root="/opt/assets").ok is False, "the site layer itself is not an agent layer"


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
    assert out[1]["id"] is None, "an unmapped hit is evidence the search ran, not something to place"
    assert "pallet" in [o["id"] for o in out[2:]] and all(o["source"] == "catalog" for o in out[2:])
    assert Path(CATALOG).exists()
