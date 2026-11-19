"""The helper API as the model sees it. Kept free of pxr imports so the Workflow module can put it in the instructions."""

CHEATSHEET = """You author OpenUSD edits through the helper object `h` (plus `stage`, `Usd`, `Sdf`, `UsdGeom`, `Gf`).
Units: meters, Z-up, floor at z=0. Only catalog assets exist; get ids from search_assets.
  h.search(query, k=8) -> [{id, name, size_m:[w,d,h], tags}]
  h.dock(n) -> (x, y)                              # dock marker position, n in 1..4
  h.site_volumes() -> [{name, kind: lane|keep_clear, aabb:[xmin,ymin,xmax,ymax]}]
  h.place_asset(catalog_id, x, y, yaw_deg=0, name=None) -> prim_path   # catalog ids only (never file paths); yaw about Z; local X = long side
  h.move(prim_path, dx, dy) ; h.set_position(prim_path, x, y) ; h.set_rotation(prim_path, yaw_deg) ; h.remove(prim_path) ; h.clear()
  h.add_floor_marking(name, [(x,y),(x,y),...], width=0.08, color=(1,0.85,0)) -> prim_path
  h.add_light(name, x, y, z=3.5, intensity=20000) -> prim_path
  h.footprint(prim_path) -> [xmin,ymin,xmax,ymax] ; h.distance_to_dock(prim_path, n) -> meters
  h.check_clearance(min_clearance_m=0.6) -> [{asset, volume, kind, clearance_m, required_m, status}]   # [] means OK
  h.agent_prims() -> [{path, kind: asset|floor_marking|light, catalog_id, aabb}]   # everything you have authored so far
  h.build_area() -> [xmin,ymin,xmax,ymax]                                            # the open floor meant for the brief
  h.footprint_at(catalog_id, x, y, yaw_deg=0) -> [xmin,ymin,xmax,ymax]              # real footprint (pivots are NOT centred)
  h.conflicts_for(catalog_id, x, y, yaw_deg=0, exclude=None) -> [{with, kind}]     # [] means that placement is free
  h.conflicts(x, y, w, d, yaw_deg=0, exclude=None) -> [{with, kind}]                 # same for a plain w x d rectangle centred at (x,y)
  h.find_spot(catalog_id, region=None, yaw_deg=0, step=0.25, prefer=(x,y)) -> (x, y) or None   # first free spot, nearest `prefer`
  h.nudge_clear(prim_path, max_m=3.0) -> {moved, from, to, conflicts_left}   # least move that clears lanes/keep-clear/overlaps
  h.fix_all() -> [moves]                                                        # nudge_clear every conflicting asset (one call fixes a layout)
Plan the whole layout in ONE apply: use h.find_spot/h.conflicts in loops, then h.place_asset each item with an explicit name.
Naming: an explicit name REPLACES an existing prim of that name (re-placing is how you move things); unnamed assets get the catalog id.
Rules: keep every asset >= the required clearance from every lane and keep_clear volume (check with h.check_clearance()).
Place things on open floor; do not overlap assets with each other. Prefer a few well-placed assets over many.
"""
