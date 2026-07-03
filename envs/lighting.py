import math
import numpy as np


DEFAULT_LIGHTING_CONFIG = {
    "enabled": False,
    "l1_diffuse_color": True,
    "l1_rgb_range": [0.0, 3.5],
    "l2_direction": True,
    "l2_theta_deg_range": [8, 82],
    "l2_phi_rad_range": [0, 2 * math.pi],
    "l2_dramatic_side_prob": 0.35,
    "l2_dramatic_theta_deg_range": [68, 82],
    "l3_specular": True,
    "l3_prob": 0.5,
    "l3_specular_range": [0.3, 6.0],
    "l3_shininess_range": [10, 250],
    "l4_shadows": True,
    "l4_prob": 0.5,
    "point_light_scale": 1.0,
}


def _merge_config(config):
    merged = dict(DEFAULT_LIGHTING_CONFIG)
    merged.update(config or {})
    return merged


def direction_from_spherical(theta_deg, phi_rad):
    theta = np.deg2rad(theta_deg)
    x = math.sin(theta) * math.cos(phi_rad)
    y = math.sin(theta) * math.sin(phi_rad)
    z = -abs(math.cos(theta))
    vec = np.array([x, y, z], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        return [0.0, 0.0, -1.0]
    return (vec / norm).tolist()


def sample_lighting(config, rng):
    config = _merge_config(config)
    enabled = bool(config.get("enabled", False))
    summary = {
        "enabled": enabled,
        "l1_rgb_tint": None,
        "l2_direction": None,
        "l2_theta_deg": None,
        "l2_phi_rad": None,
        "l2_dramatic_side": False,
        "l3_enabled": False,
        "l3_specular_strength": None,
        "l3_shininess": None,
        "l4_shadow": None,
        "point_light_scale": float(config.get("point_light_scale", 1.0)),
        "l3_materials_updated": 0,
        "l3_materials_seen": 0,
        "l3_errors": [],
    }

    if not enabled:
        return config, summary

    if config.get("l1_diffuse_color", True):
        low, high = config["l1_rgb_range"]
        summary["l1_rgb_tint"] = rng.uniform(float(low), float(high), size=3).tolist()

    if config.get("l2_direction", True):
        dramatic = bool(rng.random() < float(config["l2_dramatic_side_prob"]))
        theta_range = config["l2_dramatic_theta_deg_range"] if dramatic else config["l2_theta_deg_range"]
        theta = float(rng.uniform(float(theta_range[0]), float(theta_range[1])))
        phi_range = config["l2_phi_rad_range"]
        phi = float(rng.uniform(float(phi_range[0]), float(phi_range[1])))
        summary["l2_dramatic_side"] = dramatic
        summary["l2_theta_deg"] = theta
        summary["l2_phi_rad"] = phi
        summary["l2_direction"] = direction_from_spherical(theta, phi)

    if config.get("l2_direction", True) and config.get("l4_shadows", True):
        summary["l4_shadow"] = bool(rng.random() < float(config["l4_prob"]))

    if config.get("l3_specular", True):
        l3_enabled = bool(rng.random() < float(config["l3_prob"]))
        summary["l3_enabled"] = l3_enabled
        if l3_enabled:
            specular_range = config["l3_specular_range"]
            shininess_range = config["l3_shininess_range"]
            summary["l3_specular_strength"] = float(rng.uniform(float(specular_range[0]), float(specular_range[1])))
            summary["l3_shininess"] = float(rng.uniform(float(shininess_range[0]), float(shininess_range[1])))

    return config, summary


def apply_lighting_to_light_specs(direction_lights, point_lights, lighting_summary):
    direction_lights = [[list(light[0]), list(light[1])] for light in direction_lights]
    point_lights = [[list(light[0]), list(light[1])] for light in point_lights]

    if not lighting_summary.get("enabled", False):
        return direction_lights, point_lights

    rgb_tint = lighting_summary.get("l1_rgb_tint")
    if rgb_tint is not None:
        for light in direction_lights:
            light[1] = list(rgb_tint)
        for light in point_lights:
            point_scale = float(lighting_summary.get("point_light_scale", 1.0))
            light[1] = (np.array(rgb_tint, dtype=np.float64) * point_scale).tolist()

    direction = lighting_summary.get("l2_direction")
    if direction is not None and direction_lights:
        direction_lights[0][0] = list(direction)

    return direction_lights, point_lights


def _iter_shape_materials(render_component):
    candidates = []
    for attr_name in ("render_shapes", "visual_shapes", "shapes"):
        if hasattr(render_component, attr_name):
            try:
                candidates.extend(list(getattr(render_component, attr_name)))
            except TypeError:
                pass

    for method_name in ("get_render_shapes", "get_visual_shapes", "get_shapes"):
        if hasattr(render_component, method_name):
            try:
                candidates.extend(list(getattr(render_component, method_name)()))
            except TypeError:
                pass

    for shape in candidates:
        material = None
        if hasattr(shape, "material"):
            material = getattr(shape, "material")
        elif hasattr(shape, "get_material"):
            try:
                material = shape.get_material()
            except TypeError:
                material = None
        if material is not None:
            yield material


def _iter_actor_materials(actor):
    components = []
    for attr_name in ("components",):
        if hasattr(actor, attr_name):
            try:
                components.extend(list(getattr(actor, attr_name)))
            except TypeError:
                pass

    for method_name in ("get_components",):
        if hasattr(actor, method_name):
            try:
                components.extend(list(getattr(actor, method_name)()))
            except TypeError:
                pass

    for component in components:
        yield from _iter_shape_materials(component)


def _set_first_available(obj, names, value):
    for name in names:
        if hasattr(obj, name):
            try:
                setattr(obj, name, value)
                return True
            except Exception:
                pass

        setter_name = f"set_{name}"
        if hasattr(obj, setter_name):
            try:
                getattr(obj, setter_name)(value)
                return True
            except Exception:
                pass

    return False


def apply_l3_specular(scene, lighting_summary):
    if not lighting_summary.get("enabled", False) or not lighting_summary.get("l3_enabled", False):
        return lighting_summary

    specular = lighting_summary.get("l3_specular_strength")
    shininess = lighting_summary.get("l3_shininess")
    if specular is None or shininess is None:
        return lighting_summary

    actors = []
    if hasattr(scene, "get_all_actors"):
        try:
            actors = list(scene.get_all_actors())
        except Exception as exc:
            lighting_summary["l3_errors"].append(f"get_all_actors failed: {exc}")
            return lighting_summary

    seen = 0
    updated = 0
    for actor in actors:
        try:
            for material in _iter_actor_materials(actor):
                seen += 1
                changed = False
                changed |= _set_first_available(material, ("specular", "specular_strength"), specular)
                changed |= _set_first_available(material, ("shininess",), shininess)

                # SAPIEN's PBR material may expose roughness instead of shininess.
                if not hasattr(material, "shininess"):
                    roughness = float(np.clip(1.0 - (shininess - 10.0) / 240.0, 0.02, 1.0))
                    changed |= _set_first_available(material, ("roughness",), roughness)

                if changed:
                    updated += 1
        except Exception as exc:
            actor_name = actor.get_name() if hasattr(actor, "get_name") else type(actor).__name__
            lighting_summary["l3_errors"].append(f"{actor_name}: {exc}")

    lighting_summary["l3_materials_seen"] = seen
    lighting_summary["l3_materials_updated"] = updated
    return lighting_summary
