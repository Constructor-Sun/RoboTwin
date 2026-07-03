import numpy as np


DEFAULT_BACKGROUND_APPEARANCE_CONFIG = {
    "enabled": False,
    "b1_scene_theme": True,
    "b1_wall_rgb_multiplier_range": [0.4, 1.8],
    "b1_floor_color_range": [0.4, 1.8],
    "b2_surface_appearance": True,
    "b2_table_metallic_range": [0.0, 0.8],
    "b2_table_roughness_range": [0.05, 0.95],
    "b2_table_color_tint_range": [0.4, 1.8],
}


def merge_background_appearance_config(config):
    merged = dict(DEFAULT_BACKGROUND_APPEARANCE_CONFIG)
    merged.update(config or {})
    return merged


def _sample_rgb_multiplier(rng, value_range):
    low, high = value_range
    return rng.uniform(float(low), float(high), size=3).tolist()


def sample_background_appearance(config, rng):
    config = merge_background_appearance_config(config)
    enabled = bool(config.get("enabled", False))
    summary = {
        "enabled": enabled,
        "b1_scene_theme": False,
        "wall_rgb_multiplier": None,
        "floor_color": None,
        "b2_surface_appearance": False,
        "table_metallic": None,
        "table_roughness": None,
        "table_color_tint": None,
    }

    if not enabled:
        return config, summary

    if config.get("b1_scene_theme", True):
        summary["b1_scene_theme"] = True
        summary["wall_rgb_multiplier"] = _sample_rgb_multiplier(
            rng, config["b1_wall_rgb_multiplier_range"]
        )
        summary["floor_color"] = _sample_rgb_multiplier(
            rng, config["b1_floor_color_range"]
        )

    if config.get("b2_surface_appearance", True):
        summary["b2_surface_appearance"] = True
        metallic_range = config["b2_table_metallic_range"]
        roughness_range = config["b2_table_roughness_range"]
        summary["table_metallic"] = float(
            rng.uniform(float(metallic_range[0]), float(metallic_range[1]))
        )
        summary["table_roughness"] = float(
            rng.uniform(float(roughness_range[0]), float(roughness_range[1]))
        )
        summary["table_color_tint"] = _sample_rgb_multiplier(
            rng, config["b2_table_color_tint_range"]
        )

    return config, summary


def create_render_material(color=None, texture_path=None, metallic=0.1, roughness=0.3):
    import sapien.core as sapien

    material = sapien.render.RenderMaterial()
    if texture_path is not None:
        material.set_base_color_texture(sapien.render.RenderTexture2D(texture_path))
    rgb = [1, 1, 1] if color is None else list(color[:3])
    material.base_color = [*rgb, 1]
    material.metallic = float(metallic)
    material.roughness = float(roughness)
    return material


def _add_floor_visual(scene, height, material):
    import sapien.core as sapien

    entity = sapien.Entity()
    entity.set_name("ground_visual")
    entity.set_pose(sapien.Pose([0, 0, height - 0.002]))
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        sapien.render.RenderShapeBox([6.0, 6.0, 0.001], material)
    )
    entity.add_component(render_component)
    scene.add_entity(entity)
    return entity


def get_wall_material_kwargs(summary):
    tint = summary.get("wall_rgb_multiplier")
    if not summary.get("enabled", False) or tint is None:
        return {}
    return {"color_tint": tint}


def get_table_material_kwargs(summary):
    if not summary.get("enabled", False) or not summary.get("b2_surface_appearance", False):
        return {}
    return {
        "color_tint": summary.get("table_color_tint"),
        "metallic": summary.get("table_metallic"),
        "roughness": summary.get("table_roughness"),
    }


def add_ground_with_optional_material(scene, height, summary):
    floor_color = summary.get("floor_color")
    if not summary.get("enabled", False) or floor_color is None:
        return scene.add_ground(height)

    material = create_render_material(color=floor_color, metallic=0.0, roughness=0.8)
    try:
        return scene.add_ground(height, render_material=material)
    except TypeError:
        try:
            ground = scene.add_ground(height, material=material)
            _add_floor_visual(scene, height, material)
            return ground
        except TypeError:
            ground = scene.add_ground(height)
            _add_floor_visual(scene, height, material)
            return ground
