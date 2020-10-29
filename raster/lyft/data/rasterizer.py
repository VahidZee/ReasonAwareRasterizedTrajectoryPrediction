import types

from l5kit.data.filter import filter_tl_faces_by_status
from l5kit.data.map_api import MapAPI
from l5kit.data import get_frames_slice_from_scenes
from l5kit.dataset import AgentDataset
from l5kit.geometry import rotation33_as_yaw, transform_point, transform_points
from l5kit.rasterization.semantic_rasterizer import elements_within_bounds, cv2_subpixel
from l5kit.rasterization import SemanticRasterizer, SemBoxRasterizer
from l5kit.rasterization import build_rasterizer as _build_rasterizer
from typing import List, Optional
import numpy as np
import drawSvg as draw
import warnings
import functools

CV2_SHIFT = 8
CV2_SHIFT_VALUE = 2 ** CV2_SHIFT

a, b, c, d = None, None, None, None


def render_semantic_map(self, center_in_world: np.ndarray, raster_from_world: np.ndarray,
                        tl_faces: np.ndarray = None, svg_args=None, face_color=False) -> draw.Drawing:
    """Renders the semantic map at given x,y coordinates.
    Args:
        center_in_world (np.ndarray): XY of the image center in world ref system
        raster_from_world (np.ndarray):
    Returns:
        drawvg.Drawing object
    """
    # filter using half a radius from the center
    raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2
    svg_args = svg_args or dict()

    # get active traffic light faces
    if face_color:
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())

    # setup canvas
    raster_size = self.render_context.raster_size_px
    origin = self.render_context.origin
    d = draw.Drawing(*raster_size, origin=tuple(origin), displayInline=False, **svg_args)

    for idx in elements_within_bounds(center_in_world, self.bounds_info["lanes"]["bounds"], raster_radius):
        lane = self.proto_API[self.bounds_info["lanes"]["ids"][idx]].element.lane

        # get image coords
        lane_coords = self.proto_API.get_lane_coords(self.bounds_info["lanes"]["ids"][idx])
        xy_left = cv2_subpixel(transform_points(lane_coords["xyz_left"][:, :2], raster_from_world))
        xy_right = cv2_subpixel(transform_points(lane_coords["xyz_right"][:, :2], raster_from_world))

        if face_color:
            lane_type = "black"  # no traffic light face is controlling this lane
            lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
            for tl_id in lane_tl_ids.intersection(active_tl_ids):
                if self.proto_API.is_traffic_face_colour(tl_id, "red"):
                    lane_type = "red"
                elif self.proto_API.is_traffic_face_colour(tl_id, "green"):
                    lane_type = "green"
                elif self.proto_API.is_traffic_face_colour(tl_id, "yellow"):
                    lane_type = "yellow"

        for line in [xy_left, xy_right]:
            vector = line / CV2_SHIFT_VALUE
            vector[:, 1] = raster_size[1] + origin[1] - vector[:, 1]
            vector[:, 0] = vector[:, 0] + origin[0]
            drawn_shape = draw.Lines(*vector.reshape(-1)) if not face_color else draw.Lines(
                *vector.reshape(-1), close=False, stroke=lane_type)
            d.append(drawn_shape)
    return d


def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
) -> draw.Drawing:
    if agent is None:
        ego_translation_m = history_frames[0]["ego_translation"]
        ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])
    else:
        ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
        ego_yaw_rad = agent["yaw"]

    raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
    world_from_raster = np.linalg.inv(raster_from_world)

    # get XY of center pixel in world coordinates
    center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
    center_in_world_m = transform_point(center_in_raster_px, world_from_raster)

    return self.render_semantic_map(center_in_world_m, raster_from_world, history_tl_faces[0])


def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
    """
    A utility function to get the rasterisation and trajectory target for a given agent in a given frame
    Args:
        scene_index (int): the index of the scene in the zarr
        state_index (int): a relative frame index in the scene
        track_id (Optional[int]): the agent to rasterize or None for the AV
    Returns:
        dict: the rasterised image in (Cx0x1) if the rast is not None, the target trajectory
        (position and yaw) along with their availability, the 2D matrix to center that agent,
        the agent track (-1 if ego) and the timestamp
    """
    frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]
    tl_faces = self.dataset.tl_faces
    try:
        if self.cfg["raster_params"]["disable_traffic_light_faces"]:
            tl_faces = np.empty(0, dtype=self.dataset.tl_faces.dtype)  # completely disable traffic light faces
    except KeyError:
        warnings.warn(
            "disable_traffic_light_faces not found in config, this will raise an error in the future",
            RuntimeWarning,
            stacklevel=2,
        )
    data = self.sample_function(state_index, frames, self.dataset.agents, tl_faces, track_id)

    target_positions = np.array(data["target_positions"], dtype=np.float32)
    target_yaws = np.array(data["target_yaws"], dtype=np.float32)

    history_positions = np.array(data["history_positions"], dtype=np.float32)
    history_yaws = np.array(data["history_yaws"], dtype=np.float32)

    timestamp = frames[state_index]["timestamp"]
    track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

    result = {
        "target_positions": target_positions,
        "target_yaws": target_yaws,
        "target_availabilities": data["target_availabilities"],
        "history_positions": history_positions,
        "history_yaws": history_yaws,
        "history_availabilities": data["history_availabilities"],
        "world_to_image": data["raster_from_world"],  # TODO deprecate
        "raster_from_world": data["raster_from_world"],
        "raster_from_agent": data["raster_from_agent"],
        "agent_from_world": data["agent_from_world"],
        "world_from_agent": data["world_from_agent"],
        "track_id": track_id,
        "timestamp": timestamp,
        "centroid": data["centroid"],
        "yaw": data["yaw"],
        "extent": data["extent"],
    }

    # when rast is None, image could be None
    image = data["image"]
    if image is not None:
        # 0,1,C -> C,0,1
        result["image"] = image
    return result


def build_rasterizer(config, data_manager, svg=False, svg_args=None, face_color=True):
    rasterizer = _build_rasterizer(config, data_manager)
    if svg:
        rasterizer.render_context.origin = rasterizer.render_context.raster_size_px * \
                                           rasterizer.render_context.center_in_raster_ratio
        render_semantics = functools.partial(render_semantic_map, svg_args=svg_args, face_color=face_color)
        if isinstance(rasterizer, SemanticRasterizer):
            rasterizer.render_semantic_map = types.MethodType(render_semantics, rasterizer)
            rasterizer.rasterize = types.MethodType(rasterize, rasterizer)

        if isinstance(rasterizer, SemBoxRasterizer):
            rasterizer.sat_rast.render_semantic_map = types.MethodType(render_semantics, rasterizer.sat_rast)
            rasterizer.rasterize = types.MethodType(rasterize, rasterizer.sat_rast)
    return rasterizer


def agent_dataset(cfg: dict, zarr_dataset, rasterizer, perturbation=None, agents_mask=None, min_frame_history=10,
                  min_frame_future=1, svg=False):
    data = AgentDataset(cfg, zarr_dataset, rasterizer, perturbation, agents_mask, min_frame_history, min_frame_future)
    if svg:
        data.get_frame = types.MethodType(get_frame, data)
    return data
