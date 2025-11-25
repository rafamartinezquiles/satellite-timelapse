# This module is responsible for rendering a miniature tennis court overlay on top of existing video frames and  
# for converting player/ball positions from the original video coordinate system into the coordinate system of the mini court.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.config import CourtDimensions, PlayerHeights
from src.geometry_utils import (
    axis_distances,
    bbox_center,
    bbox_foot_point,
    bbox_height,
    closest_keypoint_index,
    convert_meters_to_pixels,
    convert_pixels_to_meters,
    euclidean_distance,
)

# Type aliases for better readability.
BBoxDict = Dict[int, Tuple[float, float, float, float]]

# MiniCourtPoint: (x, y) coordinate in the mini-court pixel space.
MiniCourtPoint = Tuple[float, float]


@dataclass
class MiniCourtLayout:
    """
    Aggregate parameters that define the layout and positioning of the mini court
    within the original frame.

    Attributes
    ----------
    rectangle_width_px : int
        Width of the mini court background rectangle in pixels.
    rectangle_height_px : int
        Height of the mini court background rectangle in pixels.
    outer_margin_px : int
        Margin from the frame edges to the background rectangle.
        The rectangle is anchored to the right side of the frame.
    court_padding_px : int
        Inner padding between the background rectangle and the actual court lines.
    """

    rectangle_width_px: int = 200
    rectangle_height_px: int = 360
    outer_margin_px: int = 50
    court_padding_px: int = 30


class MiniCourt:
    """
    Responsible for:

    1. Computing the mini-court layout and geometry
    2. Drawing
    3. Coordinate transformations
    """

    def __init__(self, reference_frame: np.ndarray, layout: MiniCourtLayout | None = None):
        """
        Initialize the mini court with respect to a reference frame.

        The reference_frame is used solely to determine the size of the original
        video frames so that the mini court can be positioned consistently
        (e.g., anchored to the right side).
        """
        self.layout = layout or MiniCourtLayout()
        self.court_dims = CourtDimensions()
        self.player_heights = PlayerHeights()

        # Compute and store the background rectangle coordinates based on the frame size and layout configuration.
        self._set_background_rect(reference_frame)

        # Define the court rectangle inside the background rectangle.
        self._set_court_rect()

        # Precompute mini-court keypoints and line definitions.
        self._build_court_keypoints()
        self._set_court_lines()

    # -------------------------------------------------------------------------
    # Layout helpers
    # -------------------------------------------------------------------------

    def _set_background_rect(self, frame: np.ndarray) -> None:
        """
        Determine the position of the mini-court background rectangle within
        the original frame.

        The rectangle is placed near the top-right corner with a fixed margin.
        """
        h, w = frame.shape[:2]

        # Right edge (x2) is offset from frame's right side by outer_margin_px.
        self.bg_x2 = w - self.layout.outer_margin_px

        # Top edge (y1) is set so that the rectangle is vertically placed near
        # the top with outer_margin_px margin.
        self.bg_y2 = self.layout.outer_margin_px + self.layout.rectangle_height_px
        self.bg_x1 = self.bg_x2 - self.layout.rectangle_width_px
        self.bg_y1 = self.bg_y2 - self.layout.rectangle_height_px

    def _set_court_rect(self) -> None:
        """
        Determine the inner court rectangle inside the background rectangle.

        The court is inset from the background rectangle by court_padding_px on all sides.
        """
        pad = self.layout.court_padding_px

        # Court rectangle boundaries inside the background rectangle.
        self.court_x1 = self.bg_x1 + pad
        self.court_y1 = self.bg_y1 + pad
        self.court_x2 = self.bg_x2 - pad
        self.court_y2 = self.bg_y2 - pad

        # Width is used frequently for converting meters to pixels.
        self.court_width_px = self.court_x2 - self.court_x1

    def _meters_to_court_pixels(self, meters: float) -> float:
        """
        Convert a distance in meters to a distance in mini-court pixels along
        the horizontal axis.

        The conversion is based on the ratio between:
        - The real-world double-court width (DOUBLE_LINE_WIDTH_M).
        - The pixel width of the mini court (court_width_px).
        """
        return convert_meters_to_pixels(
            meters,
            self.court_dims.DOUBLE_LINE_WIDTH_M,
            self.court_width_px,
        )

    def _build_court_keypoints(self) -> None:
        """
        Build a flat list [x0, y0, x1, y1, ..., x13, y13] of 14 court keypoints in the mini-court coordinate system.

        These keypoints correspond to:
        - 0, 1, 2, 3: four baseline corners.
        - 4–7: doubles alleys corners (near and far).
        - 8–11: service box corners (no-man's land and singles width).
        - 12–13: center service line intersections.

        The exact mapping between indices and semantic locations is encoded by how they are used to draw lines (see _set_court_lines).
        """
        # We define 14 keypoints, each with (x, y) coordinates.
        kps = [0.0] * (14 * 2)

        # 0: near-left baseline corner
        kps[0] = float(self.court_x1)
        kps[1] = float(self.court_y1)

        # 1: near-right baseline corner
        kps[2] = float(self.court_x2)
        kps[3] = float(self.court_y1)

        # 2: far-left baseline corner
        kps[4] = float(self.court_x1)
        kps[5] = self.court_y1 + self._meters_to_court_pixels(
            2 * self.court_dims.HALF_COURT_LINE_HEIGHT_M
        )

        # 3: far-right baseline corner
        kps[6] = kps[0] + self.court_width_px
        kps[7] = kps[5]

        # 4–7: doubles alleys
        # alley_px: horizontal distance from the doubles sideline to the singles sideline.
        alley_px = self._meters_to_court_pixels(self.court_dims.DOUBLE_ALLY_DIFFERENCE_M)

        # 4: near-left inner (singles) sideline
        kps[8] = kps[0] + alley_px
        kps[9] = kps[1]

        # 5: far-left inner (singles) sideline
        kps[10] = kps[4] + alley_px
        kps[11] = kps[5]

        # 6: near-right inner (singles) sideline
        kps[12] = kps[2] - alley_px
        kps[13] = kps[3]

        # 7: far-right inner (singles) sideline
        kps[14] = kps[6] - alley_px
        kps[15] = kps[7]

        # 8–11: service boxes (no-man's land and singles width)
        no_mans_px = self._meters_to_court_pixels(self.court_dims.NO_MANS_LAND_HEIGHT_M)
        single_width_px = self._meters_to_court_pixels(self.court_dims.SINGLE_LINE_WIDTH_M)

        # 8: near-left service line intersection (left singles area, net side)
        kps[16] = kps[8]
        kps[17] = kps[9] + no_mans_px

        # 9: near-right service line intersection (right singles area, net side)
        kps[18] = kps[16] + single_width_px
        kps[19] = kps[17]

        # 10: far-left service line intersection (left singles area, far side)
        kps[20] = kps[10]
        kps[21] = kps[11] - no_mans_px

        # 11: far-right service line intersection (right singles area, far side)
        kps[22] = kps[20] + single_width_px
        kps[23] = kps[21]

        # 12–13: center service line intersections (net-side and far-side)
        kps[24] = 0.5 * (kps[16] + kps[18])
        kps[25] = kps[17]
        kps[26] = 0.5 * (kps[20] + kps[22])
        kps[27] = kps[21]

        self.court_keypoints_flat = kps

    def _set_court_lines(self) -> None:
        """
        Define the set of line segments that will be drawn on the mini court.

        Each segment is defined as a pair of indices into the keypoints array, where each index corresponds to (x, y) in court_keypoints_flat.
        """
        # Pairs of keypoint indices that define the main court lines.
        self.court_lines = [
            (0, 2),   # near baseline
            (4, 5),   # far-left baseline segment
            (6, 7),   # far-right baseline segment
            (1, 3),   # near-right baseline segment
            (0, 1),   # near baseline full width
            (8, 9),   # singles line near side
            (10, 11), # singles line far side
            (2, 3),   # near-right baseline full width
        ]

    # -------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------

    def draw_background_rect(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw a semi-transparent white rectangle that serves as the background for the mini court.

        The rectangle is blended over the original frame to avoid hiding content completely.
        """
        # Create an all-black mask with the same shape as the frame.
        mask = np.zeros_like(frame, np.uint8)

        # Draw a solid white rectangle on the mask where the background should be.
        cv2.rectangle(
            mask,
            (self.bg_x1, self.bg_y1),
            (self.bg_x2, self.bg_y2),
            (255, 255, 255),
            thickness=cv2.FILLED,
        )

        out = frame.copy()
        alpha = 0.5

        # Blend the rectangle into the frame.
        blended = cv2.addWeighted(frame, alpha, mask, 1 - alpha, 0)

        # Apply blended region only where mask is non-zero.
        out[mask.astype(bool)] = blended[mask.astype(bool)]
        return out

    def draw_court(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the court keypoints, court lines, and the net on top of a frame.

        This method assumes that the background rectangle is already drawn, but it does not strictly require it. 
        The mini court lines are drawn in black, and the net is drawn in blue.
        """
        # Draw all keypoints as small circles for debugging and clarity.
        for i in range(0, len(self.court_keypoints_flat), 2):
            x = int(self.court_keypoints_flat[i])
            y = int(self.court_keypoints_flat[i + 1])
            cv2.circle(frame, (x, y), 4, (0, 0, 0), thickness=-1)

        # Draw the predefined court lines using the keypoint indices.
        for start_idx, end_idx in self.court_lines:
            x1 = int(self.court_keypoints_flat[start_idx * 2])
            y1 = int(self.court_keypoints_flat[start_idx * 2 + 1])
            x2 = int(self.court_keypoints_flat[end_idx * 2])
            y2 = int(self.court_keypoints_flat[end_idx * 2 + 1])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)

        # Draw the net as a horizontal line at the mid-height between the near and far baselines.
        net_y = int(0.5 * (self.court_keypoints_flat[1] + self.court_keypoints_flat[5]))
        cv2.line(
            frame,
            (int(self.court_keypoints_flat[0]), net_y),
            (int(self.court_keypoints_flat[2]), net_y),
            (255, 0, 0),
            thickness=2,
        )

        return frame

    def draw_mini_court(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Draw the mini court on each frame in a sequence.

        The sequence is processed frame by frame:
        1. Background rectangle is drawn.
        2. Court lines, keypoints, and net are drawn on top.

        Parameters
        ----------
        frames : List[np.ndarray]
            List of frames to be processed.

        Returns
        -------
        List[np.ndarray]
            List of frames with the mini court drawn.
        """
        output: List[np.ndarray] = []
        for frame in frames:
            f = self.draw_background_rect(frame)
            f = self.draw_court(f)
            output.append(f)
        return output

    # -------------------------------------------------------------------------
    # Mini-court coordinate conversion
    # -------------------------------------------------------------------------

    def mini_court_position_from_world(
        self,
        world_point: MiniCourtPoint,
        reference_keypoint: MiniCourtPoint,
        reference_kp_index: int,
        player_height_px: float,
        player_height_m: float,
    ) -> MiniCourtPoint:
        """
        Convert a point from world coordinates (original frame pixels) to
        mini-court coordinates.

        The conversion uses a local metric scale estimated from the player's
        bounding-box height in pixels and their assumed real-world height in
        meters. The point is considered relative to a specific court keypoint.
        """
        # Compute pixel offsets in original frame coordinate space.
        dx_px, dy_px = axis_distances(world_point, reference_keypoint)

        # Convert pixel offsets to metric units (meters) using player's height as a scaling reference.
        dx_m = convert_pixels_to_meters(dx_px, player_height_m, player_height_px)
        dy_m = convert_pixels_to_meters(dy_px, player_height_m, player_height_px)

        # Convert metric offsets to mini-court pixels.
        dx_mini_px = self._meters_to_court_pixels(dx_m)
        dy_mini_px = self._meters_to_court_pixels(dy_m)

        # Anchor the position to the corresponding mini-court keypoint.
        ref_mini_x = self.court_keypoints_flat[reference_kp_index * 2]
        ref_mini_y = self.court_keypoints_flat[reference_kp_index * 2 + 1]

        return ref_mini_x + dx_mini_px, ref_mini_y + dy_mini_px

    def _player_height_over_window(
        self,
        player_tracks: List[BBoxDict],
        frame_idx: int,
        track_id: int,
        window_before: int = 20,
        window_after: int = 50,
    ) -> float:
        """
        Compute a robust estimate of a player's height in pixels over a temporal
        window around a target frame index.

        This method looks at a range of frames around `frame_idx` and collects
        bounding-box heights for the given player. The maximum height observed
        is used, under the assumption that occasional scale changes due to
        perspective or detection noise should not reduce the estimated height.
        """
        start = max(0, frame_idx - window_before)
        end = min(len(player_tracks), frame_idx + window_after)

        heights: List[float] = []

        for i in range(start, end):
            bbox = player_tracks[i].get(track_id)
            if bbox is not None:
                heights.append(bbox_height(bbox))

        # Use a fallback minimum value of 1.0 to avoid zero-scale issues.
        return max(heights) if heights else 1.0

    def convert_to_mini_court_tracks(
        self,
        player_tracks: List[BBoxDict],
        ball_tracks: List[Dict[int, Tuple[float, float, float, float]]],
        original_court_keypoints_flat: List[float],
    ) -> Tuple[List[Dict[int, MiniCourtPoint]], List[Dict[int, MiniCourtPoint]]]:
        """
        Convert original-frame player and ball bounding boxes into mini-court
        coordinates for all frames.

        High-level algorithm:
        1. For each frame:
           - Obtain ball position and player bounding boxes.
           - Determine the player closest to the ball.
        2. For each player in that frame:
           - Estimate the player foot point in the original frame.
           - Find the closest court keypoint on the original-court model.
           - Estimate the metric scale from player bounding-box height.
           - Convert the player's foot point to mini-court coordinates using
             mini_court_position_from_world.
        3. Convert the ball center in a similar way, using the closest player’s
           height for scale and the closest court keypoint.
        """
        player_mini_tracks: List[Dict[int, MiniCourtPoint]] = []
        ball_mini_tracks: List[Dict[int, MiniCourtPoint]] = []

        # Mapping of player IDs to their real-world heights in meters.
        # This can be configured in PlayerHeights.
        player_id_to_height_m = {
            1: self.player_heights.PLAYER_1_HEIGHT_M,
            2: self.player_heights.PLAYER_2_HEIGHT_M,
        }

        # Subset of keypoint indices that lie on the baselines. These are often used as reference points for player positions.
        baseline_indices = [0, 2, 12, 13]

        for frame_idx, player_boxes in enumerate(player_tracks):
            # Retrieve the ball bounding box for this frame (assuming ID 1).
            ball_box = ball_tracks[frame_idx].get(1)

            # If either players or the ball are missing, skip this frame.
            if not player_boxes or ball_box is None:
                player_mini_tracks.append({})
                ball_mini_tracks.append({})
                continue

            # Compute the ball center in original frame coordinates.
            ball_center = bbox_center(ball_box)

            # Determine which player is closest to the ball in the original frame. This player will be used as the reference for the ball's
            # scale estimation.
            closest_player_id = min(
                player_boxes.keys(),
                key=lambda pid: euclidean_distance(ball_center, bbox_center(player_boxes[pid])),
            )

            frame_player_output: Dict[int, MiniCourtPoint] = {}

            # Process each player in the frame.
            for player_id, box in player_boxes.items():
                # Approximate player position as the bottom center of the bounding box, corresponding to the player's feet.
                foot = bbox_foot_point(box)

                # Find the closest original-court keypoint to the player's foot. This keypoint serves as the reference for coordinate conversion.
                kp_idx = closest_keypoint_index(foot, original_court_keypoints_flat, baseline_indices)
                ref_kp = (
                    original_court_keypoints_flat[kp_idx * 2],
                    original_court_keypoints_flat[kp_idx * 2 + 1],
                )

                # Estimate player's height in pixels over a temporal window.
                height_px = self._player_height_over_window(player_tracks, frame_idx, player_id)

                # Use configured real-world height for this player; fall back to player 1 height if unknown.
                height_m = player_id_to_height_m.get(player_id, self.player_heights.PLAYER_1_HEIGHT_M)

                # Convert player's foot position from world coordinates to mini-court coordinates.
                mini_pos = self.mini_court_position_from_world(
                    foot,
                    ref_kp,
                    kp_idx,
                    player_height_px=height_px,
                    player_height_m=height_m,
                )
                frame_player_output[player_id] = mini_pos

            # Ball is associated with the player closest to it, for scale estimation.
            frame_ball_output: Dict[int, MiniCourtPoint] = {}
            if closest_player_id in player_boxes:
                kp_idx = closest_keypoint_index(ball_center, original_court_keypoints_flat, baseline_indices)
                ref_kp = (
                    original_court_keypoints_flat[kp_idx * 2],
                    original_court_keypoints_flat[kp_idx * 2 + 1],
                )

                height_px = self._player_height_over_window(player_tracks, frame_idx, closest_player_id)
                height_m = player_id_to_height_m.get(
                    closest_player_id,
                    self.player_heights.PLAYER_1_HEIGHT_M,
                )

                mini_ball_pos = self.mini_court_position_from_world(
                    ball_center,
                    ref_kp,
                    kp_idx,
                    player_height_px=height_px,
                    player_height_m=height_m,
                )
                frame_ball_output[1] = mini_ball_pos

            player_mini_tracks.append(frame_player_output)
            ball_mini_tracks.append(frame_ball_output)

        return player_mini_tracks, ball_mini_tracks

    # -------------------------------------------------------------------------
    # Convenience for drawing points (players/ball) on mini court
    # -------------------------------------------------------------------------

    @staticmethod
    def draw_points(
        frames: List[np.ndarray],
        positions_per_frame: List[Dict[int, MiniCourtPoint]],
        color: Tuple[int, int, int] = (255, 0, 0),
    ) -> List[np.ndarray]:
        """
        Draw circular markers for a set of positions on top of each frame.

        This is typically used to render player or ball positions on the mini
        court after coordinate conversion.
        """
        for frame, positions in zip(frames, positions_per_frame):
            for _, (x, y) in positions.items():
                cv2.circle(frame, (int(x), int(y)), 5, color, thickness=-1)
        return frames

    # -------------------------------------------------------------------------
    # Simple accessors
    # -------------------------------------------------------------------------

    def mini_court_origin(self) -> Tuple[int, int]:
        """
        Get the top-left corner of the mini-court rectangle in frame coordinates.
        """
        return int(self.court_x1), int(self.court_y1)

    def mini_court_width(self) -> int:
        """
        Get the width of the mini-court rectangle in pixels.
        """
        return int(self.court_width_px)

    def mini_court_keypoints_flat(self) -> List[float]:
        """
        Return a copy of the mini-court keypoints as a flat list.
        """
        return list(self.court_keypoints_flat)
