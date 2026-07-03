from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk

from . import data as util_gau
from .renderer import _sort_gaussian
from .vtk_native_renderer import VTKNativeGaussianRenderer

# Number of randomly-sampled Gaussian centres used for fast hover picking.
# Large enough for good ray coverage; small enough for sub-millisecond O(K) query.
_PICK_SAMPLE_SIZE = 8_000

# Cone half-angle (in pixels-equivalent) around the view ray within which a splat
# centre counts as "under the cursor". Matches the legacy tolerance so hover and
# click agree. The cone widens with depth (it is an angular, footprint-proxy test).
_PICK_CONE_TOL_PX = 5.0

# Front-surface depth band: when averaging the surface depth along a ray, only
# include cone hits within this relative fraction of the nearest hit's depth, so
# the front surface is never averaged together with geometry seen through gaps
# behind it.
_PICK_DEPTH_BAND = 0.05


def _rotation_matrix_to_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
    trace = float(np.trace(rotation_matrix))

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
    elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
        scale = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
        w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
        scale = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
        w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / scale
        x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
        w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / scale
        x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / scale
        y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def _multiply_quaternions_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(1, 4)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 4)

    w1, x1, y1, z1 = left[0]
    w2 = right[:, 0]
    x2 = right[:, 1]
    y2 = right[:, 2]
    z2 = right[:, 3]

    return np.column_stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _coerce_crop_bounds(bounds) -> np.ndarray:
    """Convert a PyVista crop widget payload into xmin/xmax/ymin/ymax/zmin/zmax."""
    if bounds is None:
        raise ValueError("Crop bounds cannot be None")

    if hasattr(bounds, "GetBounds"):
        bounds = bounds.GetBounds()
    elif hasattr(bounds, "bounds"):
        bounds = bounds.bounds

    if hasattr(bounds, "points"):
        points = np.asarray(bounds.points, dtype=np.float64)
        if points.ndim == 2 and points.shape[1] == 3:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            return np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]], dtype=np.float64)

    bounds_array = np.asarray(bounds, dtype=np.float64)

    if bounds_array.size == 6:
        return bounds_array.reshape(6)

    if bounds_array.ndim == 2 and bounds_array.shape == (3, 2):
        return np.array([
            bounds_array[0, 0], bounds_array[0, 1],
            bounds_array[1, 0], bounds_array[1, 1],
            bounds_array[2, 0], bounds_array[2, 1],
        ], dtype=np.float64)

    if bounds_array.ndim == 2 and bounds_array.shape[1] == 3:
        mins = bounds_array.min(axis=0)
        maxs = bounds_array.max(axis=0)
        return np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]], dtype=np.float64)

    raise ValueError("Crop bounds must resolve to 6 values")


class GaussianActor:
    """Single-pass VTK-native actor for rendering 3D Gaussian splats.

    A true vtkActor with full scene participation: bounds, picking, depth
    testing, and compositing with other VTK geometry.
    """

    # Opacity boost applied to every splat while an absolute set_colors() data
    # view is active (feature similarity / PCA-RGB). The splat scene is built from
    # many low-opacity Gaussians, so a flat recolour at the natural opacity averages
    # into a washed-out haze; lifting opacity makes the data colour read crisp and
    # opaque (the same trick the label channel uses for painted labels).
    DATA_VIEW_BOOST = 0.95

    def __init__(self, gaussian_data: util_gau.GaussianData):
        self._renderer: VTKNativeGaussianRenderer | None = None
        self._sync_needed = True
        self._last_mtime = 0

        self._last_view_matrix = None
        self._sort_tolerance = 1e-4

        self._scale_modifier = 1.0
        self._render_mode = 7
        self.auto_sort = True
        self._crop_bounds: np.ndarray | None = None

        self._mesh = pv.PolyData(gaussian_data.xyz)
        self._mesh.point_data['rot'] = gaussian_data.rot
        self._mesh.point_data['scale'] = gaussian_data.scale
        self._mesh.point_data['opacity'] = gaussian_data.opacity
        self._mesh.point_data['sh'] = gaussian_data.sh

        self._original_mesh = self._mesh.copy()

        # Pristine, never-tinted copy of the SH coefficients for reset_colors()
        self._pristine_sh = np.asarray(gaussian_data.sh, dtype=np.float32).copy()

        # Label channel: the renderer owns the per-splat class-id buffer and the
        # label LUT. These mirror the boost + palette so they can be (re)applied
        # when the renderer is (re)bound (calls may arrive before bind_to_plotter).
        self._label_boost = 0.0
        self._pending_label_lut: np.ndarray | None = None

        # Data-view opacity boost. Enabled automatically by set_colors() (the
        # absolute feature-colour override) so the flat colours render opaque, and
        # cleared by a full reset_colors(). Buffered so it can be (re)applied when
        # the renderer is (re)bound.
        self._flat_boost = 0.0

        # Display channel: per-splat LUT index + 256-entry LUT + blend/boost.
        # The interactive similarity/class view — an N-int upload per recolour
        # instead of an SH rewrite; the SH shading stays visible underneath.
        # Buffered so state set before bind_to_plotter is applied on bind.
        self._disp_values: np.ndarray | None = None
        self._disp_lut: np.ndarray | None = None
        self._disp_mix = 0.0
        self._disp_boost = 0.0

        # Will be created when bind_to_plotter is called
        self._plotter: pv.Plotter | None = None
        self.actor: vtk.vtkActor | None = None
        # Random subsample of Gaussian positions (and matching opacities) for fast
        # hover picking. Opacity weights the surface-depth estimate so near-
        # transparent floaters don't pull the picked point off the visible surface.
        self._pick_sample_xyz: np.ndarray | None = None
        self._pick_sample_opacity: np.ndarray | None = None

    def cleanup(self):
        """Release GPU and renderer resources."""
        if self._renderer:
            self._renderer.cleanup()
            self._renderer = None

    def set_crop_bounds(self, bounds: np.ndarray):
        """Store crop bounds for use by apply_crop_box()."""
        self._crop_bounds = _coerce_crop_bounds(bounds)

    def clear_crop_box(self):
        """Clear stored crop bounds."""
        self._crop_bounds = None

    def transform(self, matrix: np.ndarray):
        """
        Apply a 4x4 homogeneous transform to the splat positions, rotations, and scales.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("GaussianActor.transform expects a 4x4 matrix")

        self._mesh.transform(matrix, inplace=True)
        self._original_mesh.transform(matrix, inplace=True)

        applied_scales = np.linalg.norm(matrix[:3, :3], axis=1)
        safe_scales = np.where(applied_scales == 0.0, 1.0, applied_scales)
        rotation_matrix = matrix[:3, :3] / safe_scales[:, None]
        applied_quaternion = _rotation_matrix_to_wxyz(rotation_matrix)

        current_rots = np.asarray(self._mesh.point_data['rot'], dtype=np.float64)
        new_rots = _multiply_quaternions_wxyz(applied_quaternion, current_rots)
        new_rots_norm = np.linalg.norm(new_rots, axis=1, keepdims=True)
        new_rots_norm = np.where(new_rots_norm == 0.0, 1.0, new_rots_norm)
        new_rots = (new_rots / new_rots_norm).astype(np.float32)

        current_scales = np.asarray(self._mesh.point_data['scale'], dtype=np.float64)
        new_scales = (current_scales * applied_scales).astype(np.float32)

        self._mesh.point_data['rot'] = new_rots
        self._original_mesh.point_data['rot'] = new_rots.copy()
        self._mesh.point_data['scale'] = new_scales
        self._original_mesh.point_data['scale'] = new_scales.copy()
        self._last_view_matrix = None
        self._sync_to_renderer()

    def remove_floaters(self, min_opacity: float = 0.05, max_scale: float = 1.0):
        """
        Cull noisy splats that are too transparent or too large.
        """
        if self.point_count == 0:
            return

        opacities = np.asarray(self._mesh.point_data['opacity']).ravel()
        scales = np.asarray(self._mesh.point_data['scale'])
        max_scales_per_splat = np.max(scales, axis=1)

        valid_mask = (opacities >= min_opacity) & (max_scales_per_splat <= max_scale)
        if not np.any(valid_mask):
            print("Cull aborted: Parameters are too aggressive and would delete the entire model.")
            return

        points_removed = valid_mask.size - int(np.sum(valid_mask))
        if points_removed == 0:
            return

        culled_mesh = self._mesh.extract_points(valid_mask)
        self.mesh = culled_mesh
        self._original_mesh = culled_mesh.copy()
        self._last_view_matrix = None
        self._sync_to_renderer()

        print(f"Culled {points_removed:,} floaters. Remaining splats: {self.point_count:,}")

    def reset_colors(self, element_ids=None):
        """
        Restore the original (pristine) SH colours, discarding any tints.

        If ``element_ids`` is provided only those splats are reset; otherwise
        all splats are restored.  No-op if the pristine snapshot no longer
        matches the current splat count (e.g. after a crop/floater cull).
        """
        if self.point_count == 0 or self._pristine_sh is None:
            return
        if self._pristine_sh.shape[0] != self.point_count:
            return

        # Mutate the resident float32 SH view in place (no whole-array copy, no
        # float64 round-trip). The renderer reads colours from its own SSBO, so
        # only the touched rows need pushing. _original_mesh keeps its own SH
        # (only its bounds are ever read), so we never write back to it.
        sh = self._mesh.point_data['sh']
        if element_ids is not None:
            sel = np.asarray(element_ids)
            sh[sel] = self._pristine_sh[sel]
        else:
            sh[:] = self._pristine_sh
            sel = None
            # A full restore leaves the data view: drop the opacity boost so the
            # pristine scene renders with its natural (translucent) compositing.
            self.set_flat_boost(0.0)

        self._push_colors_to_renderer(sel)

    def tint_gaussians(self, indices: np.ndarray, color_rgb: tuple[int, int, int], blend_factor: float = 0.6):
        """
        Tint selected splats by modifying the DC spherical harmonic coefficients.
        """
        if self.point_count == 0:
            return

        selection = np.asarray(indices)
        if selection.size == 0:
            return

        if selection.dtype == bool:
            selection = selection.ravel()
            if selection.size != self.point_count:
                raise ValueError("Boolean mask length must match the number of splats")
            selection = np.flatnonzero(selection)
        else:
            selection = selection.astype(np.intp, copy=False).ravel()

        if selection.size == 0:
            return

        blend_factor = float(np.clip(blend_factor, 0.0, 1.0))
        target_rgb = np.clip(np.asarray(color_rgb, dtype=np.float64) / 255.0, 0.0, 1.0)

        sh_c0 = 0.28209479177387814
        target_sh_dc = (target_rgb - 0.5) / sh_c0

        # Blend the tint into the DC band of only the selected splats, in place
        # on the resident float32 SH view. The per-row math runs in float64 on
        # the O(painted) slice for precision, then stores back as float32 — no
        # whole-array copy or cast, and no write-back to _original_mesh.
        sh = self._mesh.point_data['sh']
        current_dc = sh[selection, 0:3].astype(np.float64)
        new_dc = current_dc * (1.0 - blend_factor) + target_sh_dc * blend_factor
        sh[selection, 0:3] = new_dc.astype(np.float32)
        self._push_colors_to_renderer(selection)

    def apply_label_tint(self, indices: np.ndarray, color_rgb: tuple[int, int, int],
                         blend_factor: float = 0.6):
        """Reset the given splats to their pristine colour and tint them in a
        single colour-only GPU update.

        Equivalent to ``reset_colors(indices)`` followed by
        ``tint_gaussians(indices, ...)`` but resolves to one partial upload, so
        interactive label painting touches each splat's SH exactly once per
        stroke tick. Resetting first means relabelling never accumulates tint.
        """
        if self.point_count == 0:
            return

        selection = np.asarray(indices)
        if selection.size == 0:
            return
        if selection.dtype == bool:
            selection = selection.ravel()
            if selection.size != self.point_count:
                raise ValueError("Boolean mask length must match the number of splats")
            selection = np.flatnonzero(selection)
        else:
            selection = selection.astype(np.intp, copy=False).ravel()
        if selection.size == 0:
            return

        # Operate in place on the resident float32 SH view; only the selected
        # rows are touched (O(painted)), so there is no whole-array float64 copy,
        # no float32 re-cast of all N splats, and no write-back to _original_mesh.
        sh = self._mesh.point_data['sh']

        # Reset the selected splats to pristine before tinting (no muddy blend on
        # relabel). Guard against a stale pristine snapshot after a crop/cull.
        if (self._pristine_sh is not None
                and self._pristine_sh.shape[0] == self.point_count):
            sh[selection] = self._pristine_sh[selection]

        blend_factor = float(np.clip(blend_factor, 0.0, 1.0))
        target_rgb = np.clip(np.asarray(color_rgb, dtype=np.float64) / 255.0, 0.0, 1.0)
        sh_c0 = 0.28209479177387814
        target_sh_dc = (target_rgb - 0.5) / sh_c0

        base_dc = sh[selection, 0:3].astype(np.float64)
        sh[selection, 0:3] = (base_dc * (1.0 - blend_factor)
                              + target_sh_dc * blend_factor).astype(np.float32)
        self._push_colors_to_renderer(selection)

    def set_colors(self, colors_rgb: np.ndarray):
        """Set every splat's display colour to an absolute ``[N, 3]`` RGB array,
        overriding the SH coefficients in a single partial GPU update (no blend,
        no re-sort).

        Used by the feature-similarity / feature-RGB views, where the per-splat
        colour fully replaces the splat appearance. Both the DC band AND the
        higher-order (view-dependent) bands are written: the DC term is set to
        reproduce the requested colour exactly, and the higher-order bands are
        ZEROED. Leaving them intact would let the geometry shader add the splat's
        original view-dependent residual on top of the absolute colour (the
        renderer evaluates the full SH at render_mod >= 1), pushing bright colours
        past 1.0 so they clamp to white — the "white billboard" artefact. Zeroing
        them makes the splat render the flat absolute colour from every angle.
        ``reset_colors()`` restores the pristine SH (DC + all bands), so this is
        fully reversible. ``colors_rgb`` may be given in 0..255 or 0..1 range.
        """
        if self.point_count == 0:
            return
        colors = np.asarray(colors_rgb, dtype=np.float64)
        if (colors.ndim != 2 or colors.shape[1] != 3
                or colors.shape[0] != self.point_count):
            return
        # Normalise 0..255 inputs to 0..1.
        if colors.size and float(colors.max()) > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)

        sh_c0 = 0.28209479177387814
        target_sh_dc = (colors - 0.5) / sh_c0

        # Overwrite the whole SH in place on the resident float32 view: DC band to
        # the requested colour, higher-order bands to zero (so the colour is flat
        # and view-independent). This is an all-N recolour by nature (data view),
        # but we still avoid a float64 copy of the array and the _original_mesh
        # write-back. _pristine_sh keeps the untouched bands for reset_colors().
        sh = self._mesh.point_data['sh']
        sh[:, 0:3] = target_sh_dc.astype(np.float32)
        if sh.shape[1] > 3:
            sh[:, 3:] = 0.0
        self._push_colors_to_renderer(None)

        # Lift every splat toward opaque so the flat absolute colour reads crisp
        # over the many low-opacity splats (otherwise it averages into a haze).
        self.set_flat_boost(self.DATA_VIEW_BOOST)

    # ----------------------------------------------------------------------
    # Label channel — per-splat class ids rendered boosted-opacity so painted
    # labels read through the surrounding translucency (vs a diluted SH tint).
    # ----------------------------------------------------------------------
    def set_label_boost(self, boost: float) -> None:
        """Strength of the label overlay (0 = off, ~0.9 = strong)."""
        self._label_boost = float(max(0.0, min(1.0, boost)))
        if self._renderer:
            self._renderer.set_label_boost(self._label_boost)

    def get_label_boost(self) -> float:
        return self._label_boost

    def set_flat_boost(self, boost: float) -> None:
        """Data-view opacity boost (0 = off). >0 lifts every splat toward opaque so
        an absolute set_colors() recolour reads as a solid colour. Buffered and
        (re)applied when the renderer is (re)bound."""
        self._flat_boost = float(max(0.0, min(1.0, boost)))
        if self._renderer:
            self._renderer.set_flat_boost(self._flat_boost)

    def get_flat_boost(self) -> float:
        return self._flat_boost

    # ----------------------------------------------------------------------
    # Display channel — per-splat LUT indices coloured through a 256-entry LUT
    # in the geometry shader. The fast path for interactive similarity /
    # multi-class recolours: never touches the SH, so it's fully reversible by
    # just setting the mix back to 0.
    # ----------------------------------------------------------------------
    def set_display_values(self, values: np.ndarray) -> bool:
        """Set the per-splat display LUT indices ([N], uint8 semantics)."""
        arr = np.asarray(values).ravel()
        if arr.shape[0] != self.point_count:
            return False
        self._disp_values = arr.astype(np.int32, copy=True)
        if self._renderer:
            return self._renderer.set_display_values(self._disp_values)
        return True

    def set_display_lut(self, palette_rgb: np.ndarray) -> None:
        """Set the display LUT (row i colours display value i; [K<=256, 3|4])."""
        self._disp_lut = np.asarray(palette_rgb).copy()
        if self._renderer:
            self._renderer.set_display_lut(self._disp_lut)

    def set_display_mix(self, mix: float) -> None:
        """Blend of the display colour over the SH colour (0 = channel off)."""
        self._disp_mix = float(max(0.0, min(1.0, mix)))
        if self._renderer:
            self._renderer.set_display_mix(self._disp_mix)

    def get_display_mix(self) -> float:
        return self._disp_mix

    def set_display_boost(self, boost: float) -> None:
        """Opacity lift while the display channel is active (0 = natural)."""
        self._disp_boost = float(max(0.0, min(1.0, boost)))
        if self._renderer:
            self._renderer.set_display_boost(self._disp_boost)

    def update_label_lut(self, palette_rgb: np.ndarray) -> None:
        """Set the full label palette (row i == colour for class id i)."""
        self._pending_label_lut = np.asarray(palette_rgb).copy()
        if self._renderer:
            self._renderer.update_label_lut(self._pending_label_lut)

    def set_label_color(self, class_id: int, color_rgb) -> None:
        """Pin one label colour (the exact colour a stroke painted with)."""
        if self._renderer:
            self._renderer.set_lut_entry(int(class_id), color_rgb)

    def set_label_ids(self, element_ids: np.ndarray, class_id: int) -> bool:
        """Assign ``class_id`` to a subset of splats (a painted stroke). O(painted)."""
        if self._renderer:
            return self._renderer.update_class_ids(element_ids, int(class_id))
        return False

    def set_label_ids_full(self, class_ids: np.ndarray) -> bool:
        """Replace the entire per-splat class-id array (full flush / restore)."""
        if self._renderer:
            return self._renderer.set_class_ids_full(class_ids)
        return False

    def _push_colors_to_renderer(self, indices=None):
        """Push only the SH (colour) of the given splats to the renderer via an
        in-place partial GPU update — no geometry rebuild, no depth re-sort, and
        no pick-sample resample (geometry is unchanged).

        Falls back to a full ``_sync_to_renderer`` when the partial path is
        unavailable (renderer not bound yet, or the splat count changed).
        """
        if not self._renderer or self._mesh.n_points == 0:
            return
        try:
            sh = np.asarray(self._mesh.point_data['sh'], dtype=np.float32)
            if indices is None:
                idx = np.arange(self._mesh.n_points, dtype=np.intp)
            else:
                idx = np.asarray(indices)
                if idx.dtype == bool:
                    idx = np.flatnonzero(idx.ravel())
                else:
                    idx = idx.astype(np.intp, copy=False).ravel()
                if idx.size == 0:
                    return
            ok = self._renderer.update_sh(idx, sh[idx])
        except Exception:
            ok = False
        if not ok:
            self._sync_to_renderer()

    def apply_crop_box(self, bounds: np.ndarray | None = None):
        """
        Commit the current crop preview by deleting points outside the crop bounds.
        """
        if bounds is not None:
            self.set_crop_bounds(bounds)

        if self._crop_bounds is None or self.point_count == 0:
            return

        crop_bounds = np.asarray(self._crop_bounds, dtype=np.float64)
        points = np.asarray(self._mesh.points, dtype=np.float64)
        valid_mask = (
            (points[:, 0] >= crop_bounds[0]) & (points[:, 0] <= crop_bounds[1]) &
            (points[:, 1] >= crop_bounds[2]) & (points[:, 1] <= crop_bounds[3]) &
            (points[:, 2] >= crop_bounds[4]) & (points[:, 2] <= crop_bounds[5])
        )

        if not np.any(valid_mask):
            print("Crop aborted: The current crop bounds would delete the entire model.")
            return

        points_removed = valid_mask.size - int(np.sum(valid_mask))
        if points_removed == 0:
            return

        culled_mesh = self._mesh.extract_points(valid_mask)
        self.mesh = culled_mesh
        self._original_mesh = culled_mesh.copy()
        self._last_view_matrix = None
        self._sync_to_renderer()

        print(f"Applied crop: removed {points_removed:,} splats. Remaining splats: {self.point_count:,}")

    @property
    def mesh(self) -> pv.PolyData:
        return self._mesh

    @mesh.setter
    def mesh(self, new_mesh: pv.PolyData):
        self._mesh = new_mesh
        self._sync_needed = True

    @property
    def point_count(self) -> int:
        return self._mesh.n_points if self._mesh else 0

    @property
    def position(self):
        return self.actor.GetPosition()

    @position.setter
    def position(self, pos: tuple[float, float, float]):
        self.actor.SetPosition(*pos)
        self.actor.Modified()

    @property
    def scale(self):
        return self.actor.GetScale()

    @scale.setter
    def scale(self, scale_factor: tuple[float, float, float]):
        self.actor.SetScale(*scale_factor)
        self.actor.Modified()

    @property
    def scale_modifier(self) -> float:
        return self._scale_modifier

    @scale_modifier.setter
    def scale_modifier(self, value: float):
        self._scale_modifier = float(value)
        if self._renderer:
            self._renderer.set_scale_modifier(self._scale_modifier)

    @property
    def render_mode(self) -> int:
        return self._render_mode

    @render_mode.setter
    def render_mode(self, mode: int):
        self._render_mode = int(mode)
        if self._renderer:
            self._renderer.set_render_mod(self._render_mode - 4)

    @property
    def reduce_updates(self) -> bool:
        return False

    @reduce_updates.setter
    def reduce_updates(self, val: bool):
        pass

    def sort_gaussians(self):
        if self._renderer:
            self._renderer.trigger_sort()

    def bind_to_plotter(self, plotter: pv.Plotter):
        """Attach this actor to a PyVista plotter and begin rendering."""
        self._plotter = plotter
        self._renderer = VTKNativeGaussianRenderer(plotter.renderer)
        
        opacity_array = np.array(self._mesh.point_data['opacity'])
        if opacity_array.ndim == 1:
            opacity_array = opacity_array.reshape(-1, 1)
        
        gaussian_data = util_gau.GaussianData(
            xyz=np.array(self._mesh.points),
            rot=np.array(self._mesh.point_data['rot']),
            scale=np.array(self._mesh.point_data['scale']),
            opacity=opacity_array,
            sh=np.array(self._mesh.point_data['sh']),
        )
        self._renderer.load(gaussian_data)
        self._renderer.set_scale_modifier(self._scale_modifier)
        self._renderer.set_render_mod(self._render_mode - 4)
        # (Re)apply any label-channel / data-view state configured before binding.
        self._renderer.set_label_boost(self._label_boost)
        self._renderer.set_flat_boost(self._flat_boost)
        if self._pending_label_lut is not None:
            self._renderer.update_label_lut(self._pending_label_lut)
        # (Re)apply the display channel state.
        self._renderer.set_display_mix(self._disp_mix)
        self._renderer.set_display_boost(self._disp_boost)
        if self._disp_lut is not None:
            self._renderer.set_display_lut(self._disp_lut)
        if self._disp_values is not None:
            self._renderer.set_display_values(self._disp_values)
        self.actor = self._renderer.actor
        self._rebuild_pick_sample()
        
        # Trigger depth sorting when camera moves (respects auto_sort flag)
        plotter.renderer.GetActiveCamera().AddObserver(
            vtk.vtkCommand.ModifiedEvent,
            lambda *_: self._renderer.trigger_sort() if (self._renderer and self.auto_sort) else None,
        )

    def _sync_to_renderer(self):
        if self._mesh.n_points == 0 or not self._renderer:
            return

        opacity_array = np.array(self._mesh.point_data['opacity'])
        if opacity_array.ndim == 1:
            opacity_array = opacity_array.reshape(-1, 1)

        rebuilt_gaussians = util_gau.GaussianData(
            xyz=np.array(self._mesh.points),
            rot=np.array(self._mesh.point_data['rot']),
            scale=np.array(self._mesh.point_data['scale']),
            opacity=opacity_array,
            sh=np.array(self._mesh.point_data['sh']),
        )

        self._renderer.load(rebuilt_gaussians)

        self._last_mtime = self._mesh.GetMTime()
        self._sync_needed = False
        self._last_view_matrix = None
        self._rebuild_pick_sample()

    def _rebuild_pick_sample(self):
        """Cache a random subsample of Gaussian centres (and their opacities) for
        fast O(K) hover picking, K=_PICK_SAMPLE_SIZE."""
        n = self._mesh.n_points
        if n == 0:
            self._pick_sample_xyz = None
            self._pick_sample_opacity = None
            return
        k = min(_PICK_SAMPLE_SIZE, n)
        idx = np.random.choice(n, k, replace=False)
        self._pick_sample_xyz = np.asarray(self._mesh.points[idx], dtype=np.float64)
        try:
            op = np.asarray(self._mesh.point_data['opacity'], dtype=np.float64).ravel()
            self._pick_sample_opacity = op[idx]
        except Exception:
            self._pick_sample_opacity = None

    def pick_gaussian(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                       fovy_rad: float, window_height: int,
                       fast: bool = False,
                       min_world_radius: float | None = None) -> np.ndarray | None:
        """Estimate the world-space surface point under a view ray.

        A Gaussian scene has no explicit surface, so we approximate the visible
        surface depth along the ray and return a point ON the ray at that depth —
        NOT a splat centre. Returning an on-ray point is what makes the hover /
        brush sphere glide continuously: the lateral position is always exactly
        under the cursor, and the depth is a stable opacity-weighted average over
        the front-most cluster of Gaussians the ray passes through (so it does not
        snap when the single nearest splat changes).

        The gather around the ray is the larger of an angular cone (which widens
        with depth) and ``min_world_radius`` — a fixed world-space floor the caller
        sets to the brush sphere radius. Sizing the gather to the brush makes it
        zoom-invariant (so zooming in no longer starves it of centres) AND keeps
        the picked point inside the region the brush will paint, so the painted
        coverage stays full. With no floor it is the bare angular cone.

        ``fast`` uses the cached random subsample of centres for O(K) hover
        picking; the click path scans the full set for accuracy. ``ray_dir`` is
        assumed unit length (the caller normalises it), so the dot product is a
        true world-space depth along the ray.
        """
        if self.point_count == 0:
            return None

        if fast and self._pick_sample_xyz is not None:
            obj_xyz = self._pick_sample_xyz
            opacity = self._pick_sample_opacity
        else:
            obj_xyz = np.asarray(self._mesh.points, dtype=np.float64)
            try:
                opacity = np.asarray(self._mesh.point_data['opacity'], dtype=np.float64).ravel()
            except Exception:
                opacity = None

        try:
            vtk_mat = self.actor.GetMatrix()
            model_mat = np.zeros((4, 4), dtype=np.float64)
            for i in range(4):
                for j in range(4):
                    model_mat[i, j] = vtk_mat.GetElement(i, j)
            xyz_h = np.concatenate([obj_xyz, np.ones((obj_xyz.shape[0], 1), dtype=np.float64)], axis=1)
            world_xyz = (model_mat @ xyz_h.T).T[:, :3]
        except Exception:
            world_xyz = obj_xyz

        # Depth of each centre along the ray + its lateral distance from the ray.
        vecs = world_xyz - ray_origin
        t = vecs @ ray_dir
        front = t > 0.0
        if not np.any(front):
            return None
        t = t[front]
        lateral = np.linalg.norm(vecs[front] - t[:, None] * ray_dir, axis=1)
        if opacity is not None and opacity.shape[0] == front.shape[0]:
            op = np.clip(opacity[front], 0.0, 1.0)
        else:
            op = np.ones_like(t)

        # Gather radius around the ray: the larger of an angular cone (widens with
        # depth) and a world-space floor sized to the brush. The brush-sized floor
        # keeps the gather constant under zoom (no zoomed-in starvation) without
        # exceeding the brush — averaging the depth over a patch BIGGER than the
        # brush biases the pick on oblique surfaces and thins the paint coverage.
        # lateral < reach  ==  inside the gather.
        tol = _PICK_CONE_TOL_PX * (float(fovy_rad) / max(int(window_height), 1))
        floor = float(min_world_radius or 0.0)
        reach = np.maximum(tol * t, floor)
        cone = lateral < reach
        if not np.any(cone):
            return None
        t_c = t[cone]
        lat_c = lateral[cone]
        op_c = op[cone]
        reach_c = reach[cone]

        # Restrict to the front-most surface (nearest hit + a small depth band) so
        # we don't average the front surface together with background seen through
        # gaps, then take an opacity- and proximity-weighted mean depth. The
        # proximity weight reuses the per-centre reach, so the floor widens the
        # weight kernel too (it never zero-weights the centres it pulled in).
        t_near = float(t_c.min())
        front_band = t_c <= t_near * (1.0 + _PICK_DEPTH_BAND)
        prox = np.exp(-(lat_c / np.maximum(reach_c, 1e-12)) ** 2)
        w = op_c * prox * front_band
        wsum = float(w.sum())
        t_pick = float((w * t_c).sum() / wsum) if wsum > 1e-12 else t_near

        return ray_origin + t_pick * ray_dir

        return None
