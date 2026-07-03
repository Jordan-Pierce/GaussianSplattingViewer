"""Single-pass VTK-native 3D Gaussian Splatting renderer.

This replaces the hybrid VTK proxy + ModernGL architecture with a true
single-pass vtkActor that renders Gaussians inside VTK's rendering pipeline.
The key fix: render in the opaque pass (ForceOpaqueOn) to bypass VTK's
Order-Independent Transparency (OIT) accumulation buffer, allowing direct
control over blend state.
"""
from __future__ import annotations

import os
import numpy as np
import vtk
from OpenGL import GL as gl
import glm
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.util.misc import calldata_type
from vtkmodules.vtkCommonCore import VTK_OBJECT

from . import data as util_gau
from .renderer import _sort_gaussian

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_shader(name: str) -> str:
    """Read a shader from the shaders/ directory."""
    with open(os.path.join(MODULE_DIR, 'shaders', name), 'r', encoding='utf-8') as fh:
        return fh.read()


# Inline geometry and fragment shaders with full SH evaluation
# (vertex shader uses VTK's default positioning)

_VERT_SHADER = """\
#version 430 core

layout(location = 0) in vec4 vertexMC;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

void main()
{
    gl_Position = projection_matrix * view_matrix * vertexMC;
}
"""

_GEOM_SHADER = """\
#version 430 core

#define SH_C0  0.28209479177387814f
#define SH_C1  0.4886025119029199f

#define SH_C2_0  1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2  0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4  0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1  2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3  0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5  1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(std430, binding = 0) buffer gaussian_data  { float g_data[]; };
layout(std430, binding = 1) buffer gaussian_order { int   gi[];     };
layout(std430, binding = 2) buffer gaussian_class { int   g_cid[];  };
layout(std430, binding = 3) buffer label_lut      { vec4  g_lut[];  };
layout(std430, binding = 4) buffer gaussian_disp  { int   g_disp[]; };
layout(std430, binding = 5) buffer disp_lut       { vec4  g_dlut[]; };

uniform mat4  view_matrix;
uniform mat4  projection_matrix;
uniform vec3  hfovxy_focal;
uniform vec3  cam_pos;
uniform int   sh_dim;
uniform float scale_modifier;
uniform int   render_mod;
// 0 = label channel off (splats render their SH colour). >0 = painted splats
// (class id > 0) render their label colour blended by this factor AND with their
// opacity boosted toward this factor, so labels read through the translucent
// neighbours that would otherwise dilute a plain SH tint.
uniform float label_boost;
// Data-view opacity boost. >0 lifts EVERY splat's opacity toward this factor so an
// absolute set_colors() recolour (feature similarity / PCA-RGB) reads as a crisp,
// opaque colour instead of averaging into a translucent haze across the many
// low-opacity splats along each ray. Takes precedence over the label channel.
uniform float flat_boost;
// Display channel: per-splat LUT index (g_disp) coloured through a 256-entry
// LUT (g_dlut). The interactive analogue of the mesh SimilarityShader — a
// similarity/class recolour uploads N ints + a 1 KB LUT instead of rewriting
// the SH. disp_mix blends the LUT colour over the evaluated SH colour (so the
// splat KEEPS its spherical-harmonics shading underneath); disp_boost
// optionally lifts opacity like the label channel so the view reads crisp.
// Takes precedence over both flat_boost and the label channel while active.
uniform float disp_mix;
uniform float disp_boost;

out vec3  frag_color;
out float frag_alpha;
out vec3  frag_conic;
out vec2  frag_coordxy;
flat out float frag_label;   // per-splat label peak-alpha (0 = unlabelled)

#define POS_IDX     0
#define ROT_IDX     3
#define SCALE_IDX   7
#define OPACITY_IDX 10
#define SH_IDX      11

vec3 get_vec3(int offset)
{
    return vec3(g_data[offset], g_data[offset + 1], g_data[offset + 2]);
}
vec4 get_vec4(int offset)
{
    return vec4(g_data[offset], g_data[offset + 1], g_data[offset + 2], g_data[offset + 3]);
}

mat3 computeCov3D(vec3 scale, vec4 q)
{
    mat3 S = mat3(0.f);
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    S[2][2] = scale.z;

    float r = q.x, x = q.y, y = q.z, z = q.w;
    mat3 R = mat3(
        1.f - 2.f*(y*y + z*z),   2.f*(x*y - r*z),         2.f*(x*z + r*y),
            2.f*(x*y + r*z), 1.f - 2.f*(x*x + z*z),   2.f*(y*z - r*x),
            2.f*(x*z - r*y),       2.f*(y*z + r*x), 1.f - 2.f*(x*x + y*y)
    );
    mat3 M = S * R;
    return transpose(M) * M;
}

vec3 computeCov2D(vec4 mean_view,
                  float focal_x, float focal_y,
                  float tan_fovx, float tan_fovy,
                  mat3 cov3D, mat4 viewmatrix)
{
    vec4 t = mean_view;
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    t.x = clamp(t.x / t.z, -limx, limx) * t.z;
    t.y = clamp(t.y / t.z, -limy, limy) * t.z;

    mat3 J = mat3(
        focal_x / t.z, 0.f, -(focal_x * t.x) / (t.z * t.z),
        0.f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0.f, 0.f, 0.f
    );
    mat3 W   = transpose(mat3(viewmatrix));
    mat3 T   = W * J;
    mat3 cov = transpose(T) * transpose(cov3D) * T;
    cov[0][0] += 0.3f;
    cov[1][1] += 0.3f;
    return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

void main()
{
    int boxid     = gi[gl_PrimitiveIDIn];
    int total_dim = 3 + 4 + 3 + 1 + sh_dim;
    int start     = boxid * total_dim;

    vec4 g_pos        = vec4(get_vec3(start + POS_IDX), 1.f);
    vec4 g_pos_view   = view_matrix * g_pos;
    vec4 g_pos_clip   = projection_matrix * g_pos_view;
    g_pos_clip.xyz   /= g_pos_clip.w;
    g_pos_clip.w      = 1.f;

    if (any(greaterThan(abs(g_pos_clip.xyz), vec3(1.3f))))
        return;

    vec4  g_rot     = get_vec4(start + ROT_IDX);
    vec3  g_scale   = get_vec3(start + SCALE_IDX);
    float g_opacity = g_data[start + OPACITY_IDX];

    mat3 cov3d  = computeCov3D(g_scale * scale_modifier, g_rot);
    vec2 wh     = 2.f * hfovxy_focal.xy * hfovxy_focal.z;
    vec3 cov2d  = computeCov2D(g_pos_view,
                               hfovxy_focal.z, hfovxy_focal.z,
                               hfovxy_focal.x, hfovxy_focal.y,
                               cov3d, view_matrix);

    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f) return;
    float det_inv = 1.f / det;
    vec3 conic = vec3(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);

    vec2 quadwh_scr = vec2(3.f * sqrt(cov2d.x), 3.f * sqrt(cov2d.z));
    vec2 quadwh_ndc = quadwh_scr / wh * 2.f;

    // ── Spherical Harmonics color evaluation ──────────────────────────
    vec3 color;
    if (render_mod == -1) {
        float depth = -g_pos_view.z;
        depth = (depth < 0.05f) ? 1.f : depth;
        depth = 1.f / depth;
        color = vec3(depth, depth, depth);
    } else {
        int  sh_start = start + SH_IDX;
        vec3 dir      = normalize(g_pos.xyz - cam_pos);
        color         = SH_C0 * get_vec3(sh_start);

        if (sh_dim > 3 && render_mod >= 1) {
            float x = dir.x, y = dir.y, z = dir.z;
            color += -SH_C1 * y * get_vec3(sh_start + 3)
                   +  SH_C1 * z * get_vec3(sh_start + 6)
                   -  SH_C1 * x * get_vec3(sh_start + 9);

            if (sh_dim > 12 && render_mod >= 2) {
                float xx = x*x, yy = y*y, zz = z*z;
                float xy = x*y, yz = y*z, xz = x*z;
                color +=
                    SH_C2_0 * xy          * get_vec3(sh_start + 12) +
                    SH_C2_1 * yz          * get_vec3(sh_start + 15) +
                    SH_C2_2 * (2*zz-xx-yy)* get_vec3(sh_start + 18) +
                    SH_C2_3 * xz          * get_vec3(sh_start + 21) +
                    SH_C2_4 * (xx-yy)     * get_vec3(sh_start + 24);

                if (sh_dim > 27 && render_mod >= 3) {
                    color +=
                        SH_C3_0 * y * (3*xx - yy)      * get_vec3(sh_start + 27) +
                        SH_C3_1 * xy * z                * get_vec3(sh_start + 30) +
                        SH_C3_2 * y * (4*zz - xx - yy) * get_vec3(sh_start + 33) +
                        SH_C3_3 * z * (2*zz - 3*xx - 3*yy) * get_vec3(sh_start + 36) +
                        SH_C3_4 * x * (4*zz - xx - yy) * get_vec3(sh_start + 39) +
                        SH_C3_5 * z * (xx - yy)         * get_vec3(sh_start + 42) +
                        SH_C3_6 * x * (xx - 3*yy)       * get_vec3(sh_start + 45);
                }
            }
        }
        color += 0.5f;
    }

    // ── Label channel override ────────────────────────────────────────
    // A painted splat (class id > 0) takes its label colour and signals the
    // fragment shader to boost its opacity, so the label survives compositing
    // with the unpainted translucent splats around it.
    float frag_label_local = 0.f;
    if (disp_mix > 0.f) {
        // Display channel (feature similarity / multi-class preview): blend the
        // LUT colour for this splat's display value over the SH colour, keeping
        // the underlying shading visible. Optional opacity lift for readability.
        color = mix(color, g_dlut[g_disp[boxid] & 255].rgb, disp_mix);
        frag_label_local = disp_boost;
    } else if (flat_boost > 0.f) {
        // Data view: keep the splat's (set_colors) SH colour but lift its opacity
        // so the absolute colour reads solid instead of a washed-out average.
        frag_label_local = flat_boost;
    }
    // Label channel LAST: painted splats (class id > 0) render their label
    // colour over ANY base view — scene SH, the similarity heatmap, or a data
    // view — at the label boost (the shared label transparency slider), so
    // committed annotations stay visible while querying.
    if (label_boost > 0.f) {
        int cid = g_cid[boxid];
        if (cid > 0) {
            color = mix(color, g_lut[cid].rgb, label_boost);
            frag_label_local = max(frag_label_local, label_boost);
        }
    }

    // ── Emit billboard quad ───────────────────────────────────────────
    vec2 corners_ndc[4] = vec2[4](
        vec2(-1.f, -1.f), vec2( 1.f, -1.f),
        vec2(-1.f,  1.f), vec2( 1.f,  1.f)
    );

    for (int i = 0; i < 4; i++) {
        vec4 pos  = g_pos_clip;
        pos.xy   += corners_ndc[i] * quadwh_ndc;
        gl_Position  = pos;
        frag_color   = color;
        frag_alpha   = g_opacity;
        frag_conic   = conic;
        frag_coordxy = corners_ndc[i] * quadwh_scr;
        frag_label   = frag_label_local;
        EmitVertex();
    }
    EndPrimitive();
}
"""

_FRAG_SHADER = """\
#version 430 core

in vec3  frag_color;
in float frag_alpha;
in vec3  frag_conic;
in vec2  frag_coordxy;
flat in float frag_label;   // per-splat label peak-alpha (0 = unlabelled)

uniform int render_mod;

out vec4 FragColor;

void main()
{
    if (render_mod == -2) {
        FragColor = vec4(frag_color, 1.f);
        return;
    }

    float power = -0.5f * (frag_conic.x * frag_coordxy.x * frag_coordxy.x
                         + frag_conic.z * frag_coordxy.y * frag_coordxy.y)
                - frag_conic.y * frag_coordxy.x * frag_coordxy.y;

    if (power > 0.f)
        discard;

    float opacity = min(0.99f, frag_alpha * exp(power));

    // Labelled splats: lift their peak alpha toward the label_boost factor
    // (still shaped by the Gaussian falloff), so the painted colour composites
    // as near-opaque instead of being diluted by unpainted neighbours.
    if (frag_label > 0.f)
        opacity = max(opacity, min(0.99f, frag_label * exp(power)));

    if (opacity < 1.f / 255.f)
        discard;

    FragColor = vec4(frag_color, opacity);

    if (render_mod == -3) {
        FragColor.a = (FragColor.a > 0.22f) ? 1.f : 0.f;
    } else if (render_mod == -4) {
        FragColor.a   = (FragColor.a > 0.22f) ? 1.f : 0.f;
        FragColor.rgb = FragColor.rgb * exp(power);
    }
}
"""


class VTKNativeGaussianRenderer:
    """Single-pass VTK actor rendering Gaussians with full SH evaluation.

    Replaces the hybrid VTK proxy + ModernGL architecture. Renders in VTK's
    opaque pass to bypass Order-Independent Transparency (OIT), allowing
    correct blend state control.
    """

    def __init__(self, vtk_renderer: vtk.vtkRenderer):
        self._vtk_renderer = vtk_renderer
        self._gaussians: util_gau.GaussianData | None = None

        self._scale_modifier = 1.0
        self._render_mod = 3
        self._sh_dim = 3
        # Per-splat stride in the interleaved flat() buffer:
        # xyz(3) + rot(4) + scale(3) + opacity(1) + sh(sh_dim). SH starts at 11.
        self._stride = 3 + 4 + 3 + 1 + self._sh_dim

        self._data_ssbo: int | None = None
        self._index_ssbo: int | None = None
        self._class_ssbo: int | None = None
        self._lut_ssbo: int | None = None
        self._disp_ssbo: int | None = None
        self._dlut_ssbo: int | None = None
        self._ssbo_ready = False

        self._pending_data: np.ndarray | None = None
        self._pending_index: np.ndarray | None = None
        self._data_dirty = False
        self._index_dirty = False
        self._sort_needed = False

        # Label channel: per-splat class id (int32, indexed by splat id) + a small
        # RGBA LUT indexed by class id, plus the boost factor (0 = channel off).
        # Painted splats (cid > 0) render their LUT colour with boosted opacity.
        self._pending_class: np.ndarray | None = None
        self._class_dirty = False
        self._lut_capacity = 1024
        self._pending_lut = np.zeros((self._lut_capacity, 4), dtype=np.float32)
        self._lut_dirty = True
        self._label_boost = 0.0
        # Data-view opacity boost (0 = off). Set by set_flat_boost when an absolute
        # set_colors() recolour is active so the flat colours render opaque.
        self._flat_boost = 0.0

        # Display channel: per-splat LUT index (int32 [N]) + a 256-entry RGBA
        # LUT + blend/boost factors. The interactive analogue of the mesh
        # SimilarityShader's disp texture: a recolour is an N-int upload, never
        # an SH rewrite. disp_mix 0 = channel off.
        self._pending_disp: np.ndarray | None = None
        self._disp_dirty = False
        self._pending_dlut = np.zeros((256, 4), dtype=np.float32)
        self._pending_dlut[:, 3] = 1.0
        self._dlut_dirty = True
        self._disp_mix = 0.0
        self._disp_boost = 0.0

        # Colour-only partial update state. update_sh() mutates the resident
        # _pending_data in place and queues the touched rows; _on_update_shader
        # uploads only those rows via glBufferSubData (no full re-upload, no
        # depth re-sort).
        self._color_dirty = False
        self._pending_color_rows: list[np.ndarray] = []

        # Depth-sort throttle: skip re-sorting when the view matrix barely moved.
        self._sort_tolerance = 1e-4
        self._last_sort_view: np.ndarray | None = None

        self._poly = vtk.vtkPolyData()
        self._pts = vtk.vtkPoints()
        self._pts.SetDataTypeToDouble()
        self._poly.SetPoints(self._pts)

        self._mapper = vtk.vtkOpenGLPolyDataMapper()
        self._mapper.SetInputData(self._poly)

        self._actor = vtk.vtkActor()
        self._actor.SetMapper(self._mapper)
        self._actor.ForceOpaqueOn()
        self._actor.GetProperty().SetOpacity(1.0)
        self._actor.GetProperty().SetPointSize(1)

        sp = self._actor.GetShaderProperty()
        sp.SetVertexShaderCode(_VERT_SHADER)
        sp.SetGeometryShaderCode(_GEOM_SHADER)
        sp.SetFragmentShaderCode(_FRAG_SHADER)

        @calldata_type(VTK_OBJECT)
        def _on_shader(caller, event, calldata):
            self._on_update_shader(caller, event, calldata)

        self._mapper.AddObserver("UpdateShaderEvent", _on_shader)

        vtk_renderer.AddActor(self._actor)

    @property
    def actor(self) -> vtk.vtkActor:
        return self._actor

    def load(self, gaus: util_gau.GaussianData):
        """Load Gaussian data into the renderer."""
        self._gaussians = gaus
        self._sh_dim = gaus.sh_dim
        self._stride = 3 + 4 + 3 + 1 + self._sh_dim
        n = len(gaus)

        self._pts.SetData(numpy_to_vtk(gaus.xyz.astype(np.float64), deep=True))
        cells = np.empty(2 * n, dtype=np.int64)
        cells[0::2] = 1
        cells[1::2] = np.arange(n)
        verts = vtk.vtkCellArray()
        verts.ImportLegacyFormat(numpy_to_vtk(cells, deep=True, array_type=vtk.VTK_ID_TYPE))
        self._poly.SetVerts(verts)
        self._poly.Modified()

        self._pending_data = gaus.flat().astype(np.float32)
        self._pending_index = np.arange(n, dtype=np.int32)
        self._data_dirty = True
        self._index_dirty = True
        self._sort_needed = True
        # A full (re)load subsumes any queued colour-only rows.
        self._color_dirty = False
        self._pending_color_rows = []
        self._last_sort_view = None

        # Preserve the per-splat label channel across same-N reloads (e.g. a
        # transform re-syncs geometry but keeps the painted labels); reset only
        # when the splat count actually changed (crop/cull).
        if self._pending_class is None or self._pending_class.shape[0] != n:
            self._pending_class = np.zeros(n, dtype=np.int32)
        self._class_dirty = True

        # Same policy for the display channel (per-splat LUT indices).
        if self._pending_disp is None or self._pending_disp.shape[0] != n:
            self._pending_disp = np.zeros(n, dtype=np.int32)
        self._disp_dirty = True

    def update_sh(self, row_indices: np.ndarray, sh_values: np.ndarray) -> bool:
        """Write new SH coefficients for specific splats and queue a colour-only
        GPU update.

        Recolouring does not move geometry, so the depth order is unchanged: this
        path performs an in-place ``glBufferSubData`` over only the touched rows
        and never triggers a full re-upload or a depth re-sort.

        Returns False when a partial update is not possible (buffers not ready or
        a size mismatch, e.g. after a crop changed the splat count); the caller
        should fall back to a full ``load()``.
        """
        if not self._ssbo_ready or self._pending_data is None or self._stride <= 0:
            return False

        n = len(self._gaussians) if self._gaussians is not None else 0
        if n <= 0 or self._pending_data.size != n * self._stride:
            return False

        idx = np.asarray(row_indices, dtype=np.intp).ravel()
        if idx.size == 0:
            return True

        sh_values = np.asarray(sh_values, dtype=np.float32)
        if sh_values.ndim == 1:
            sh_values = sh_values.reshape(1, -1)
        if sh_values.shape[0] != idx.size or sh_values.shape[1] != self._sh_dim:
            return False

        data2d = self._pending_data.reshape(n, self._stride)
        data2d[idx, 11:11 + self._sh_dim] = sh_values
        self._pending_color_rows.append(idx)
        self._color_dirty = True
        return True

    # -- label channel ---------------------------------------------------
    def set_label_boost(self, boost: float):
        """Strength of the label overlay (0 = off). Painted splats render their
        LUT colour blended by this factor and with opacity boosted toward it."""
        self._label_boost = float(max(0.0, min(1.0, boost)))

    def set_flat_boost(self, boost: float):
        """Data-view opacity boost (0 = off). When >0 every splat is rendered
        toward opaque so an absolute set_colors() recolour reads as a solid colour
        rather than a translucent average over the low-opacity splats."""
        self._flat_boost = float(max(0.0, min(1.0, boost)))

    # -- display channel ---------------------------------------------------
    def set_display_values(self, values: np.ndarray) -> bool:
        """Replace the per-splat display LUT indices (uint8 semantics, [N]).

        A full replace is an N-int32 host write + one glBufferData next render —
        ~4 bytes/splat, versus ~236 bytes/splat for an SH rewrite. Returns False
        on a size mismatch (caller may retry after the next load)."""
        if self._pending_disp is None:
            return False
        arr = np.asarray(values).ravel()
        if arr.shape[0] != self._pending_disp.shape[0]:
            return False
        self._pending_disp[:] = arr.astype(np.int32, copy=False)
        self._disp_dirty = True
        return True

    def set_display_lut(self, palette_rgb: np.ndarray) -> None:
        """Upload the display LUT; row i colours display value i ([K<=256, 3|4])."""
        p = np.asarray(palette_rgb, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] < 3:
            return
        m = min(p.shape[0], 256)
        scale = 255.0 if p[:m, :3].max() > 1.0 else 1.0
        self._pending_dlut[:m, :3] = p[:m, :3] / scale
        self._pending_dlut[:m, 3] = 1.0
        self._dlut_dirty = True

    def set_display_mix(self, mix: float):
        """Blend factor of the display-LUT colour over the SH colour (0 = off)."""
        self._disp_mix = float(max(0.0, min(1.0, mix)))

    def set_display_boost(self, boost: float):
        """Opacity lift applied while the display channel is active (0 = natural)."""
        self._disp_boost = float(max(0.0, min(1.0, boost)))

    def update_class_ids(self, element_ids: np.ndarray, class_id: int) -> bool:
        """Set ``class_id`` on a subset of splats (the painted stroke). O(painted).

        Returns False if the class buffer is not allocated yet (caller may retry
        after the next ``load``)."""
        if self._pending_class is None:
            return False
        idx = np.asarray(element_ids, dtype=np.intp).ravel()
        if idx.size == 0:
            return True
        n = self._pending_class.shape[0]
        idx = idx[(idx >= 0) & (idx < n)]
        if idx.size == 0:
            return True
        self._pending_class[idx] = int(class_id)
        self._class_dirty = True
        return True

    def set_class_ids_full(self, class_ids: np.ndarray) -> bool:
        """Replace the entire per-splat class-id array (e.g. a full flush)."""
        if self._pending_class is None:
            return False
        arr = np.asarray(class_ids, dtype=np.int32).ravel()
        if arr.shape[0] != self._pending_class.shape[0]:
            return False
        self._pending_class[:] = arr
        self._class_dirty = True
        return True

    def update_label_lut(self, palette_rgb: np.ndarray) -> None:
        """Upload the label palette; row i == RGB for ``class_id == i`` (0..255)."""
        p = np.asarray(palette_rgb, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] < 3:
            return
        m = min(p.shape[0], self._lut_capacity)
        if p[:m, :3].max() > 1.0:
            self._pending_lut[:m, :3] = p[:m, :3] / 255.0
        else:
            self._pending_lut[:m, :3] = p[:m, :3]
        self._pending_lut[:m, 3] = 1.0
        self._lut_dirty = True

    def set_lut_entry(self, class_id: int, rgb) -> None:
        """Pin one LUT row to the exact colour a stroke painted with."""
        cid = int(class_id)
        if cid <= 0 or cid >= self._lut_capacity:
            return
        try:
            r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
        except Exception:
            return
        scale = 255.0 if max(r, g, b) > 1.0 else 1.0
        self._pending_lut[cid] = (r / scale, g / scale, b / scale, 1.0)
        self._lut_dirty = True

    def set_scale_modifier(self, modifier: float):
        """Set the scale modifier for all splats."""
        self._scale_modifier = float(modifier)

    def set_render_mod(self, mod: int):
        """Set the render mode (-4..3, maps from UI index via index-4)."""
        self._render_mod = int(mod)

    def trigger_sort(self):
        """Mark that splats need re-sorting on the next render."""
        self._sort_needed = True

    def _ensure_shader_sources(self) -> None:
        """Re-assert the custom program if external code cleared any stage.

        On VTK 9.6, ``vtkShaderProperty.ClearAll*ShaderReplacements()`` ALSO
        nulls the corresponding full ``Set*ShaderCode`` override. If a caller
        (e.g. a generic shader-uninstall pass over scene actors) does that to
        this actor, the splats would render through VTK's default fragment
        shader as opaque white billboard quads — permanently. Guarded so the
        codes are only re-set when actually missing (Set*ShaderCode bumps the
        property MTime, which would otherwise force a rebuild every frame).
        """
        try:
            sp = self._actor.GetShaderProperty()
            if not (sp.HasVertexShaderCode() and sp.HasGeometryShaderCode()
                    and sp.HasFragmentShaderCode()):
                sp.SetVertexShaderCode(_VERT_SHADER)
                sp.SetGeometryShaderCode(_GEOM_SHADER)
                sp.SetFragmentShaderCode(_FRAG_SHADER)
        except Exception:
            pass

    def _on_update_shader(self, _caller, _event, calldata):
        program = calldata
        if program is None:
            return

        # Self-heal the shader property (one wrong frame at most, not a session).
        self._ensure_shader_sources()

        # Clear any stale GL errors
        while gl.glGetError() != gl.GL_NO_ERROR:
            pass

        # Initialize SSBOs on first shader compile
        if not self._ssbo_ready:
            ids = gl.glGenBuffers(6)
            self._data_ssbo = int(ids[0])
            self._index_ssbo = int(ids[1])
            self._class_ssbo = int(ids[2])
            self._lut_ssbo = int(ids[3])
            self._disp_ssbo = int(ids[4])
            self._dlut_ssbo = int(ids[5])
            self._ssbo_ready = True

        # Compute camera matrices first — needed for depth sort
        vtk_cam = self._vtk_renderer.GetActiveCamera()
        size = self._vtk_renderer.GetSize()
        w, h = max(size[0], 1), max(size[1], 1)

        pos = np.array(vtk_cam.GetPosition(), dtype=np.float32)
        focal_pt = np.array(vtk_cam.GetFocalPoint(), dtype=np.float32)
        up = np.array(vtk_cam.GetViewUp(), dtype=np.float32)
        fovy_deg = vtk_cam.GetViewAngle()

        fovy_rad = np.radians(fovy_deg)
        aspect = w / h
        htany = np.tan(fovy_rad / 2.0)
        htanx = htany * aspect
        focal_len = h / (2.0 * htany)

        view_mat = np.array(glm.lookAt(
            glm.vec3(*pos.tolist()),
            glm.vec3(*focal_pt.tolist()),
            glm.vec3(*up.tolist()),
        ), dtype=np.float32)

        proj_mat = np.array(glm.perspective(
            fovy_rad, float(aspect), 0.01, 100.0,
        ), dtype=np.float32)

        # Depth-sort Gaussians back-to-front before uploading the index buffer.
        # Skip the (O(N log N)) re-sort when the camera barely moved since the
        # last sort — recolouring and sub-threshold jitter don't change order.
        if self._sort_needed and self._gaussians is not None:
            if (self._last_sort_view is not None
                    and np.max(np.abs(view_mat - self._last_sort_view)) < self._sort_tolerance):
                pass
            else:
                try:
                    sorted_idx = _sort_gaussian(self._gaussians, view_mat)
                    self._pending_index = sorted_idx.reshape(-1).astype(np.int32)
                    self._index_dirty = True
                    self._last_sort_view = view_mat.copy()
                except Exception as e:
                    print(f"Gaussian sort error: {e}")
            self._sort_needed = False

        # Upload data SSBO. A full upload subsumes any queued colour-only rows;
        # otherwise push just the touched SH rows via glBufferSubData.
        if self._data_dirty and self._pending_data is not None:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._data_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            self._pending_data.nbytes,
                            self._pending_data, gl.GL_DYNAMIC_DRAW)
            self._data_dirty = False
            self._color_dirty = False
            self._pending_color_rows = []
        elif self._color_dirty and self._pending_data is not None:
            self._upload_color_rows()
            self._color_dirty = False
            self._pending_color_rows = []

        # Upload sort-index SSBO
        if self._index_dirty and self._pending_index is not None:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._index_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            self._pending_index.nbytes,
                            self._pending_index, gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
            self._index_dirty = False

        # Upload the per-splat class-id buffer (whole array; N int32 is a few MB
        # even at 1M splats — one cheap upload beats scattered sub-ranges).
        if self._class_dirty and self._pending_class is not None:
            cbuf = np.ascontiguousarray(self._pending_class, dtype=np.int32)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._class_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            cbuf.nbytes, cbuf, gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
            self._pending_class = cbuf
            self._class_dirty = False

        # Upload the label LUT.
        if self._lut_dirty and self._pending_lut is not None:
            lbuf = np.ascontiguousarray(self._pending_lut, dtype=np.float32)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._lut_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            lbuf.nbytes, lbuf, gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
            self._pending_lut = lbuf
            self._lut_dirty = False

        # Upload the per-splat display values (whole array; 4N bytes).
        if self._disp_dirty and self._pending_disp is not None:
            dbuf = np.ascontiguousarray(self._pending_disp, dtype=np.int32)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._disp_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            dbuf.nbytes, dbuf, gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
            self._pending_disp = dbuf
            self._disp_dirty = False

        # Upload the display LUT.
        if self._dlut_dirty and self._pending_dlut is not None:
            dlbuf = np.ascontiguousarray(self._pending_dlut, dtype=np.float32)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._dlut_ssbo)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,
                            dlbuf.nbytes, dlbuf, gl.GL_DYNAMIC_DRAW)
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
            self._pending_dlut = dlbuf
            self._dlut_dirty = False

        # Bind SSBOs to shader
        if self._ssbo_ready:
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self._data_ssbo)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self._index_ssbo)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 2, self._class_ssbo)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 3, self._lut_ssbo)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 4, self._disp_ssbo)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 5, self._dlut_ssbo)

        pid = program.GetHandle()

        # Set uniforms
        self._set_mat4(pid, "view_matrix", view_mat)
        self._set_mat4(pid, "projection_matrix", proj_mat)
        self._set_v3(pid, "cam_pos", pos)
        self._set_v3(pid, "hfovxy_focal", np.array([htanx, htany, focal_len], np.float32))
        self._set_1f(pid, "scale_modifier", self._scale_modifier)
        self._set_1i(pid, "sh_dim", self._sh_dim)
        self._set_1i(pid, "render_mod", self._render_mod)
        self._set_1f(pid, "label_boost", self._label_boost)
        self._set_1f(pid, "flat_boost", self._flat_boost)
        self._set_1f(pid, "disp_mix", self._disp_mix)
        self._set_1f(pid, "disp_boost", self._disp_boost)

        # Set up correct blend state and depth testing for the opaque pass
        ostate = self._vtk_renderer.GetRenderWindow().GetState()
        ostate.vtkglEnable(gl.GL_BLEND)
        ostate.vtkglEnable(gl.GL_DEPTH_TEST)
        ostate.vtkglBlendFuncSeparate(
            gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA,
            gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA,
        )
        ostate.vtkglDepthMask(gl.GL_FALSE)

    @staticmethod
    def _set_mat4(pid, name, mat):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            gl.glUniformMatrix4fv(loc, 1, gl.GL_FALSE,
                                  mat.T.flatten().astype(np.float32))

    @staticmethod
    def _set_v3(pid, name, v):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            gl.glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))

    @staticmethod
    def _set_1f(pid, name, v):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            gl.glUniform1f(loc, float(v))

    @staticmethod
    def _set_1i(pid, name, v):
        loc = gl.glGetUniformLocation(pid, name)
        if loc >= 0:
            gl.glUniform1i(loc, int(v))

    def _upload_color_rows(self):
        """Push the queued colour-only row changes via glBufferSubData.

        Touched rows are grouped into contiguous index runs so a typical brush
        dab uploads a handful of small sub-ranges instead of the whole buffer.
        """
        if not self._pending_color_rows or self._pending_data is None or self._data_ssbo is None:
            return

        rows = np.unique(np.concatenate(self._pending_color_rows))
        if rows.size == 0:
            return

        stride = self._stride
        flat = self._pending_data

        # Split the sorted row indices into contiguous runs ([a..b] consecutive).
        if rows.size > 1:
            split_at = np.flatnonzero(np.diff(rows) != 1) + 1
            runs = np.split(rows, split_at)
        else:
            runs = [rows]

        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self._data_ssbo)
        for run in runs:
            a = int(run[0])
            b = int(run[-1])
            seg = flat[a * stride:(b + 1) * stride]
            gl.glBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,
                               a * stride * 4, seg.nbytes, seg)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)

    def cleanup(self):
        """Release GPU resources."""
        if self._vtk_renderer is not None and self._actor is not None:
            self._vtk_renderer.RemoveActor(self._actor)

        try:
            buffers = [b for b in (self._data_ssbo, self._index_ssbo,
                                   self._class_ssbo, self._lut_ssbo,
                                   self._disp_ssbo, self._dlut_ssbo) if b is not None]
            if buffers:
                gl.glDeleteBuffers(len(buffers), buffers)
        except Exception:
            pass

        self._ssbo_ready = False
        self._data_ssbo = None
        self._index_ssbo = None
        self._class_ssbo = None
        self._lut_ssbo = None
        self._disp_ssbo = None
        self._dlut_ssbo = None
