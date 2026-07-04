"""Extract B-Rep features for BRepFormer input.

Produces three feature types from an OpenCascade TopoDS_Shape:
  1. Face UV-grids:  [Nf x 10 x 10 x 7]  (xyz + normal + trimming mask)
  2. Edge curves:    [Ne x 10 x 12]       (xyz + tangent + left_normal + right_normal)
  3. Topology distances: 4 matrices [Nf x Nf] (shortest, centroid, angular, edge-path)
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

UV_GRID_RES = 10
EDGE_SAMPLE_PTS = 10


def extract_face_uv_grids(shape) -> np.ndarray:
    """Sample a 10x10 UV-grid on each face of the shape.

    At each grid point: xyz position (3), surface normal (3), trimming mask (1).
    Returns: ndarray of shape (Nf, 10, 10, 7).
    """
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.gp import gp_Pnt, gp_Vec
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        from OCC.Core.BRepTools import breptools
        from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
        from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
        from OCC.Extend.TopologyUtils import TopologyExplorer
    except ImportError:
        logger.warning("PythonOCC not available for UV-grid extraction")
        return np.zeros((0, UV_GRID_RES, UV_GRID_RES, 7), dtype=np.float32)

    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    nf = len(faces)

    grids = np.zeros((nf, UV_GRID_RES, UV_GRID_RES, 7), dtype=np.float32)

    for fi, face in enumerate(faces):
        adaptor = BRepAdaptor_Surface(face, True)
        u1, u2 = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v1, v2 = adaptor.FirstVParameter(), adaptor.LastVParameter()

        # Clamp extreme parameter ranges
        u1 = max(u1, -1e6)
        u2 = min(u2, 1e6)
        v1 = max(v1, -1e6)
        v2 = min(v2, 1e6)

        props = BRepLProp_SLProps(adaptor, 1, 1e-6)

        # Trimming classifier for the face boundary
        try:
            classifier = BRepTopAdaptor_FClass2d(face, 1e-6)
        except Exception:
            classifier = None

        u_steps = np.linspace(u1, u2, UV_GRID_RES)
        v_steps = np.linspace(v1, v2, UV_GRID_RES)

        for ui, u in enumerate(u_steps):
            for vi, v in enumerate(v_steps):
                props.SetParameters(u, v)

                # Position
                pnt = adaptor.Value(u, v)
                grids[fi, ui, vi, 0] = pnt.X()
                grids[fi, ui, vi, 1] = pnt.Y()
                grids[fi, ui, vi, 2] = pnt.Z()

                # Normal
                if props.IsNormalDefined():
                    n = props.Normal()
                    grids[fi, ui, vi, 3] = n.X()
                    grids[fi, ui, vi, 4] = n.Y()
                    grids[fi, ui, vi, 5] = n.Z()

                # Trimming mask (1 if inside face boundary)
                if classifier is not None:
                    try:
                        from OCC.Core.gp import gp_Pnt2d
                        state = classifier.Perform(gp_Pnt2d(u, v))
                        grids[fi, ui, vi, 6] = 1.0 if state in (TopAbs_IN, TopAbs_ON) else 0.0
                    except Exception:
                        grids[fi, ui, vi, 6] = 1.0
                else:
                    grids[fi, ui, vi, 6] = 1.0

    return grids


def extract_edge_curves(shape) -> np.ndarray:
    """Sample points along each edge of the shape.

    At each point: xyz (3), tangent (3), left face normal (3), right face normal (3).
    Returns: ndarray of shape (Ne, 10, 12).
    """
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
        from OCC.Core.BRepLProp import BRepLProp_CLProps, BRepLProp_SLProps
        from OCC.Extend.TopologyUtils import TopologyExplorer
        from OCC.Core.TopExp import topexp
        from OCC.Core.TopoDS import topods
    except ImportError:
        logger.warning("PythonOCC not available for edge curve extraction")
        return np.zeros((0, EDGE_SAMPLE_PTS, 12), dtype=np.float32)

    topo = TopologyExplorer(shape)
    edges = list(topo.edges())
    ne = len(edges)

    curves = np.zeros((ne, EDGE_SAMPLE_PTS, 12), dtype=np.float32)

    for ei, edge in enumerate(edges):
        adaptor = BRepAdaptor_Curve(edge)
        t1, t2 = adaptor.FirstParameter(), adaptor.LastParameter()

        # Clamp
        t1 = max(t1, -1e6)
        t2 = min(t2, 1e6)

        props = BRepLProp_CLProps(adaptor, 1, 1e-6)

        t_steps = np.linspace(t1, t2, EDGE_SAMPLE_PTS)

        # Get adjacent faces for normal computation
        adj_faces = list(topo.faces_from_edge(edge))
        left_face = adj_faces[0] if len(adj_faces) > 0 else None
        right_face = adj_faces[1] if len(adj_faces) > 1 else left_face

        for ti, t in enumerate(t_steps):
            props.SetParameter(t)

            # Position
            pnt = adaptor.Value(t)
            curves[ei, ti, 0] = pnt.X()
            curves[ei, ti, 1] = pnt.Y()
            curves[ei, ti, 2] = pnt.Z()

            # Tangent
            if props.IsTangentDefined():
                from OCC.Core.gp import gp_Dir
                d = gp_Dir()
                props.Tangent(d)
                curves[ei, ti, 3] = d.X()
                curves[ei, ti, 4] = d.Y()
                curves[ei, ti, 5] = d.Z()

            # Left face normal at closest point
            if left_face is not None:
                n = _face_normal_at_point(left_face, pnt)
                if n is not None:
                    curves[ei, ti, 6:9] = n

            # Right face normal
            if right_face is not None:
                n = _face_normal_at_point(right_face, pnt)
                if n is not None:
                    curves[ei, ti, 9:12] = n

    return curves


def _face_normal_at_point(face, pnt) -> Optional[np.ndarray]:
    """Compute face normal at the closest UV point to a 3D point."""
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.gp import gp_Pnt2d

        surface = BRep_Tool.Surface(face)
        sa = ShapeAnalysis_Surface(surface)
        uv = sa.ValueOfUV(pnt, 1e-3)

        adaptor = BRepAdaptor_Surface(face, True)
        props = BRepLProp_SLProps(adaptor, 1, 1e-6)
        props.SetParameters(uv.X(), uv.Y())

        if props.IsNormalDefined():
            n = props.Normal()
            return np.array([n.X(), n.Y(), n.Z()], dtype=np.float32)
    except Exception:
        pass
    return None


def compute_topology_distances(shape) -> dict[str, np.ndarray]:
    """Compute 4 topology distance matrices between faces.

    Returns dict with keys:
      - "face_shortest": shortest path distance on face adjacency graph
      - "face_centroid": Euclidean distance between face centroids
      - "face_angular": sum of dihedral angles along shortest path
      - "edge_path": minimum edge hops between faces
    Each value is an (Nf, Nf) float32 array.
    """
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        from OCC.Core.TopTools import TopTools_IndexedMapOfShape
        from OCC.Extend.TopologyUtils import TopologyExplorer
        import networkx as nx
    except ImportError as e:
        logger.warning(f"Dependencies not available for topology distances: {e}")
        return {
            "face_shortest": np.zeros((0, 0), dtype=np.float32),
            "face_centroid": np.zeros((0, 0), dtype=np.float32),
            "face_angular": np.zeros((0, 0), dtype=np.float32),
            "edge_path": np.zeros((0, 0), dtype=np.float32),
        }

    topo = TopologyExplorer(shape)
    faces = list(topo.faces())
    nf = len(faces)

    # Use TopTools_IndexedMapOfShape for robust shape-identity lookup (hashes by TShape+orientation).
    # This replaces the broken id(face) approach which failed when topo.faces_from_edge returned
    # fresh Python wrappers around the same C++ shapes.
    face_map = TopTools_IndexedMapOfShape()
    for f in faces:
        face_map.Add(f)

    def face_index(face) -> int:
        """Return 0-based index of face in `faces`, or -1 if not found."""
        i = face_map.FindIndex(face)  # 1-based, 0 if not found
        return i - 1 if i > 0 else -1

    # --- Compute face centroids and normals ---
    centroids = np.zeros((nf, 3), dtype=np.float32)
    normals = np.zeros((nf, 3), dtype=np.float32)

    for fi, face in enumerate(faces):
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        cp = props.CentreOfMass()
        centroids[fi] = [cp.X(), cp.Y(), cp.Z()]

        adaptor = BRepAdaptor_Surface(face, True)
        u_mid = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
        v_mid = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2
        sl_props = BRepLProp_SLProps(adaptor, 1, 1e-6)
        sl_props.SetParameters(u_mid, v_mid)
        if sl_props.IsNormalDefined():
            n = sl_props.Normal()
            normals[fi] = [n.X(), n.Y(), n.Z()]

    # --- Build adjacency graph ---
    G = nx.Graph()
    for i in range(nf):
        G.add_node(i)

    for edge in topo.edges():
        adj_faces = list(topo.faces_from_edge(edge))
        for a in range(len(adj_faces)):
            for b in range(a + 1, len(adj_faces)):
                ia = face_index(adj_faces[a])
                ib = face_index(adj_faces[b])
                if ia >= 0 and ib >= 0 and ia != ib:
                    # Dihedral angle between the two faces
                    dot = np.clip(np.dot(normals[ia], normals[ib]), -1, 1)
                    angle = float(np.arccos(dot))
                    G.add_edge(ia, ib, weight=1.0, angle=angle)

    # --- Distance matrices ---
    # 1. Face Shortest Distance (Dijkstra on unit-weight graph)
    face_shortest = np.full((nf, nf), float("inf"), dtype=np.float32)
    np.fill_diagonal(face_shortest, 0.0)
    try:
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        for i, dists in lengths.items():
            for j, d in dists.items():
                face_shortest[i, j] = d
    except Exception:
        pass

    # 2. Face Centroid Distance (Euclidean)
    face_centroid = np.zeros((nf, nf), dtype=np.float32)
    for i in range(nf):
        face_centroid[i] = np.linalg.norm(centroids - centroids[i], axis=1)

    # 3. Face Angular Distance (sum of dihedral angles along shortest path)
    face_angular = np.full((nf, nf), float("inf"), dtype=np.float32)
    np.fill_diagonal(face_angular, 0.0)
    try:
        for source in range(nf):
            paths = nx.single_source_dijkstra_path(G, source, weight="angle")
            for target, path in paths.items():
                total_angle = 0.0
                for k in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[k], path[k + 1])
                    total_angle += edge_data.get("angle", 0.0)
                face_angular[source, target] = total_angle
    except Exception:
        pass

    # 4. Shortest Edge Path (same as face_shortest for unit-weight graph)
    edge_path = face_shortest.copy()

    return {
        "face_shortest": face_shortest,
        "face_centroid": face_centroid,
        "face_angular": face_angular,
        "edge_path": edge_path,
    }
