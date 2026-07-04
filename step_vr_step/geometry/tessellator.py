"""B-Rep tessellation wrapper for converting OCCT shapes to triangle meshes.

Wraps BRepMesh_IncrementalMesh with configurable quality parameters.
Used for preview mesh generation and glTF/Datasmith mesh output.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import TessellationConfig

logger = logging.getLogger(__name__)


def tessellate_shape(shape, config: TessellationConfig | None = None) -> Optional[tuple]:
    """Tessellate an OCCT shape into triangle mesh.

    Args:
        shape: TopoDS_Shape to tessellate
        config: Tessellation quality configuration

    Returns:
        Tuple of (vertices_Nx3, faces_Mx3, normals_Nx3) or None if shape is invalid
    """
    if config is None:
        config = TessellationConfig()

    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.BRepTools import breptools
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopLoc import TopLoc_Location
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.gp import gp_Pnt

        if shape is None or shape.IsNull():
            return None

        # Clear cached triangulation so deflection params are actually respected
        breptools.Clean(shape)

        # Perform tessellation
        mesh = BRepMesh_IncrementalMesh(
            shape,
            config.linear_deflection,
            config.relative,
            config.angular_deflection,
            config.parallel,
        )
        mesh.Perform()

        if not mesh.IsDone():
            logger.warning("Tessellation failed")
            return None

        all_vertices = []
        all_normals = []
        all_faces = []
        vertex_offset = 0

        exp = TopExp_Explorer(shape, TopAbs_FACE)
        while exp.More():
            face = exp.Current()
            loc = TopLoc_Location()
            triangulation = BRep_Tool.Triangulation(face, loc)

            if triangulation is None:
                exp.Next()
                continue

            trsf = loc.Transformation()
            nb_nodes = triangulation.NbNodes()
            nb_triangles = triangulation.NbTriangles()

            # Extract vertices
            for i in range(1, nb_nodes + 1):
                pnt = triangulation.Node(i)
                pnt.Transform(trsf)
                all_vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

            # Extract normals (if available)
            has_normals = triangulation.HasNormals()
            for i in range(1, nb_nodes + 1):
                if has_normals:
                    normal = triangulation.Normal(i)
                    all_normals.append([normal.X(), normal.Y(), normal.Z()])
                else:
                    all_normals.append([0, 0, 1])  # placeholder

            # Extract triangles
            for i in range(1, nb_triangles + 1):
                tri = triangulation.Triangle(i)
                n1, n2, n3 = tri.Get()
                all_faces.append([
                    n1 - 1 + vertex_offset,
                    n2 - 1 + vertex_offset,
                    n3 - 1 + vertex_offset,
                ])

            vertex_offset += nb_nodes
            exp.Next()

        if not all_vertices:
            return None

        vertices = np.array(all_vertices, dtype=np.float64)
        faces = np.array(all_faces, dtype=np.int32)
        normals = np.array(all_normals, dtype=np.float64)

        logger.debug(f"Tessellated: {len(vertices)} vertices, {len(faces)} triangles")
        return vertices, faces, normals

    except ImportError:
        logger.warning("PythonOCC not available for tessellation")
        return None
    except Exception as e:
        logger.error(f"Tessellation error: {e}")
        return None
