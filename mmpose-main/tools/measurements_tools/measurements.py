"""Foot13: Calculation of five metrics; the input consists of 13 key points in a fixed order."""

import numpy as np


def angle(a, vertex, b):
    """Angle formed by three points, in degrees."""
    u, v = a - vertex, b - vertex
    cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def compute_measurements(points):
    """Return CA, HVA, NMP, SI, and CSI."""
    points = np.asarray(points, dtype=float).reshape(13, 2)
    FM, FL, MM, ML, HM, HL, HaMP, HeMP, FMMP, HR, STF, NT, DIM = points

    with np.errstate(divide="ignore", invalid="ignore"):
        ca = angle(DIM, FM, HM)
        hva = angle(HaMP, FMMP, HeMP)
        hva = min(hva, 180 - hva)

        axis = STF - HR
        offset = NT - HR
        cross = axis[0] * offset[1] - axis[1] * offset[0]
        nmp = abs(cross) / np.dot(axis, axis)  * 100

        midfoot_width = np.linalg.norm(MM - ML)
        si = midfoot_width / np.linalg.norm(HM - HL) * 100
        csi = midfoot_width / np.linalg.norm(FM - FL) * 100

    values = {
        "ca_deg": ca,
        "hva_deg": hva,
        "nmp_percent": nmp,
        "si_percent": si,
        "csi_percent": csi,
    }
    return {name: float(value) if np.isfinite(value) else np.nan
            for name, value in values.items()}
