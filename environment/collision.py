# collision.py — Sadece Arazi İçin Hard Constraint Kontrolü
import numpy as np


def check_terrain_collision(pos, terrain, terrain_margin=25.0):
    """
    Sadece araziye çarpma kontrolü (Hard Constraint).
    AGL < margin ise True döndürür.
    """
    x, y, z = pos

    # Arazi kontrolü
    ground = terrain.get_height(x, y)
    if not np.isnan(ground) and ground + terrain_margin >= z:
        return True  # Arazi ile çarpışma var

    return False  # Çarpışma yok