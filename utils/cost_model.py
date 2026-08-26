# utils/cost_model.py
import numpy as np

# --- EVRENSEL SABİTLER (TÜM ALGORİTMALAR İÇİN) ---
# Bu değerler tüm plannerlar için aynı "dünyayı" temsil eder.
TERRAIN_COLLISION_PENALTY = 1e6  # Çarpışma: Çok yüksek maliyet (neredeyse yasak)
HARD_THREAT_PENALTY = 1e6  # Hard Zone: Çok yüksek maliyet
THREAT_COST_SCALE = 5000.0  # Soft zone risk ölçeği


def compute_threat_cost(px, py, threats):
    """
    Verilen (x, y) noktası için toplam tehdit maliyetini hesaplar.

    Dönüş:
        total_cost (float): Soft riskler + Hard zone cezası
        in_hard_zone (bool): Nokta herhangi bir hard zone içinde mi?
    """
    total_cost = 0.0
    in_hard_zone = False

    for t in threats:
        # Threat sınıfı hard zone için 'inf' döner
        c = t.get_risk_cost(px, py, cost_scale=THREAT_COST_SCALE)

        if c == float('inf'):
            in_hard_zone = True
            # Hard zone cezası ekle (sonsuz değil, yönetilebilir büyük sayı)
            total_cost += HARD_THREAT_PENALTY
        else:
            # Soft zone riskini ekle
            total_cost += c

    return total_cost, in_hard_zone