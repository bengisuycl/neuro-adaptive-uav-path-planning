# planners/dijkstra.py
# F-16 TEZ PROJESİ - STANDARDİZE EDİLMİŞ DIJKSTRA PLANLAYICI
# Özellikler: Feasibility Engine Entegrasyonu + Snap-to-Goal Düzeltmesi

import heapq
import math
from wcci_conference_project.utils.feasibility import FeasibilityEngine


class DijkstraPlanner:
    def __init__(self, terrain, threats,
                 safe_alt=800.0,
                 step=50,
                 max_iter=2000000,
                 goal_tolerance=2000.0,  # Standart hedef yarıçapı
                 check_resolution=50.0,  # Standart güvenlik tarama hassasiyeti
                 threat_weight=20.0,  # Standart risk ağırlığı
                 **kwargs):  # Fazlalık parametreleri yutmak için (örn: heuristic_weight)

        self.terrain = terrain
        self.threats = threats
        self.safe_alt = safe_alt
        self.step = step
        self.max_iter = max_iter
        self.goal_tolerance = goal_tolerance

        # --- ORTAK HAKEM MOTORU (FEASIBILITY ENGINE) ---
        # Tüm algoritmalarla eşit şartlarda (ceteris paribus) yarışması için
        self.feasibility = FeasibilityEngine(
            terrain, threats,
            check_resolution=check_resolution,
            w_risk=threat_weight,
            use_neural_risk=kwargs.get("use_neural_risk", False),
            neural_risk_weight=kwargs.get("neural_risk_weight", 0.0),
            neural_risk_samples=kwargs.get("neural_risk_samples", 3),
            neural_corridor_half_width=kwargs.get("neural_corridor_half_width", 1800.0),
        )

    def heuristic(self, a, b):
        """
        Dijkstra'da heuristic normalde 0'dır ama hedef kontrolü (distance check)
        için Öklid mesafesi fonksiyonunu yardımcı olarak kullanıyoruz.
        """
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def reconstruct_path(self, came_from, current):
        """Bulunan yolu geriye doğru takip ederek oluşturur."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        # 3D koordinat formatına (x, y, alt) çevir
        return [(p[0], p[1], self.safe_alt) for p in path]

    def plan(self, start, goal):
        print("   🔄 Dijkstra Planner Started (Standardized + SnapToGoal)")

        start2d = (float(start[0]), float(start[1]))
        goal2d = (float(goal[0]), float(goal[1]))

        # Başlangıç Noktası Güvenlik Kontrolü
        is_start_ok, _, _ = self.feasibility.check_segment(start2d, start2d, self.safe_alt)
        if not is_start_ok:
            print("⚠️ Dijkstra Başlatılamadı: Başlangıç noktası güvenli değil!")
            return None

        open_set = []
        heapq.heappush(open_set, (0.0, start2d))

        came_from = {}
        cost_so_far = {start2d: 0.0}
        closed_set = set()

        # Grid Hareketleri (8 Yön)
        motions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Yatay/Dikey
            (1, 1), (-1, 1), (1, -1), (-1, -1)  # Çapraz
        ]

        iter_cnt = 0
        while open_set:
            iter_cnt += 1
            if iter_cnt > self.max_iter:
                print(f"⚠️ Dijkstra Max Iteration ({self.max_iter}) Reached!")
                break

            current_cost, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            # --- 1. HEDEF KONTROLÜ VE "SNAP-TO-GOAL" ---
            # Eğer hedef toleransı içine girdiysek:
            if self.heuristic(current, goal2d) < self.goal_tolerance:
                path = self.reconstruct_path(came_from, current)

                # KRİTİK DÜZELTME:
                # Grid noktası hedefin tam merkezi olmayabilir.
                # Son noktadan tam hedef merkezine (goal2d) direk uçuş var mı bakıyoruz.
                is_feasible_to_goal, _, _ = self.feasibility.check_segment(current, goal2d, self.safe_alt)

                if is_feasible_to_goal:
                    # Eğer yol açıksa, tam hedefi de yola ekle (Simülasyon başarısı için şart)
                    path.append((goal2d[0], goal2d[1], self.safe_alt))

                return path

            # --- 2. KOMŞULARI GEZ ---
            for dx, dy in motions:
                neighbor = (current[0] + dx * self.step, current[1] + dy * self.step)

                if neighbor in closed_set:
                    continue

                # --- 3. ORTAK FEASIBILITY MOTORU ---
                # Artık manuel check yok, FeasibilityEngine var.
                is_feasible, segment_cost, reason = self.feasibility.check_segment(
                    current, neighbor, self.safe_alt
                )

                if not is_feasible:
                    continue  # Engel, Tehdit veya Sınır hatası -> Atla

                new_cost = cost_so_far.get(current, float('inf')) + segment_cost

                # Dijkstra'da priority = cost (Heuristic yok)
                priority = new_cost

                if new_cost < cost_so_far.get(neighbor, float('inf')):
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (priority, neighbor))

        print("❌ Dijkstra yol bulamadı (Open set tükendi).")
        return None
