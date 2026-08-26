from wcci_conference_project.environment.threats import Threat


def get_all_scenarios(safe_alt):
    """
    Return all thesis benchmark scenarios.

    safe_alt is computed from the DEM before scenario creation so all
    scenarios remain terrain-safe by default.
    """
    return [
        {
            "id": "S1_Base",
            "name": "S1_Base (Acik Arazi)",
            "description": "Az sayida tehdit, genis manevra alani.",
            "start": (450000, 4350000, safe_alt),
            "goal": (475000, 4375000, safe_alt),
            "threats": [
                Threat(460000, 4360000, 2000, "SAM", 0.9),
                Threat(465000, 4365000, 1500, "Radar", 0.5),
                Threat(457000, 4358000, 1000, "EW", 0.7),
            ],
        },
        {
            "id": "S2_Dense",
            "name": "S2_Dense (Yogun Tehdit)",
            "description": "Dar gecitler ve yogun SAM aglari.",
            "start": (450000, 4350000, safe_alt),
            "goal": (475000, 4380000, safe_alt),
            "threats": [
                Threat(460000, 4360000, 2000, "SAM", 0.9),
                Threat(462000, 4365000, 2000, "SAM", 0.9),
                Threat(468000, 4370000, 2000, "SAM", 0.9),
                Threat(455000, 4355000, 1500, "Radar", 0.6),
            ],
        },
        {
            "id": "S3_Long",
            "name": "S3_Long (Uzun Menzil)",
            "description": "Uzun ucus suresi ve stratejik kacinma.",
            "start": (450000, 4350000, safe_alt),
            "goal": (480000, 4385000, safe_alt),
            "threats": [
                Threat(460000, 4360000, 2500, "SAM_Big", 1.0),
                Threat(470000, 4375000, 2400, "SAM", 0.9),
            ],
        },
        {
            "id": "S4_DynamicThreat",
            "name": "S4_DynamicThreat (Hareketli IADS)",
            "description": "Kenar bolgelere itilmis hareketli radar/SAM devriyeleri ile ortada dar ama kullanilabilir bir taktik koridor.",
            "start": (425000, 4342000, safe_alt),
            "goal": (490000, 4410000, safe_alt),
            "threats": [
                Threat(
                    440000,
                    4354000,
                    1700,
                    "Radar",
                    0.75,
                    dynamic=True,
                    speed_mps=40.0,
                    motion_pattern="patrol_line",
                    motion_param=10000.0,
                    predictive_horizon_s=42.0,
                    corridor_weight=0.45,
                ),
                Threat(
                    469000,
                    4390000,
                    1800,
                    "SAM",
                    0.92,
                    dynamic=True,
                    speed_mps=26.0,
                    motion_pattern="patrol_circle",
                    motion_param=4200.0,
                    predictive_horizon_s=36.0,
                    corridor_weight=0.46,
                ),
                Threat(449000, 4396000, 1350, "EW", 0.65),
                Threat(478500, 4372000, 2000, "SAM", 0.96),
                Threat(
                    484500,
                    4403000,
                    1600,
                    "SAM",
                    0.85,
                    dynamic=True,
                    speed_mps=32.0,
                    motion_pattern="patrol_line",
                    motion_param=7500.0,
                    predictive_horizon_s=30.0,
                    corridor_weight=0.42,
                ),
            ],
        },
        {
            "id": "S5_RLPilotDemo",
            "name": "S5_RLPilotDemo (Dusuk Tehdit / Demo)",
            "description": "Tehditlerin gorev koridorundan uzak tutuldugu, RL-Pilot seciminin dogal olarak tetiklendigi demo senaryosu.",
            "start": (448000, 4351000, safe_alt),
            "goal": (474000, 4372000, safe_alt),
            "threats": [
                Threat(430000, 4405000, 1800, "Radar", 0.45),
                Threat(492000, 4330000, 2000, "SAM", 0.55),
            ],
        },
    ]
