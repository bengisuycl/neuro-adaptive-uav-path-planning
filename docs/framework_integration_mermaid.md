# Project Framework Integration

```mermaid
flowchart TB
  %% Top-level styling
  classDef input fill:#253b57,stroke:#f0c419,color:#f8f8f2,stroke-width:2px;
  classDef perception fill:#0e3142,stroke:#00d8ff,color:#f8f8f2,stroke-width:2px;
  classDef decision fill:#251f36,stroke:#c77dff,color:#f8f8f2,stroke-width:2px;
  classDef action fill:#0f3328,stroke:#44ffb0,color:#f8f8f2,stroke-width:2px;
  classDef output fill:#301722,stroke:#ff5c8a,color:#f8f8f2,stroke-width:2px;
  classDef support fill:#1f2937,stroke:#9ca3af,color:#f8f8f2,stroke-width:1px;

  subgraph I["INPUT"]
    DEM["DEM / TERRAIN<br/>Terrain(DEM raster)<br/>h_terrain(x,y), slope, bounds<br/>AGL margin enforcement"]
    THR["THREAT FIELD<br/>Threat objects: static + dynamic<br/>hard zone, soft zone, patrol corridor<br/>predictive directional risk"]
    MIS["MISSION PARAMS<br/>Start / Goal / Goal radius<br/>Time budget<br/>Scenario definition S1...S4"]
    ACM["AIRCRAFT MODEL<br/>Reduced-order fixed-wing model<br/>state: [x,y,h,psi], speed V<br/>inputs: throttle, phi, gamma"]
    DST["DISTURBANCES<br/>Wind_xy, phi noise, gamma noise<br/>Monte Carlo repetitions"]
  end

  subgraph P["PERCEPTION"]
    DNN["DNN-TRE<br/>Neural tactical risk estimator"]
    FEAT["Feature Vector<br/>dist_norm<br/>rel_alt_norm<br/>slope<br/>threat level"]
    NET["MLP Risk Regressor<br/>4-128-128-64-1<br/>ReLU + Softplus<br/>log1p target transform"]
    RSK["Risk Summary<br/>peak risk<br/>mean risk<br/>corridor risk<br/>high-risk fraction"]
    FEAT --> NET --> RSK
  end

  subgraph D["DECISION"]
    ADV["RL Advisor<br/>Meta-decision layer"]
    FUSE["Multi-Attribute Risk Fusion<br/>capability score<br/>opportunity score<br/>pressure score<br/>fused tactical risk"]
    SEL["Planner Selection Policy<br/>low risk -> RL-Pilot<br/>moderate risk -> K-GNP<br/>high/dynamic risk -> T-GnP"]
    ADV --> FUSE --> SEL
  end

  subgraph A["ACTION"]
    AST["A* / Dijkstra<br/>Grid-search baselines"]
    RRT["RRT* / PSO<br/>Sampling / optimization baselines"]
    KGN["K-GNP<br/>Kinematic Guidance Navigation Planner<br/>heading-aware feasible expansion<br/>bounded curvature via R_min"]
    TGN["T-GnP<br/>Tactical Guidance Navigation Planner<br/>turn penalty + adaptive speed<br/>optional neural risk augmentation"]
    RLP["RL-Pilot<br/>DQN-based reactive policy"]
    NAD["Neuro-Adaptive<br/>Perception -> Advisor -> planner switch"]
    SIM["SIMULATION / EXECUTION<br/>Lookahead guidance<br/>curvature-based target speed<br/>throttle-bank-gamma tracking<br/>terrain and threat updates"]
  end

  subgraph O["OUTPUT"]
    BEN["BENCHMARK METRICS<br/>Path length | Mission time | Risk exposure<br/>Min AGL | Threat violations | Track error<br/>Calc. time | Success rate"]
    FIG["FIGURES / DEMO ASSETS<br/>2D comparative map<br/>3D tactical + dynamic plots<br/>GIF flight animation<br/>DNN-TRE risk-map comparison"]
  end

  DEM --> FEAT
  THR --> FEAT
  MIS --> FEAT
  ACM --> FEAT

  RSK --> ADV
  MIS --> ADV
  THR --> ADV

  SEL --> KGN
  SEL --> TGN
  SEL --> RLP

  MIS --> AST
  MIS --> RRT
  MIS --> NAD
  NAD --> SEL

  AST --> SIM
  RRT --> SIM
  KGN --> SIM
  TGN --> SIM
  RLP --> SIM

  ACM --> SIM
  DST --> SIM
  THR --> SIM
  DEM --> SIM

  SIM --> BEN
  SIM --> FIG
  DNN --> FIG

  DNN --- FEAT
  DNN --- NET
  DNN --- RSK

  class DEM,THR,MIS,ACM,DST input;
  class DNN,FEAT,NET,RSK perception;
  class ADV,FUSE,SEL decision;
  class AST,RRT,KGN,TGN,RLP,NAD,SIM action;
  class BEN,FIG output;
```

## Notes

- `DNN-TRE` is a standalone tactical risk estimation module and also a perception layer for the neuro-adaptive planner.
- `RL Advisor` does not directly fly the aircraft; it selects among planner modes.
- `K-GNP` and `T-GnP` are the core flyability-aware planning contributions.
- `Simulation` is the common execution layer used for fair comparison across all planners.
