# Crocodile — Pipeline Diagram

See [PIPELINE.md](PIPELINE.md) for the prose version of this — roles, current
status, terminology. This file is the visual companion; keep them in sync.

**Legend**: grey dashed = legacy/superseded, not on the active path · amber
dashed = designed but not yet built · blue = current active-work highlight.

## Full System Overview

```mermaid
flowchart TB
    subgraph SENSORS["Acquisition (Arduino / Teensy @ 1000 Hz)"]
        S1[PPG / Heart Rate]
        S2[EDA / Skin Conductance]
        S3[Respiration]
    end

    subgraph SIGNAL["Signal Processing @ 1000Hz (BioSPPy · lib/biodata.py)"]
        P1[BPM / HRV]
        P2[SCR / SCL]
        P3[Breathing Rate]
    end

    subgraph DATA["Dataset Assembly (GAN / classifier, 1000Hz)"]
        V[Video Frames\n256×256 px]
        TS[Timestamp Alignment]
        CD[CrocodileDataset\nframes + biodata features]
        ED[EmotionDataset\nwindowed signals + labels]
    end

    subgraph BIODATA_PIPE["biodata_pipeline — Feature Extraction @ 100Hz\n(separate sampling rate — do not mix with 1000Hz path above)"]
        BF[EnhancedContinuousFeatureExtractor\n67 features/sec]
        CF[(continuous_features.csv)]
        BF --> CF
    end

    subgraph TRAIN_GAN["GAN Training — LEGACY, superseded by latent_pipeline"]
        G[Generator\nlatent z + biodata → image]
        D[Discriminator\nreal / fake]
        FID[FID Evaluation]
    end

    subgraph TRAIN_ENC["latent_pipeline — active preprocessing pipeline"]
        E1[Stage 1: Extract frames\n& build manifests]
        E2[Stage 2: Train VGG Encoder\nimage → W ∈ ℝ⁵¹²]
        E3[Stage 3: Validate\nEncoder vs Optimization]
        E4[Stage 4: Assemble\nbiodata → W dataset]
        E5["Stage 5: Biodata→W Regressor\nNOT YET BUILT"]
        SG[StyleGAN2 · frozen\nW → face]
        LPIPS[LPIPS + MSE loss]
    end

    subgraph TRAIN_CLS["Emotion Classifier"]
        CLS[1D ResNet\nECGResNet]
        EMO[Emotion Label\nANG · ARO · FEA · HAP]
    end

    subgraph INFER["Runtime Pipeline — NOT YET BUILT (design only)"]
        RF[Participant raw signals]
        FT[Feature Extraction]
        ALIGN["Alignment\nbiodata_pipeline transformer\nRidge / OT class-conditional\n(model exists — not wired to a live script)"]
        REG["Biodata → W Regressor\nNOT YET BUILT"]
        WV[W vector ℝ⁵¹²]
        FACE[StyleGAN2 · frozen\nSynthetic Face]
    end

    %% Acquisition → Processing (1000Hz path: GAN training / classifier)
    S1 & S2 & S3 --> SIGNAL
    P1 & P2 & P3 --> TS
    V --> TS
    TS --> CD
    TS --> ED

    %% Acquisition → biodata_pipeline (separate 100Hz path)
    S1 & S2 & S3 --> BF

    %% GAN training (legacy)
    CD --> G
    G --> D
    D -->|adversarial loss| G
    G --> FID

    %% Latent pipeline
    V --> E1
    E1 --> E2
    E2 -->|encoder| SG
    SG --> LPIPS
    LPIPS -->|gradient| E2
    E2 --> E3
    E3 --> E4
    CF --> E4
    E4 --> E5

    %% Classifier training
    ED --> CLS
    CLS --> EMO

    %% Runtime chain (not yet built)
    RF --> FT --> ALIGN --> REG --> WV --> FACE

    %% Styling — legacy
    style TRAIN_GAN fill:#f9fafb,stroke:#9ca3af,stroke-dasharray: 5 5
    style G fill:#f3f4f6,stroke:#9ca3af,stroke-dasharray: 5 5
    style D fill:#f3f4f6,stroke:#9ca3af,stroke-dasharray: 5 5
    style FID fill:#f3f4f6,stroke:#9ca3af,stroke-dasharray: 5 5
    style CD fill:#f3f4f6,stroke:#9ca3af,stroke-dasharray: 5 5

    %% Styling — not yet built
    style E5 fill:#fef3c7,stroke:#f59e0b,stroke-dasharray: 5 5
    style REG fill:#fef3c7,stroke:#f59e0b,stroke-dasharray: 5 5
    style RF fill:#fef3c7,stroke:#f59e0b,stroke-dasharray: 5 5
    style FT fill:#fef3c7,stroke:#f59e0b,stroke-dasharray: 5 5
    style WV fill:#fef3c7,stroke:#f59e0b,stroke-dasharray: 5 5
    style FACE fill:#fef3c7,stroke:#f59e0b,stroke-dasharray: 5 5

    %% Styling — exists, just not wired up yet
    style ALIGN fill:#dbeafe,stroke:#3b82f6,stroke-dasharray: 5 5
```

---

## Biodata Pipeline (standalone detail)

Two distinct roles: feed `latent_pipeline` Stage 4 (feature extraction), and
supply the runtime pipeline's alignment step (cross-subject transformer).

```mermaid
flowchart LR
    subgraph EXTRACT["Feature Extraction"]
        RAW[Raw sensor CSV\n100Hz: heart · gsr · respiration]
        FE[EnhancedContinuousFeatureExtractor\nenhanced_respiratory_features.py]
        CF[(continuous_features.csv\n67 features/sec)]
        RAW --> FE --> CF
    end

    subgraph SANITY["Windowing + Classification Eval\n(sanity check only — not on the critical path)"]
        DS[Data Slicer\ndata_slicer.py]
        EVAL[GroupKFold CV\nRandom Forest]
        CF --> DS --> EVAL
    end

    subgraph ALIGN["Cross-Subject Alignment — feeds the runtime pipeline"]
        REF[Dauphinais reference features]
        NEW[New subject\ncalibration recording]
        XF[Transformer\ntrain_transformer.py\nRidge / OT class-conditional]
        MODEL[("transformer_ot_classconditional.pkl\nbest: NPA 54.2%")]
        REF --> XF
        NEW --> XF
        XF --> MODEL
    end

    CF -.->|reference set| REF

    style ALIGN fill:#dbeafe,stroke:#3b82f6
```

---

## Latent Pipeline — Stage Detail

```mermaid
flowchart TD
    S1["Stage 1\nstage1_extract.py\nExtract frames, align timestamps,\nbuild pool manifests"]
    S2["Stage 2\ntrain_synthetic.py → train_frames.py\nVGG EmotionEncoder\nimage B×3×H×H → W B×512\nLPIPS + MSE + temporal smoothness\n(stalled epoch 14/20 — resume from latest.pt)"]
    S3["Stage 3\nstage3_validate.py\nCompare encoder vs\noptimization-based inversion"]
    S4["Stage 4\nstage4_assemble.py\nRun encoder on all biodata frames\nwrite biodata → W CSV"]
    S5["Stage 5\nNOT YET BUILT\nFit biodata → W regressor\non biodata_w_dataset.csv"]

    S1 --> S2 --> S3 --> S4 --> S5

    style S2 fill:#dbeafe,stroke:#3b82f6
    style S5 fill:#fef3c7,stroke:#f59e0b,stroke-dasharray: 5 5
```
