"""Raw-data paths, gene sets and ephys features shared by the AnnData-based analyses."""
from paths import REPO, VALIDATION

RAW = REPO                                          # raw public data live at the repository root

PATCHSEQ_DIR = RAW / "PatchSeqExpression"
EPHYS_CSV = RAW / "m1_patchseq_ephys_features.csv"
ALLEN_DIR = RAW / "Gene_Expression"

# Rapid primary-response genes (rPRGs): transcribed within minutes of activity,
# independent of new protein synthesis (Tyssowski et al. 2018, Neuron).
IEG_RAPID = ["Fos", "Fosb", "Junb", "Jun", "Egr1", "Egr2", "Egr3", "Egr4",
             "Npas4", "Arc", "Nr4a1", "Nr4a2", "Nr4a3", "Ier2", "Dusp1",
             "Dusp6", "Btg2", "Klf4", "Atf3"]
# Delayed primary / secondary-response genes: hours-scale plasticity effectors.
IEG_DELAYED = ["Bdnf", "Nptx2", "Homer1", "Pcsk1", "Gadd45b", "Gadd45g",
               "Rgs2", "Scg2", "Vgf", "Inhba", "Crem", "Syt4", "Ptgs2"]
# Broad IEG / plasticity-effector set (key "IEG_proposal").
IEG_PROPOSAL = ["Fos", "Fosb", "Junb", "Egr1", "Egr2", "Npas4", "Arc", "Homer1",
                "Nr4a1", "Nr4a2", "Bdnf", "Nptx2", "Syt4", "Camk2g", "Rgs4",
                "Dusp1", "Atf3", "Crem", "Gadd45b", "Gadd45g", "Prkcz"]
# Generic cell-stress / dissociation response (van den Brink et al. 2017).
STRESS = ["Hspa1a", "Hspa1b", "Hspa8", "Hsph1", "Dnajb1", "Hspb1", "Ubc",
          "Socs3", "Zfp36", "Cebpd", "Cebpb", "Mt1", "Mt2"]

GENE_SETS = {"IEG_rapid": IEG_RAPID, "IEG_delayed": IEG_DELAYED,
             "IEG_proposal": IEG_PROPOSAL, "stress": STRESS}

# Intrinsic electrophysiology features (Scala et al. 2021 extraction).
EPHYS_KEY = ["Input resistance (MOhm)", "Resting membrane potential (mV)",
             "Rheobase (pA)", "AP threshold (mV)", "AP width (ms)",
             "Afterhyperpolarization (mV)", "Membrane time constant (ms)",
             "Sag ratio", "ISI adaptation index", "Max number of APs",
             "Burstiness", "Latency (ms)"]

VALIDATION_TABLES = VALIDATION / "tables"           # harmonised M1 -> V1 set-test tables
V1_GOUWENS = RAW / "ReplicationData" / "MouseV1_Gouwens2020"   # Allen V1 interneuron Patch-seq
