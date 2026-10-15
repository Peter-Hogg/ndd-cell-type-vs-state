#!/usr/bin/env bash
# Run every analysis in dependency order. Raw inputs are expected at the repository root (see README).
# Steps that download or re-quantify raw data are skipped when their outputs exist:
#   --refetch-ibl   re-stream IBL units from DANDI:000409 (~14 GB)
# Not run here (manual, long downloads or external tools):
#   analysis/v1_patchseq/scripts/09_v1_extract_probe.py      single-NWB extraction check
#   analysis/v1_patchseq/scripts/10_v1_quantify_l5et.py      Salmon re-quantification of V1 L5 ET reads
#   analysis/v1_patchseq/scripts/09_v1_extract_ephys.py --download --extract   DANDI:000020 NWBs (~31 GB)
set -euo pipefail
cd "$(dirname "$0")/analysis"
run() { echo "== $*"; uv run python "$@"; }

M1=m1_patchseq/scripts; AT=allen_atlas/scripts; V1=v1_patchseq/scripts
MQ=macaque_patchseq/scripts; IBL=ibl_neuropixels/scripts; X=cross_dataset/scripts

# ---- gene-level pipeline (M1 discovery, atlas, macaque and V1 tests)
run $M1/01_l5_plasticity_ephys.py
run $M1/01_l5_plasticity_ephys.py --exon-only
run $M1/02_figures_ephys_confound.py
run $M1/03_gene_level_rheobase.py
run $M1/03b_module_is_it_subtype.py
run $M1/04_ephys_wide_screen.py
run $M1/05_glutamatergic_within_type.py
run $M1/06_per_pyramidal_class.py
run $MQ/07_dembrow_confirmatory.py
run $MQ/08_dembrow_prep_diagnostic.py
run $AT/11_atlas_extract_panels.py
run $AT/12_figure1_atlas.py
run $X/13_figure2_exvivo.py
run $AT/14_within_cluster_spread.py
run $AT/15_state_axis_function.py
run $X/16_two_stage_candidate_test.py
run $M1/17_bidirectional_test.py
run $X/18_l5pt_targeted_candidates.py
run $X/19_setlevel_crossclass.py
run $M1/20_layer_meta_analysis.py
run $X/21_breakdown_and_genomewide_pool.py
run $X/22_repairs_excitatory.py
run $M1/23_functional_categories_by_layer.py
run $M1/24_which_ephys_metric.py
run $X/25_layer_specificity_test.py
run $X/26_characterise_candidates.py
run $M1/27_composition_artifact_test.py
run $V1/28_v1_confirmatory.py
run $X/29_figure5_replication.py
run $X/30_index_vs_set_exploratory.py
run $X/31_ieg_independent_of_channels.py
run $M1/32_index_all_features_M1.py
run $AT/33_ltp_index_spread.py
run $M1/34_normalisation_comparison.py
run $M1/35_hknorm_panels_all_features.py
run $X/36_adaptation_three_cohorts.py
run $X/37_adaptation_population_control.py
run $X/38_asd_m1_v1.py
run $AT/39_asd_atlas_patterns.py

# ---- harmonised M1 -> V1 set test and its sensitivity analyses
run cross_dataset/m1_v1_validation/scripts/run_validation.py
run cross_dataset/m1_v1_validation/scripts/run_channel_receptor_exploration.py
run cross_dataset/m1_v1_validation/scripts/run_scaffolding_panel_exploration.py
run cross_dataset/ephys_spread/scripts/run_ephys_spread.py

# ---- whole-dataset AnnData pipeline (type vs state, cells vs nuclei, IBL, per-cluster, V1 interneurons)
if [[ ! -f ibl_neuropixels/data/ibl_motor_units.parquet || "${1:-}" == "--refetch-ibl" ]]; then
  run $IBL/ibl_extract_units.py all --workers 8 --comparison-scope all
fi
run $M1/01_build_patchseq.py
run $M1/02_patchseq_type_vs_state.py
run $M1/02b_ieg_coherence.py
run $AT/03_allen_cells_vs_nuclei.py
run $M1/04_joint_embedding.py
run $IBL/05_ibl_trait_vs_state.py
run $M1/06_l5_ieg_index.py
run $M1/06b_l5_it_robustness.py
run $M1/06_l5_ieg_index.py --layer 23
run $M1/06c_cross_layer_check.py
run $M1/07_per_cluster_slopes.py
run $M1/07b_pvalb_check.py
run $X/08_v1_label_transfer.py
run $V1/10_v1_pvalb_test.py
run $X/20_figs1-3_type_vs_state.py
run $X/21_fig4_in_vivo_context.py
run $M1/22_fig5_l5_pyramidal.py
run $M1/23_fig6_cross_layer.py
run $X/24_fig0_headline_replication.py
echo "done: results/ and figures/ in each analysis/<dataset>/ folder"
