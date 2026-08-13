"""PCA neural-trajectory tutorials and reproducible analysis helpers."""

from .artifacts import load_artifacts, save_artifacts
from .contrasts import (
    contrast_fold_metrics,
    contrast_label,
    contrast_slug,
    parse_contrast,
)
from .eegbci import (
    ALL_CONDITIONS,
    BIDS_NAMES,
    DEFAULT_CONDITIONS,
    LABEL_INFO,
    LABEL_NAMES,
    format_pair_keys,
    load_eegbci_container,
    setup_data_bids,
)
from .provenance import build_manifest, write_manifest
from .spectral import (
    SPECTRAL_BANDS,
    hilbert_log_amplitude_envelope,
)
from .viz import facet_figures, overlay_figures
from .wakeman_henson import (
    MEG_SENSOR_SETS,
    load_wakeman_henson,
    preprocessing_config,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_CONDITIONS",
    "BIDS_NAMES",
    "DEFAULT_CONDITIONS",
    "LABEL_INFO",
    "LABEL_NAMES",
    "MEG_SENSOR_SETS",
    "SPECTRAL_BANDS",
    "build_manifest",
    "contrast_fold_metrics",
    "contrast_label",
    "contrast_slug",
    "facet_figures",
    "format_pair_keys",
    "hilbert_log_amplitude_envelope",
    "load_artifacts",
    "load_eegbci_container",
    "load_wakeman_henson",
    "overlay_figures",
    "parse_contrast",
    "save_artifacts",
    "setup_data_bids",
    "write_manifest",
    "preprocessing_config",
]
