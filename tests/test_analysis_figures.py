import os
import pytest
from analysis.common import get_out_dir
from analysis.figures import generate_all_figures


def test_figures_generation():
    out_dir = get_out_dir()
    expected_pdfs = [
        "fig1_dose_response.pdf",
        "fig2_effective_sigma.pdf",
        "fig3_tradeoff.pdf",
        "fig4_transitions.pdf",
    ]
    for pdf_name in expected_pdfs:
        p = os.path.join(out_dir, pdf_name)
        assert os.path.exists(p), f"Missing figure {pdf_name}"
        assert os.path.getsize(p) > 1000, f"Figure {pdf_name} is too small"
