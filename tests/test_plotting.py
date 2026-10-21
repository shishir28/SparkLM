import pytest


matplotlib = pytest.importorskip("matplotlib")

from sparklm.plotting import plot_losses


def test_plot_losses_creates_missing_parent_directory(tmp_path):
    output_path = tmp_path / "nested" / "losses.png"

    plot_losses(
        epochs_seen=[0, 1],
        tokens_seen=[2, 4],
        train_losses=[2.0, 1.0],
        val_losses=[2.2, 1.5],
        path=output_path,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
