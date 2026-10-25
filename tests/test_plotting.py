import pytest
import torch


matplotlib = pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt  # noqa: E402

from sparklm import plotting  # noqa: E402
from sparklm.plotting import plot_losses, plot_values  # noqa: E402


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


def call_plot_values(path, num_points=3, num_epochs=5, examples_seen=40):
    plot_values(
        epochs_seen=torch.linspace(0, num_epochs, num_points),
        examples_seen=torch.linspace(0, examples_seen, num_points),
        train_values=[0.9, 0.5, 0.2][:num_points],
        val_values=[1.0, 0.6, 0.3][:num_points],
        label="loss",
        path=path,
    )


@pytest.fixture
def captured_figure(monkeypatch):
    figures = []
    original_close = plotting.plt.close

    def capture_then_close(*args, **kwargs):
        figures.append(plt.gcf())
        original_close(*args, **kwargs)

    monkeypatch.setattr(plotting.plt, "close", capture_then_close)
    return figures


def test_plot_values_creates_missing_parent_directory(tmp_path):
    output_path = tmp_path / "nested" / "spam_classifier_loss.png"

    call_plot_values(output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_plot_values_uses_epochs_bottom_and_examples_top(tmp_path, captured_figure):
    call_plot_values(tmp_path / "plot.png", num_epochs=5, examples_seen=40)

    epochs_axis, examples_axis = captured_figure[0].axes
    assert epochs_axis.get_xlabel() == "Epochs"
    assert examples_axis.get_xlabel() == "Examples Seen"
    assert epochs_axis.lines[0].get_xdata()[-1] == pytest.approx(5.0)
    assert examples_axis.lines[0].get_xdata()[-1] == pytest.approx(40.0)


def test_plot_values_labels_series_from_label(tmp_path, captured_figure):
    call_plot_values(tmp_path / "plot.png")

    epochs_axis = captured_figure[0].axes[0]
    legend_texts = [text.get_text() for text in epochs_axis.get_legend().get_texts()]
    assert epochs_axis.get_ylabel() == "Loss"
    assert legend_texts == ["Train Loss", "Validation Loss"]
    assert list(epochs_axis.lines[0].get_ydata()) == [0.9, 0.5, 0.2]
    assert list(epochs_axis.lines[1].get_ydata()) == [1.0, 0.6, 0.3]


def test_plot_values_closes_its_figure(tmp_path):
    plt.close("all")

    call_plot_values(tmp_path / "plot.png")

    assert plt.get_fignums() == []


def test_plot_values_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        plot_values(
            epochs_seen=torch.linspace(0, 5, 3),
            examples_seen=torch.linspace(0, 40, 3),
            train_values=[0.9, 0.5],
            val_values=[1.0, 0.6],
            label="loss",
            path=tmp_path / "plot.png",
        )
    plt.close("all")
