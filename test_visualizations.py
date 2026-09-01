import argparse
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import main_best as study


class VisualizationTests(unittest.TestCase):
    @staticmethod
    def _capture_chart(function, *args):
        captured = {}

        def capture(fig, out_dir, filename):
            captured["fig"] = fig
            captured["filename"] = filename

        with patch.object(study, "_save_chart", side_effect=capture):
            result = function(*args)
        return result, captured["fig"], captured["filename"]

    def test_ablation_labels_are_inside_axes_and_do_not_overlap(self):
        optimizer_order, display_labels, _ = study._ablation_figure_metadata()
        rows = []
        for estimator_idx, estimator in enumerate(("knn", "svm", "rf")):
            for optimizer_idx, optimizer in enumerate(optimizer_order):
                rows.append(
                    {
                        "Dataset": "Synthetic",
                        "Estimator": estimator,
                        "Optimizer": optimizer,
                        "N_Features_Selected": 5.0 + 0.16 * optimizer_idx,
                        "AS_test": 0.80 + 0.008 * optimizer_idx + 0.002 * estimator_idx,
                    }
                )
        summary = pd.DataFrame(rows)
        original = summary.copy(deep=True)

        with tempfile.TemporaryDirectory() as tmp:
            filename, fig, captured_filename = self._capture_chart(
                study.generate_ablation_accuracy_features_tradeoff,
                "Synthetic",
                summary,
                tmp,
            )
        self.assertEqual(filename, captured_filename)
        pd.testing.assert_frame_equal(summary, original)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_names = set(display_labels.values())
        for ax in fig.axes:
            axes_bbox = ax.get_window_extent(renderer)
            labels = [text for text in ax.texts if text.get_text() in label_names]
            self.assertEqual(len(labels), len(optimizer_order))
            label_boxes = [text.get_window_extent(renderer) for text in labels]
            for label_box in label_boxes:
                self.assertGreaterEqual(label_box.x0, axes_bbox.x0 - 1.0)
                self.assertLessEqual(label_box.x1, axes_bbox.x1 + 1.0)
                self.assertGreaterEqual(label_box.y0, axes_bbox.y0 - 1.0)
                self.assertLessEqual(label_box.y1, axes_bbox.y1 + 1.0)
            for first_idx, first in enumerate(label_boxes):
                for second_idx, second in enumerate(
                    label_boxes[first_idx + 1:],
                    start=first_idx + 1,
                ):
                    self.assertEqual(
                        study._bbox_overlap_area(first, second),
                        0.0,
                        msg=(
                            labels[first_idx].get_text(),
                            labels[first_idx].get_position(),
                            labels[second_idx].get_text(),
                            labels[second_idx].get_position(),
                        ),
                    )
            self.assertTrue(any("Preferred region" in text.get_text() for text in ax.texts))
        plt.close(fig)

    def test_sensitivity_uses_adjacent_dual_axis_bars_for_every_parameter(self):
        parameters = (
            "beta_min",
            "beta_max",
            "pcr",
            "mahalanobis_q",
            "pop_size",
            "epochs",
        )
        sensitivity_values = [0.1, 0.2, 0.3, 0.4]
        rows = []
        for estimator_idx, estimator in enumerate(("knn", "svm", "rf")):
            for value_idx, value in enumerate(sensitivity_values):
                rows.append(
                    {
                        "Estimator": estimator,
                        "Optimizer": "DSADE",
                        "SensitivityValue": value,
                        "F1_test": 0.72 + 0.03 * value_idx + 0.005 * estimator_idx,
                        "N_Features_Selected": 4.0 + value_idx + 0.2 * estimator_idx,
                    }
                )
        source = pd.DataFrame(rows)
        original = source.copy(deep=True)

        for parameter in parameters:
            with self.subTest(parameter=parameter), tempfile.TemporaryDirectory() as tmp:
                args = argparse.Namespace(
                    sensitivity_values=list(sensitivity_values),
                    optimizers=["DSA-DE"],
                    sensitivity_parameter=parameter,
                )
                filename, fig, captured_filename = self._capture_chart(
                    study.generate_sensitivity_main_figure,
                    source,
                    tmp,
                    args,
                )
                self.assertEqual(filename, captured_filename)
                self.assertIn(parameter, filename)
                fig.canvas.draw()

                left_axes = [
                    ax for ax in fig.axes if ax.get_ylabel() == "Mean F1-score"
                ]
                right_axes = [
                    ax for ax in fig.axes if ax.get_ylabel() == "Mean selected features"
                ]
                self.assertEqual(len(left_axes), 3)
                self.assertEqual(len(right_axes), 3)
                for left_ax, right_ax in zip(left_axes, right_axes):
                    self.assertEqual(len(left_ax.patches), len(sensitivity_values))
                    self.assertEqual(len(right_ax.patches), len(sensitivity_values))
                    self.assertFalse(left_ax.lines)
                    self.assertFalse(right_ax.lines)
                    f1_centers = np.array(
                        [bar.get_x() + bar.get_width() / 2 for bar in left_ax.patches]
                    )
                    feature_centers = np.array(
                        [bar.get_x() + bar.get_width() / 2 for bar in right_ax.patches]
                    )
                    self.assertTrue(np.all(feature_centers > f1_centers))
                    self.assertTrue(
                        np.allclose(
                            feature_centers - f1_centers,
                            [bar.get_width() for bar in left_ax.patches],
                        )
                    )
                    for f1_bar, feature_bar in zip(left_ax.patches, right_ax.patches):
                        self.assertEqual(feature_bar.get_hatch(), "///")
                        self.assertTrue(
                            np.allclose(
                                f1_bar.get_facecolor()[:3],
                                feature_bar.get_edgecolor()[:3],
                                atol=1e-7,
                            )
                        )
                    self.assertGreaterEqual(len(left_ax.texts), len(sensitivity_values))
                    self.assertGreaterEqual(len(right_ax.texts), len(sensitivity_values))
                    self.assertTrue(
                        all(left_ax.get_ylim()[0] <= text.get_position()[1] <= left_ax.get_ylim()[1]
                            for text in left_ax.texts)
                    )
                    self.assertTrue(
                        all(right_ax.get_ylim()[0] <= text.get_position()[1] <= right_ax.get_ylim()[1]
                            for text in right_ax.texts)
                    )

                legend_labels = {
                    text.get_text()
                    for legend in fig.legends
                    for text in legend.get_texts()
                }
                self.assertIn("Mean F1-score", legend_labels)
                self.assertIn("Mean selected features", legend_labels)
                plt.close(fig)

        pd.testing.assert_frame_equal(source, original)


if __name__ == "__main__":
    unittest.main()
