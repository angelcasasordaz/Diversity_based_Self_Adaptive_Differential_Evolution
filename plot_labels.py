"""Manuscript display labels only; never use these as optimizer/result identities."""

PLOT_LABEL_OVERRIDES = {
    "MaCRO-DE-t": "DSA-DE",
    "MACRO-DE-T": "DSA-DE",
}


def plot_display_label(name, present_methods=()):
    """Keep the source method identifiable when real DSADE shares the figure."""
    name = str(name)
    label = PLOT_LABEL_OVERRIDES.get(name, name)
    if name.upper() == "MACRO-DE-T":
        label = "DSA-DE"
        if any(str(method).strip().upper() in {"DSADE", "DSA-DE", "DSA_DE"}
               for method in present_methods):
            label = "DSA-DE (MaCRO-DE-t)"
    return label


def report_display_label(label, present_labels=()):
    """Display compound report headings while preserving their stored keys."""
    method, separator, suffix = str(label).partition(" | ")
    methods = [str(item).partition(" | ")[0] for item in present_labels]
    return plot_display_label(method, methods) + separator + suffix
