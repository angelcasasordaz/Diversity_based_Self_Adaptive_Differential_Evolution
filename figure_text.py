"""Figure-visible translations; stored data, identities and filename tokens stay English."""
from matplotlib.text import Text
from matplotlib.axes import Axes
import re


SPANISH = {
    'Accuracy': 'Exactitud', 'Precision': 'Precisión', 'Recall': 'Sensibilidad',
    'F1-Score': 'F1-Score', 'Fitness': 'Aptitud', 'Features': 'Características',
    'Time': 'Tiempo', 'Selected features': 'Características seleccionadas',
    'Runtime': 'Tiempo de ejecución', 'Average runtime (s)': 'Tiempo promedio (s)',
    'Average runtime (sec)': 'Tiempo promedio (s)', 'No data': 'Sin datos',
    'Average selected features': 'Promedio de características seleccionadas',
    'Dataset': 'Conjunto de datos', 'Metaheuristics': 'Metaheurísticas',
    'Iteration': 'Iteración', 'Mean': 'Media', 'Median': 'Mediana',
    'Dataset mean': 'Media por conjunto', 'Feature\nefficiency': 'Eficiencia de\ncaracterísticas',
    'Final stage': 'Etapa final', 'No stored curves': 'Sin curvas en caché',
    'Value per dataset/run': 'Valor por conjunto/corrida',
    'Cached runs across datasets': 'Corridas en caché entre conjuntos',
    'Average rank (1 = best)': 'Rango promedio (1 = mejor)',
    'Holm-adjusted p (all algorithm pairs)': 'p ajustado por Holm (todos los pares de algoritmos)',
}


def visible_text(text, language):
    """Translate only the explicit vocabulary and label templates used by figures."""
    if language not in ('en', 'es'):
        raise ValueError(f'Unsupported figure language: {language}')
    if language == 'en':
        return text
    if text in SPANISH:
        return SPANISH[text]
    match = re.fullmatch(r'Holm-adjusted p; (\d+)-pair family', text)
    if match:
        return f'p ajustado por Holm; familia de {match[1]} pares'
    suffix = ' versus other algorithms (configured reference)'
    if text.endswith(suffix):
        return text[:-len(suffix)] + ' frente a otros algoritmos (referencia configurada)'
    suffix = ': cached run mean per matched block'
    if text.endswith(suffix):
        return visible_text(text[:-len(suffix)], language) + ': media de corridas en caché por bloque emparejado'
    for source, translated in SPANISH.items():
        for suffix, target in (
            (' (test)', ' (prueba)'),
            (' (test): cached runs across datasets', ' (prueba): Corridas en caché entre conjuntos'),
            (' (test): one cached run mean per dataset', ' (prueba): una media en caché por conjunto'),
            (' (0–1)', ' (0–1)'),
            (' (0–1): cached run mean', ' (0–1): media de corridas en caché'),
        ):
            if text == source + suffix:
                return translated + target
    # Heatmap titles identify the selected classifier as well as the metric.
    if ' — ' in text:
        classifier, label = text.split(' — ', 1)
        return classifier + ' — ' + visible_text(label, language)
    return text


def localize_figure(fig, language, *, protected=()):
    """Protect dataset and method names even if they happen to match vocabulary."""
    if language not in ('en', 'es'):
        raise ValueError(f'Unsupported figure language: {language}')
    protected = set(protected)
    # Text artists alone are regenerated from the tick formatter on draw.
    # Update explicit textual ticks as well, leaving numeric formatters intact.
    for ax in fig.findobj(match=Axes):
        for ticks, labels, setter in ((ax.get_xticks(), ax.get_xticklabels(), ax.set_xticks),
                                      (ax.get_yticks(), ax.get_yticklabels(), ax.set_yticks)):
            original = [label.get_text() for label in labels]
            translated = [text if text in protected else visible_text(text, language) for text in original]
            if translated != original:
                setter(ticks, translated)
    for artist in fig.findobj(match=Text):
        if artist.get_text() not in protected:
            artist.set_text(visible_text(artist.get_text(), language))
    return fig
