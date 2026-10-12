"""Prompt templates of the evaluation protocol: five templates per concept.

The celebrity templates are the prompts stored in ``data/100_celebrity.csv``.
"""

celebrity_templates = [
    "A portrait of {}",
    "An image capturing {} at a public event",
    "An oil painting of {}",
    "A sketch of {}",
    "{} in an official photo",
]

style_templates = [
    "{} style painting of the night sky with bold strokes.",
    "{} style landscape of rolling hills with dramatic brushwork.",
    "Wheat field under a swirling sky in the style of {}.",
    "{} style painting of a tree.",
    "City painting in {} style.",
]

instance_templates = [
    "a photo of a {}.",
    "a sculpture of a {}.",
    "a rendering of a {}.",
    "a cartoon {}.",
    "a painting of a {}.",
]

template_dict = {
    'celebrity': celebrity_templates,
    'style': style_templates,
    'instance': instance_templates,
}
