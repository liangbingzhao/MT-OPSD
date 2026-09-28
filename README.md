# MT-OPSD: On-Policy Self-Distillation for Multi-Turn Image Editing

Project page for **MT-OPSD**, by Liangbing Zhao, Le Zhuo and Mohamed Elhoseiny.

The page lives on the `gh-pages` branch and is served by GitHub Pages at https://liangbingzhao.github.io/MT-OPSD/.

## Local preview

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Layout

```
index.html               page content
static/css/style.css     styles (light + dark themes)
static/js/main.js        turn explorers, charts, tables (result numbers live here)
static/images/           figures converted from the paper PDFs
static/images/turns/     per-turn tiles for the interactive explorers
```

## TODO before release

- Replace the placeholder `href="#"` links marked `data-soon` in `index.html` (Paper / arXiv) and remove their `soon` tags.
- Update the BibTeX entry once the arXiv ID is available.
