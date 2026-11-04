# EgoWorld website prototype

An English research landing page and benchmark report built from semantic HTML, CSS, local SVG assets and expandable output details.

## Preview

From the repository root:

```bash
python3 website/build.py
python3 -m http.server 4173 --bind 127.0.0.1 --directory website/dist
```

Open `http://127.0.0.1:4173` on the same machine, or use your normal SSH port forwarding. You can also open the generated `website/dist/index.html` directly. Follow **Benchmarks** for six figures, the 45-run table, and CSV and JSON downloads. Stop the server with Ctrl+C.

The build needs Python 3 and its standard library only. Generated `dist/` files are ignored; source SVGs live in `docs/assets/`. Relative asset paths support both a domain root and a GitHub Pages project subpath.

## Content and design

- `index.html`: page structure, English content, metadata, and repository/document links.
- `benchmarks.html`: benchmark narrative and accessible figure captions. The build fills the trial table from the audited JSON.
- `styles.css`: color/type/spacing tokens, desktop and mobile layouts, focus states, reduced motion, and print styles.
- `build.py`: assembles a portable static directory.
- [Design brief](../docs/website-design.md): reference analysis, visual rules, page map, and publication checklist.
- [Implementation ledger](../docs/en/status.md): source of truth for capability claims.

The benchmark charts use the checked-in formal and diagnostic summaries. The native-product and fixed-model tracks have separate figures. Charts open as full-size SVGs, and wider figures and tables scroll on small screens. [Analysis and source data](../docs/efficiency-convergence.md)

To regenerate the SVG and CSV assets, run the following with Matplotlib installed in your Python environment, then run the static build again.

```bash
python3 egoworld/experiments/convergence_figures.py
python3 website/build.py
```

## Publish later

This task prepares the website; no hosting or domain is enabled. To use GitHub Pages later:

1. Enable Pages with GitHub Actions as the source in repository settings.
2. Copy `github-pages.yml.example` to `.github/workflows/pages.yml`.
3. Run the **Deploy project website** workflow manually from the intended branch.
4. Verify the returned deployment URL, links, desktop/mobile layout, and access settings.
5. Set the repository About website field and add canonical/Open Graph URL metadata once the real address exists.

Alternatively, upload the contents of `website/dist/` to any static host. Do not add a fake paper, benchmark, demo video, signup, or download link while the corresponding artifact is unavailable. Before publishing real inference examples, verify their provenance, permission, configuration, and limitations.
