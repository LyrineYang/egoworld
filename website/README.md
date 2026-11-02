# EgoWorld website prototype

A complete English research landing page, prepared for later publication. It uses semantic HTML, CSS, local SVG assets, and native expandable output details. There is no runtime JavaScript, external font request, analytics, account system, or model execution.

## Preview

From the repository root:

```bash
python3 website/build.py
python3 -m http.server 4173 --bind 127.0.0.1 --directory website/dist
```

Open `http://127.0.0.1:4173` on the same machine, or use your normal SSH port forwarding. You can also open the generated `website/dist/index.html` directly. Stop the server with Ctrl+C.

The build needs Python 3 and its standard library only. Generated `dist/` files are ignored; source SVGs live in `docs/assets/`. Relative asset paths support both a domain root and a GitHub Pages project subpath.

## Content and design

- `index.html`: page structure, English content, metadata, and repository/document links.
- `styles.css`: color/type/spacing tokens, desktop and mobile layouts, focus states, reduced motion, and print styles.
- `build.py`: assembles a portable static directory.
- [Design brief](../docs/website-design.md): reference analysis, visual rules, page map, and publication checklist.
- [Implementation ledger](../docs/en/status.md): source of truth for capability claims.

All conceptual illustrations are labeled. The website does not present invented performance numbers or pretend to execute inference. Output rows expand with mouse, touch, or keyboard. The processing diagram scrolls horizontally on small screens and opens at full size.

## Publish later

This task prepares the website; no hosting or domain is enabled. To use GitHub Pages later:

1. Enable Pages with GitHub Actions as the source in repository settings.
2. Copy `github-pages.yml.example` to `.github/workflows/pages.yml`.
3. Run the **Deploy project website** workflow manually from the intended branch.
4. Verify the returned deployment URL, links, desktop/mobile layout, and access settings.
5. Set the repository About website field and add canonical/Open Graph URL metadata once the real address exists.

Alternatively, upload the contents of `website/dist/` to any static host. Do not add a fake paper, benchmark, demo video, signup, or download link while the corresponding artifact is unavailable. Before publishing real inference examples, verify their provenance, permission, configuration, and limitations.
