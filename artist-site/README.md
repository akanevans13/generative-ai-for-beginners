# Collage/AI — Digital Artist Portfolio

A minimal static site for a digital collage and AI film artist.

## Quick start

1. Open `index.html` in a browser, or run a static server:
   - Python 3: `python3 -m http.server 8080`
   - Node: `npx serve -l 8080`
2. Visit `http://localhost:8080`.

## Customize

- Branding
  - Update title and description in `<head>` of `index.html`.
  - Replace `assets/img/favicon.png`.
- Hero
  - Replace background texture `assets/img/hero-noise.png`.
- Works
  - Put thumbnails in `assets/img/work-*.jpg` and large files in `assets/img/work-*-large.jpg`.
  - Duplicate an `<a class="gallery-item">` to add more.
- Films
  - Add posters to `assets/img/film-*.jpg` and videos to `assets/video/*.mp4`.
  - Duplicate a `.film-card` block.
- About
  - Edit copy in the `#about` section, replace `assets/img/about.jpg`.
- Contact
  - Replace `action` in the form with your endpoint (e.g., Formspree or email Lambda).

## Assets

- Place images under `assets/img/` and videos under `assets/video/`.
- Recommended thumbnail sizes: 800–1200px wide JPGs; posters 1280×720.

## Deploy

- GitHub Pages: push and enable Pages on the repository.
- Netlify/Vercel: drag-and-drop the folder or connect the repo (no build step).

## License

MIT — use freely, please keep credits if helpful.