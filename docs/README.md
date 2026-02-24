# BORES Documentation

This directory contains the MkDocs documentation for the BORES framework.

## Building Locally

To build and preview the documentation locally:

```bash
# Install MkDocs and dependencies
uv pip install --system mkdocs-material mkdocs-minify-plugin pymdown-extensions

# Serve locally (with live reload)
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

## Building for Production

```bash
# Build static site
mkdocs build

# Output will be in site/ directory
```

## Deployment

Documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch via the GitHub Actions workflow in `.github/workflows/docs.yml`.

## Structure

- `index.md` - Home page
- `getting-started/` - Installation, quickstart, core concepts
- `tutorials/` - Step-by-step learning tutorials
- `guides/` - Comprehensive user guides
- `advanced/` - Advanced topics
- `examples/` - Complete working examples
- `best-practices/` - Best practices and tips
- `reference/` - API reference and glossary
- `stylesheets/` - Custom CSS
- `javascripts/` - Custom JS (MathJax config)

## Adding Content

1. Create `.md` files in appropriate directories
2. Add entries to `mkdocs.yml` nav section
3. Use Material for MkDocs features (admonitions, tabs, cards, etc.)
4. Test locally with `mkdocs serve`
5. Commit and push to deploy

## Links

- [Material for MkDocs Docs](https://squidfunk.github.io/mkdocs-material/)
- [MkDocs Docs](https://www.mkdocs.org/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)
