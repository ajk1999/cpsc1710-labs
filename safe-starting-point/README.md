# CPSC 1710 Labs

Student-facing assignments and starter materials for CPSC 1710, Fall 2026.

## Labs

- [Lab 1: Meet a deep-learning notebook](lab-01/)
- [Lab 2: From one pixel to your classifier](lab-02/)

Use the [live course hub](https://xiuyechen.github.io/cpsc1710-labs/) for the simplest experience. Each assignment is available as a webpage and a printable PDF.

## Opening the files

The HTML files have no build step. After cloning, open `index.html` directly in Chrome, Safari, or Firefox. Avoid VS Code's **Open Preview** for these files: its internal `file+.vscode-resource` links do not work as normal browser addresses.

You can also serve the repository locally:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000/`.

## Credits

Materials are by [Xiuye Chen](https://github.com/xiuyechen), developed with Codex, and shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## development log

- I asked for a first version with two sliders—Spoon Necessity and Container Dependence—to classify foods as Soup or Salad.
- I replaced the fixed rule with eight labeled examples and asked for a three-nearest-neighbor vote, showing the selected examples and their distances.
- I asked for an autocomplete food picker that fills in preset slider values.
- I found the graph cluttered, so I asked to show only the selected food and its three nearest neighbors, remove the dataset buttons, and make the prediction section full width.
- I pointed out that the picker only offered training examples. I asked for different foods, including unlikely soups or salads, and a message rejecting training-set entries.
- I asked for more foods that unexpectedly predict Soup, then requested alphabetical ordering in the dropdown.
- I asked to pretty up the page with soup and salad line drawings, a matching verdict illustration, and a playful subtitle with sparkles around “believe.”
