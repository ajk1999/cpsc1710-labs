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

## Soup or Salad?

- **Name and purpose:** Soup or Salad is a food classifier that forces every food into one of two categories, showing how the predictiosn a model makes depend on the examples and features provided to that model.
- **How to use it:** Open [the classifier](https://ajk1999.github.io/cpsc1710-labs/). Choose a food from the dropdown to load preset scores, or you can adjust Spoon Necessity and Container Dependence from 1–10.  Read the prediction and neighbor distances below.  
- **How it predicts:** The classifier compares your two scores with eight labeled training foods, finds the three closest examples using straight-line distance, and predicts whichever label—Soup or Salad—gets the most votes.  Both scores count equally.
- **One limitation:** It has no “neither” option and only considers two features.  For example, cookie dough is predicted as Soup because its preset scores resemble soup examples, even though that label does not describe it well.

## development log

- I asked for a first version with two sliders—Spoon Necessity and Container Dependence—to classify foods as Soup or Salad.
- I replaced the fixed rule with eight labeled examples and asked for a three-nearest-neighbor vote, showing the selected examples and their distances.
- I asked for an autocomplete food picker that fills in preset slider values.
- I found the graph cluttered, so I asked to show only the selected food and its three nearest neighbors, remove the dataset buttons, and make the prediction section full width.
- I pointed out that the picker only offered training examples. I asked for different foods, including unlikely soups or salads, and a message rejecting training-set entries.
- I asked for more foods that unexpectedly predict Soup, then requested alphabetical ordering in the dropdown.
- I asked to pretty up the page with soup and salad line drawings, a matching verdict illustration, and a playful subtitle with sparkles around “believe.”
