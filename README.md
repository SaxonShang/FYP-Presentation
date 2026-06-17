# Imperial College London LaTeX templates

This is the official Beamer template for Imperial
College London. Letter and poster templates are also available.

## FYP presentation deck

The project presentation source is `fyp_presentation.tex`. It uses the Imperial
Beamer theme and should be compiled with XeLaTeX.

VS Code LaTeX Workshop is configured in `.vscode/settings.json`; open this
folder in VS Code and run `LaTeX Workshop: Build LaTeX project`, or run:

```bash
./build_presentation.sh
```

If the build command reports that `latexmk` is missing, install a TeX Live
distribution with XeLaTeX support, for example:

```bash
sudo apt install latexmk texlive-xetex texlive-latex-extra texlive-fonts-recommended
```

## Issues

Please report any bugs in the templates on GitHub (https://github.com/ImperialCollegeLondon/imperial_latex_templates/issues).

## Copyright

© Imperial College London, 2024. These templates, including logo and fonts, are 
for use of Imperial staff and students only for university business. All rights 
reserved to the copyright owners.
