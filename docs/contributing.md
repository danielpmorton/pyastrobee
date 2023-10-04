# Contributing Guide

Contributions are welcomed! If there is something you'd like to add, open a [pull request](https://github.com/danielpmorton/pyastrobee/pulls) for your development branch. 

Note that pushing directly to the `master` branch is not allowed without a pull request approval.

Before creating the PR, ensure that you:

- Run Pylint/Pylance
  - Warnings should be kept to a minimum, with some exceptions allowable. In general if Pylint rates the code over 9/10 it should be in good shape
- Run Black to format the code
- Create Google-style docstrings for any new functions, classes, and files. (see autoDocstring extension)

If you're using VSCode, see the recommended installations in `.vscode/extensions.json`. The most important extensions to contributing are Pylint, Pylance, Black, and autoDocstring. Autoformatting on save with Black should automatically be enabled via `.vscode/settings.json`
