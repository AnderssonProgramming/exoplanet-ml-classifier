# Commit Convention

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification.

## Format

```
<type>(<scope>): <short description>

[optional body]

[optional footer(s)]
```

The **type** and **short description** are mandatory. The scope is recommended for this
project to identify which module or notebook the change affects.

---

## Allowed Types

| Type       | When to use                                                                 |
|------------|-----------------------------------------------------------------------------|
| `feat`     | A new feature or capability is added to the project                        |
| `fix`      | A bug, incorrect metric, or broken pipeline step is corrected              |
| `docs`     | Documentation changes only (README, docstrings, Markdown cells)            |
| `chore`    | Build system, dependency, or repo-maintenance changes                      |
| `refactor` | Code is restructured without changing observable behaviour                 |
| `test`     | New or updated test cases; no production code changes                      |
| `style`    | Formatting, whitespace, flake8 fixes — no logic changes                    |

---

## Domain-Specific Examples

```
feat(models): add XGBoost classifier with tuned hyperparameter grid
```
```
fix(preprocessing): correct SMOTE threshold comparison from >= to <
```
```
docs(notebooks): add EDA insights Markdown section to 01_eda.ipynb
```
```
chore(deps): pin imbalanced-learn to 0.12.x for SMOTE API stability
```
```
refactor(evaluation): extract _compute_f1_from_pr into private helper
```
```
test(data_loader): add edge-case test for empty target column in describe_dataset
```
```
style(src): apply flake8 E501 line-length fixes across all modules
```

---

## Rules

1. Use the **imperative mood** in the short description: "add", "fix", "remove" — not "added" or "fixing".
2. Keep the short description under **72 characters**.
3. Reference a GitHub issue in the footer when applicable: `Closes #12`.
4. Breaking changes must include `BREAKING CHANGE:` in the footer.
5. Scope values for this project: `models`, `preprocessing`, `data_loader`,
   `evaluation`, `visualization`, `constants`, `notebooks`, `tests`, `ci`, `docs`.
