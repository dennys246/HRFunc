# Contributing to your-tool

First off, thanks for taking the time to contribute! 🎉  
This document outlines the process and guidelines for contributing to **HRfunc**.

---

## 🛠 Ways to Contribute
- Report bugs
- Suggest new features or improvements
- Improve documentation
- Submit pull requests with bug fixes or new features
- Share examples or tutorials

---

## ⚙️ Development Setup
1. Fork the repo and clone your fork:
   ```bash
   git clone https://github.com/dennys246/hrfunc.git
   cd your-tool
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -e .[dev]
   ```

3. Run the test suite to make sure everything works:
   ```bash
   pytest
   ```

---

## 📋 Code Style
- Follow [PEP8](https://peps.python.org/pep-0008/) conventions.
- Use [black](https://black.readthedocs.io/) for formatting:
  ```bash
  black src/ tests/
  ```
- Use [isort](https://pycqa.github.io/isort/) for import ordering:
  ```bash
  isort src/ tests/
  ```

---

## 🧪 Testing
Write unit tests for new functionality in the `tests/` folder.

Ensure existing tests pass before submitting:
  ```bash
  pytest --maxfail=1
  ```

Key current tests:
- tests/test_estimation.py 
    -> Assesses ability to estimate HRF's from a RPC experiment
    -> Tests all of the above in all fNIRS formats
- tests/test_localization.py
    -> Ensure's nearest neighbor function finds true matches
    -> Ensure's nearest neighbor function doesn't find edge cases
    -> Tests all of the above in all fNIRS formats
- tests/test_hashing.py
    -> (In Progress) Ensure's hashing table find's true contexts
    -> (In Progress) Ensure's hashing table doesn't find edge cases
---

## 🔀 Pull Request Process
1. Create a new branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. Commit changes with clear messages:
   ```bash
   git commit -m "Add feature: my new feature"
   ```
3. Push to your fork and open a Pull Request.
4. Describe the change, why it’s useful, and reference related issues.

---

## 📖 Documentation
- Update `README.md` if your changes affect usage.

---

## 🤝 Community Guidelines
- Be respectful and constructive.
- Ask questions — contributions aren’t just code!
- Check open issues before creating new ones.

---

## 📜 License
By contributing, you agree that your contributions will be licensed under the [BSD-3 License](LICENSE).
