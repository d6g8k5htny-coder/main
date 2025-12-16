# Contributing to the Complexity Physics Framework

Thank you for your interest in contributing to this research project. This document outlines guidelines for contributing to the theoretical framework, empirical validation, and codebase.

## Table of Contents

- [Types of Contributions](#types-of-contributions)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Guidelines](#contribution-guidelines)
- [Pull Request Process](#pull-request-process)
- [Code Standards](#code-standards)
- [Scientific Standards](#scientific-standards)
- [Community Guidelines](#community-guidelines)

## Types of Contributions

We welcome contributions in several areas:

### Theoretical Extensions
- Mathematical derivations and proofs
- Alternative formulations of complexity functionals
- Connections to other theoretical frameworks
- Identification of additional falsifiable predictions

### Empirical Validation
- Analysis of new datasets (heliophysics, particle physics, gravitational waves)
- Statistical methodology improvements
- Uncertainty quantification refinements
- Independent verification of existing results

### Code Improvements
- Bug fixes and performance optimizations
- New validation tools and utilities
- Documentation improvements
- Test coverage expansion

### Documentation
- Clarifications of mathematical derivations
- Tutorial notebooks
- Literature review additions

## Getting Started

1. **Read the manuscript**: Familiarize yourself with the theoretical framework in `docs/manuscript.docx` or build it using `node build_full_manuscript.js`.

2. **Review existing issues**: Check if your contribution idea has already been discussed.

3. **Open a discussion**: For significant theoretical contributions, open an issue to discuss the approach before implementing.

## Development Setup

### Prerequisites

- Node.js 16+ (for manuscript building)
- Python 3.9+ (for validation code)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/dylanroy/complexity-physics-framework.git
cd complexity-physics-framework

# Install Python dependencies
pip install -e .

# Install Node.js dependencies (for manuscript building)
npm install docx
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_gauge_theory.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Building the Manuscript

```bash
# Build the manuscript document
node build_full_manuscript.js

# Or build the full version with all content
node body
```

## Contribution Guidelines

### For Theoretical Contributions

1. **Mathematical rigor**: All derivations must be complete and verifiable. State all assumptions explicitly.

2. **Notation consistency**: Use the notation established in the manuscript. See Appendix K (Symbol Index) for reference.

3. **Physical interpretation**: Connect mathematical results to physical observables where possible.

4. **Falsifiability**: If proposing new predictions, clearly state how they could be tested.

### For Empirical Contributions

1. **Data sources**: Use publicly available, peer-reviewed data sources. Document all data provenance.

2. **Error analysis**: Include proper uncertainty quantification with both statistical and systematic errors.

3. **Reproducibility**: All analysis must be reproducible from the provided code and data.

4. **Visualization**: Include clear figures with proper labels and uncertainties.

### For Code Contributions

1. **Follow existing patterns**: Match the coding style of existing modules.

2. **Write tests**: New functionality requires corresponding unit tests.

3. **Document functions**: Include docstrings with parameter descriptions and examples.

4. **Keep dependencies minimal**: Avoid adding new dependencies unless necessary.

## Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines above.

3. **Run tests** to ensure nothing is broken:
   ```bash
   python -m pytest tests/
   ```

4. **Update documentation** if your changes affect the API or methodology.

5. **Submit a pull request** with a clear description of:
   - What the change does
   - Why it's needed
   - How it was tested
   - Any relevant references

6. **Address review feedback** promptly.

## Code Standards

### Python

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable names

```python
def compute_complexity(
    gauge_group: str,
    representation: np.ndarray,
    lambda_param: float = 1.0
) -> float:
    """
    Compute the representation complexity K(G,R).

    Parameters
    ----------
    gauge_group : str
        Gauge group identifier (e.g., "SU3xSU2xU1")
    representation : np.ndarray
        Representation matrix
    lambda_param : float
        Coupling parameter (default: 1.0)

    Returns
    -------
    float
        Computed complexity value
    """
    ...
```

### JavaScript

- Use ES6+ features
- Prefer `const` over `let`
- Use descriptive function and variable names

## Scientific Standards

### Citation Requirements

- Cite original sources for all theoretical results
- Include DOIs where available
- Use consistent citation format (author-year)

### Data Integrity

- Never modify raw data files
- Document all data transformations
- Preserve original units and conventions

### Reproducibility Checklist

- [ ] Random seeds are set and documented
- [ ] All parameters are explicitly stated
- [ ] Data sources are cited with access dates
- [ ] Analysis code produces identical results on re-run

## Community Guidelines

### Code of Conduct

- Be respectful and constructive in all interactions
- Focus on scientific merit rather than personal opinions
- Acknowledge contributions appropriately

### Asking Questions

- Search existing issues and discussions first
- Provide context and specific details
- Include relevant code snippets or error messages

### Reporting Issues

When reporting bugs, include:
- Python/Node.js version
- Operating system
- Steps to reproduce
- Expected vs. actual behavior
- Relevant error messages

## Recognition

Contributors will be acknowledged in:
- The README.md contributors section
- Relevant publication acknowledgments (for significant contributions)
- Release notes

## Contact

For questions about contributing:
- Open a GitHub issue for technical questions
- Email the maintainer for collaboration inquiries

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
