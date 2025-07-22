# setup.py
from setuptools import setup, find_packages

setup(
    name="cds",                           # your package name
    version="0.1.0",
    packages=find_packages(where="src"),  # look for modules under src/
    package_dir={"": "src"},              # map root package to src/
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "openai",
        "jinja2",
        "markdown",
        "pandas",
        "numpy",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "cds-cli=main:main",          # optional CLI entrypoint
        ],
    },
)
