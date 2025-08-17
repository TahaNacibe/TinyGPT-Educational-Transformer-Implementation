from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tinygpt",
    version="0.1.0",
    author="TinyGPT Contributors",
    author_email="",
    description="A minimal GPT implementation for educational purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TinyGPT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "tinygpt-train=tiny_gpt:main",
            "tinygpt-chat=examples.interactive_chat:interactive_chat",
        ],
    },
    keywords="gpt, transformer, nlp, education, pytorch, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/TinyGPT/issues",
        "Source": "https://github.com/yourusername/TinyGPT",
        "Documentation": "https://github.com/yourusername/TinyGPT#readme",
    },
)
