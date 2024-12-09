from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="web-indexer",
    version="0.1.0",
    author="Rajath",
    author_email="rajath@docrag.io",
    description="A web indexing and RAG system built with LangChain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rajath-com/web-indexer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.1.0",
        "crawl4ai>=0.3.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-dotenv>=0.19.0",
        "chromadb>=0.4.0",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=1.0",
            "ruff>=0.1.0",
        ],
    }
) 