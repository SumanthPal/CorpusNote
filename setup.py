from setuptools import setup, find_packages

setup(
    name="corpus-cli",
    version="1.0.0",
    description="A powerful CLI for indexing and chatting with documents using AI",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.0.0",
        "chromadb>=0.4.0",
        "google-generativeai>=0.3.0",
        "pypdf>=3.0.0",
        "python-docx>=1.0.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "psutil>=5.9.0",
        "watchdog>=3.0.0",
        # Add any other dependencies from your requirements.txt
    ],
    entry_points={
        "console_scripts": [
            "corpus=corpus.__main__:app",
        ],
    },
    python_requires=">=3.8",
    include_package_data=True,
)