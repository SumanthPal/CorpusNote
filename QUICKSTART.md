# Corpus CLI - Quick Start Guide 🚀

Welcome to Corpus! This guide will get you up and running in minutes.

## 1. Installation (2 minutes)

### Option A: Automatic Installation (Recommended)

```bash
git clone https://github.com/yourusername/corpus-cli.git
cd corpus-cli
./install.sh  # or install.ps1 on Windows
```

### Option B: Using Make

```bash
git clone https://github.com/yourusername/corpus-cli.git
cd corpus-cli
make install
```

## 2. Configuration (1 minute)

The installer will prompt you to configure. If you skipped it:

```bash
# Activate virtual environment first
source env/bin/activate  # or env\Scripts\activate on Windows

# Run configuration wizard
corpus config setup
```

You'll need:
- A Google Gemini API key (get one at https://makersuite.google.com/app/apikey)

## 3. Your First Index (2 minutes)

Let's index some documents:

```bash
# Index a single PDF
corpus index ~/Documents/research-paper.pdf

# Index a folder
corpus index ~/Documents/Research

# Index Python code
corpus index ~/Projects/my-app --pattern "*.py"
```

## 4. Chat with Your Documents (∞ minutes)

Now the fun part - talk to your documents:

```bash
# Start interactive chat
corpus chat

# Ask a specific question
corpus ask "What are the main findings in the research?"

# Chat with specific documents
corpus chat --filter "research*.pdf"
```

## 5. Generate Diagrams (Optional)

If you have D2 installed:

```bash
# Generate from description
corpus diagram "user authentication flow"

# Generate from your documents
corpus diagram --search "system architecture"
```

## Common Commands

```bash
corpus status          # Check what's indexed
corpus list            # List all documents
corpus update ~/Docs   # Update with new files
corpus clear           # Start fresh
```

## Tips for Best Results

1. **Index related documents together** - This gives better context for answers
2. **Use descriptive filenames** - Makes filtering easier
3. **Ask specific questions** - "What does section 3.2 say about X?" works better than "tell me about X"

## Next Steps

- Read the full documentation: `corpus --help`
- Explore configuration options: `corpus config show`
- Set up document watching: `corpus watch ~/Documents`

## Need Help?

- Run `corpus --help` for command details
- Check configuration: `corpus config validate`
- View system info: `corpus info`

Happy researching! 🎉