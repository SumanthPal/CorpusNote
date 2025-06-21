# First time setup
./setup.sh
/Users/sumanth/Downloads/OAE-Spec-Full
# Index your documents
python main.py index ~/Documents/Research
python main.py index ~/Documents/Research --pattern "*atomic*.pdf"

# Check status
python main.py status
python main.py list --sort size --limit 10

# Start chatting
python main.py chat
python main.py chat --filter "atomic"

# Ask one-off questions
python main.py ask "What are the main protocols discussed?"
python main.py ask "Explain atomic ethernet" --filter "*.pdf"

# Update with new files
python main.py update ~/Documents/Research

python main.py update '/Users/sumanth/Library/Mobile Documents/com~apple~CloudDocs/MULLIGANSTEW'                                                                            ─╯
# Watch for changes
python main.py watch ~/Documents/Research --interval 300

# Analyze before indexing
python main.py analyze ~/Documents/NewFolder

# Export chat history
python main.py export --output my_research_notes.md

# Maintenance
python main.py remove "old_paper.pdf"
python main.py clear --yes


