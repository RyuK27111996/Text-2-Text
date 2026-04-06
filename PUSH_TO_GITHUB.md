# Push Changes to GitHub

## Prerequisites

1. **Git installed** on your system
2. **GitHub account** and repository created
3. **Git configured** with your credentials (or SSH key)

## Steps to push changes (PowerShell)

```powershell
# 1. Navigate to the repo
cd D:\Text-2-Text\Text-2-Text

# 2. Check git status
git status

# 3. Stage all changes
git add .

# 4. Commit with a descriptive message
git commit -m "Add Ollama support, improve docs, add concurrency guidance

- Add OllamaService for local gemma3 model
- Update README with Ollama setup, concurrency tuning, examples
- Add troubleshooting section for Windows socket errors
- Improve response parsing for NDJSON streaming
- Document 15-concurrent-user capability and semaphore tuning"

# 5. Push to main branch
git push origin main

# If you get "permission denied", use one of these:
# a) SSH key setup (one-time):
#    git remote set-url origin git@github.com:RyuK27111996/Text-2-Text.git
#    git push origin main
#
# b) Personal access token (if using HTTPS):
#    git push origin main
#    (Enter token when prompted)
```

## Verify push succeeded

```powershell
# Check the remote
git remote -v

# View recent commits
git log --oneline -5

# View on GitHub
# Open: https://github.com/RyuK27111996/Text-2-Text
```

## If you need to undo the last commit (before pushing)

```powershell
# Undo last commit but keep changes
git reset --soft HEAD~1

# Undo last commit and discard changes
git reset --hard HEAD~1
```

## If push is rejected (remote has newer changes)

```powershell
# Pull latest changes first
git pull origin main

# Resolve any conflicts if needed, then push
git push origin main
```

---

**Summary of changes being pushed:**
- ✅ Updated `README.md` with Ollama docs, concurrency guidance, examples, troubleshooting
- ✅ `app/services/ollama.py` – OllamaService with NDJSON streaming support
- ✅ `app/config.py` – Ollama settings (base_url, default_model)
- ✅ `app/main.py` – Dual backend selection (Ollama vs Gemini)
- ✅ `app/schemas.py` – Model validation
