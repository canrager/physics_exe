# physics.exe Hackathon Codebase

## Quick Start

```bash
# 1. Install UV (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# On Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone this repo
git clone https://github.com/canrager/physics_exe.git
cd physics_exe

# 3. Run the starter script (this installs everything automatically)
uv run main.py
```

You should see: `Plot saved to exponential.png -- your repo setup is working!`

## What is UV?

UV is a tool that manages Python and its packages for you. Think of it like an app store for Python libraries — you tell it what you need and it downloads and installs everything automatically.

You never have to worry about "virtual environments" or "pip install" when using UV. The single command `uv run main.py` will:

1. Create an isolated environment for this project (if one doesn't exist yet).
2. Install every dependency listed in `pyproject.toml`.
3. Run the script.

### Installing UV

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Setup

```bash
# 1. Clone this repo (replace <repo-url> with the actual URL from GitHub)
git clone <repo-url>
cd physics_exe

# 2. Run the starter script
uv run main.py
```

You should see: `Plot saved to exponential.png -- your repo setup is working!`

If you see that message, everything is installed and working.

## Git in 5 Minutes

Git tracks changes to your files so you can undo mistakes and collaborate with others. Here are the commands you need:

| Command | What it does |
|---|---|
| `git status` | Shows which files you changed since your last save point. **Run this often.** |
| `git add <file>` | Stages a file — marks it to be included in your next save point. Use `git add .` to stage everything. |
| `git commit -m "message"` | Creates a save point (called a "commit") with a short description of what you changed. |
| `git push` | Uploads your commits to GitHub so others can see them. |
| `git pull` | Downloads the latest commits from GitHub that others pushed. |

A typical workflow looks like this:

```bash
# 1. Check what changed
git status

# 2. Stage your changes
git add main.py

# 3. Commit with a message
git commit -m "add exponential plot"

# 4. Push to GitHub
git push
```

### Branches

A branch lets you work on something without affecting the main code. This is useful when multiple people work on the same repo.

```bash
# Create a new branch and switch to it
git checkout -b my-feature

# ... make changes, add, commit as usual ...

# Switch back to main
git checkout main
```

When your feature is ready, push your branch and open a **Pull Request** on GitHub to merge it into main.

## Files in This Repo

- `main.py`: Starter script that plots an exponential function and saves it as a PNG.
- `pyproject.toml`: Project configuration — lists the Python version and dependencies.
- `uv.lock`: Auto-generated lock file that pins exact dependency versions (don't edit this by hand).
- `README.md`: This file — setup instructions and guides.
- `LICENSE`: MIT license for the project.
- `.gitignore`: Tells git which files to ignore (e.g. generated images, virtual environments).
