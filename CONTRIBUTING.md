# Contributing to Orgo MCP Server

```
 ___  ____   ___  ___    __  __  ___ ___
/ _ \|  _ \ / __|/ _ \  |  \/  |/ __| _ \
| (_) | |_) | (_ | (_) | | |\/| | (__|  _/
 \___/|____/ \___|\___ / |_|  |_|\___|_|

        Welcome, Future Contributor!
```

---

## Your Quest Begins Here

So you want to help AI agents control virtual computers? **Excellent choice, adventurer!**

Whether you're fixing a tiny typo or building a massive new feature, you're about to become part of something awesome. Let's make this journey fun!

---

## The Contributor's Map

```
    START
      |
      v
+-------------+     +--------------+     +---------------+
|  Fork       | --> |  Branch      | --> |  Code         |
+-------------+     +--------------+     +---------------+
                                                |
                                                v
+-------------+     +--------------+     +---------------+
|  Celebrate  | <-- |  PR          | <-- |  Test         |
+-------------+     +--------------+     +---------------+
```

---

## Level 1: Setting Up Your Dev Environment

### Prerequisites Checklist

- [ ] Python 3.10 or higher
- [ ] Git installed and configured
- [ ] An [Orgo API key](https://orgo.ai) for testing
- [ ] Your favorite code editor
- [ ] Coffee (optional but recommended)

### Installation Speedrun

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/orgo-mcp.git
cd orgo-mcp

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Set up your environment
cp .env.example .env
# Add your ORGO_API_KEY to .env
```

> **Achievement Unlocked:** Dev Environment Ready!

---

## Level 2: Choose Your Adventure

### Bug Hunter (Difficulty: 2/5)

```
+----------------------------------+
|     BUG HUNTER                   |
+----------------------------------+
|                                  |
|   Found a bug? Squash it!        |
|                                  |
|   XP: +50 for fixes              |
|   XP: +100 with tests included   |
|                                  |
+----------------------------------+
```

1. Check [existing issues](https://github.com/nickvasilescu/orgo-mcp/issues)
2. Comment "I'll take this!" to claim it
3. Fix, test, PR - you know the drill!

### Feature Architect (Difficulty: 3/5)

```
+----------------------------------+
|     FEATURE ARCHITECT            |
+----------------------------------+
|                                  |
|   Building something new?        |
|                                  |
|   XP: +200 for features          |
|   XP: +300 with documentation    |
|                                  |
+----------------------------------+
```

1. Open an issue first to discuss
2. Get a thumbs up from maintainers
3. Build it, document it, ship it!

### Docs Wizard (Difficulty: 1/5)

```
+----------------------------------+
|     DOCS WIZARD                  |
+----------------------------------+
|                                  |
|   Words matter too!              |
|                                  |
|   XP: +25 for typo fixes         |
|   XP: +75 for new docs           |
|                                  |
+----------------------------------+
```

No fix is too small. Found a typo? Fix it! Missing explanation? Add it!

---

## Level 3: The Code Style Guide

### The Golden Rules

| Rule | Description | Why |
|------|-------------|-----|
| Keep it clean | Follow PEP 8 | Consistency is key |
| Be descriptive | Meaningful names | Future you will thank you |
| Comment wisely | Explain the "why" | Not the "what" |
| Test everything | If untested, unfinished | Trust the tests |

### Project Structure

```
orgo-mcp/
|
+-- orgo_mcp.py      # The main server - where the magic happens
+-- pyproject.toml   # Project configuration
+-- README.md        # User-facing documentation
+-- CONTRIBUTING.md  # You are here!
|
+-- tests/           # Your tests go here
```

---

## Level 4: Crafting the Perfect PR

### Pre-Flight Checklist

```
+--------------------------------------------------+
|  BEFORE YOU HIT THAT GREEN BUTTON...             |
+--------------------------------------------------+
|                                                  |
|  [ ] Code follows style guidelines               |
|  [ ] Tests pass locally                          |
|  [ ] Documentation updated (if needed)           |
|  [ ] Commit messages are clear                   |
|  [ ] Branch is up to date with main              |
|                                                  |
+--------------------------------------------------+
```

### Commit Message Format

```
type: short description

[optional body with more details]

Types:
- feat:     New feature
- fix:      Bug fix
- docs:     Documentation only
- style:    Formatting, no code change
- refactor: Code restructuring
- test:     Adding tests
- chore:    Maintenance tasks
```

**Examples:**
```
feat: add screenshot annotation support
fix: resolve timeout in bash command execution
docs: clarify API key configuration steps
```

---

## Boss Battle: Getting Your PR Merged

### What to Expect

```
AUTOMATED CHECKS
       |
       v
CODE REVIEW
       |
       v
FEEDBACK LOOP
       |
       v
MERGE!
```

### Review Timeline

| PR Size | Expected Wait |
|---------|---------------|
| Small fixes | 1-2 days |
| Features | 3-5 days |
| Large changes | 1-2 weeks |

> Be patient - we review everything carefully!

---

## Bonus Levels

### Community Guidelines

- **Be kind** - We're all here to learn and grow
- **Be patient** - Everyone has different experience levels
- **Be constructive** - Critique code, not people
- **Have fun** - This is open source, enjoy it!

### Getting Help

| Need | Where to Go |
|------|-------------|
| Bug reports | [GitHub Issues](https://github.com/nickvasilescu/orgo-mcp/issues) |
| Feature ideas | [GitHub Discussions](https://github.com/nickvasilescu/orgo-mcp/discussions) |
| Quick questions | Open an issue with `[Question]` prefix |

---

## Hall of Fame

Every contributor gets a spot in our hearts (and the git history)!

Your contributions help AI agents everywhere work with virtual computers more effectively.

---

## Ready to Begin?

```
+----------------------------------------------------------+
|                                                          |
|   Fork the repo, pick an issue, and start coding!        |
|                                                          |
|   We can't wait to see what you build.                   |
|                                                          |
+----------------------------------------------------------+
```

---

**Thank you for contributing to Orgo MCP Server!**

*Together, we're building the future of AI-powered computer control.*
