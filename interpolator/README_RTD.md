# Hosting Documentation on Read the Docs

## Setup Instructions

### 1. Push Your Repository to GitHub/GitLab

Make sure your repository is available on GitHub, GitLab, or Bitbucket.

### 2. Sign Up / Log In to Read the Docs

1. Go to https://readthedocs.org/
2. Sign up for a free account (or log in if you already have one)
3. You can sign in with your GitHub/GitLab account

### 3. Import Your Project

1. Go to https://readthedocs.org/dashboard/
2. Click "Import a Project"
3. Click "Import manually"
4. Fill in the details:
   - **Name**: Your project name (e.g., "5d-neural-network-interpolator")
   - **Repository URL**: Your Git repository URL
   - **Repository type**: Git
   - **Default branch**: main (or master, depending on your repo)
5. Click "Create"

### 4. Configure Build Settings

Read the Docs will automatically detect the `.readthedocs.yml` file. If you need to adjust settings:

1. Go to your project's "Admin" page
2. Under "Settings" → "Advanced Settings":
   - **Python configuration file**: Leave default (uses .readthedocs.yml)
   - **Requirements file**: Not needed (we use pyproject.toml)
   - **Python interpreter**: Python 3.12

### 5. Trigger a Build

1. Go to "Builds" tab
2. Click "Build version: latest"
3. Wait for the build to complete (usually 2-5 minutes)

### 6. Access Your Documentation

Once built, your documentation will be available at:
- **Latest version**: `https://YOUR-PROJECT-NAME.readthedocs.io/`
- **Stable version**: `https://YOUR-PROJECT-NAME.readthedocs.io/en/stable/`

## Configuration Details

The `.readthedocs.yml` file configures:

- **Python version**: 3.12
- **Installation method**: pip install from `backend/` directory
- **Dependencies**: Installs with `[docs]` extra (includes Sphinx)
- **Sphinx config**: Points to `backend/docs/source/conf.py`
- **Output**: HTML documentation

## Troubleshooting

### Build Fails

1. Check the build logs in the "Builds" tab
2. Common issues:
   - Missing dependencies → Check `pyproject.toml` has `docs` extras
   - Path issues → Verify `backend/docs/source/conf.py` exists
   - Import errors → Check that modules can be imported

### Documentation Not Updating

- Read the Docs builds on every push to default branch
- You can manually trigger builds from the dashboard
- Check that your commits are pushed to the repository

### Custom Domain (Optional)

You can set up a custom domain:
1. Go to "Admin" → "Domains"
2. Add your custom domain
3. Follow DNS configuration instructions

## Automatic Updates

Read the Docs will automatically rebuild documentation when you:
- Push commits to the default branch (main/master)
- Create new tags/versions
- Update via webhook (if configured)

