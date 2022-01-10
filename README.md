Package Structure

To import `POISEVAE` from anywhere, do `export PYTHONPATH="$PYTHONPATH:/path/to/POISEVAE"`

`POISE_VAE.py` contains the implementation to the main models. `utils` contains some useful functions and `torch.utils.Dataset` classes.

To synchronize the code between Jupyter notebooks and the scripts, please update `__version__` of the class(es) changed, so that we know which ones are newer. 
- At this point, only update `__version__` in the file you change; e.g. if you change `class POISEVAE` in the notebook, update the version here but not the version in `POISE_VAE.py` and other notebooks.

After proper testing please synchronize the newer version to the script/notebook.

Backup frequently in your local directory.
