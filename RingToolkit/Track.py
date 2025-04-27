import subprocess
from .Write import ele_setup
# ----------------------------------------------------------------------------------------------------------------------
def track(lattice):
    ele_setup(lattice)
    subprocess.run(f"elegant {lattice.name}.ele")