# AmateurTelescopeMaking

Computes the surface of the parabolic mirror of the requested aperture and f number.
Finds the closest sphere.
Shows spherical aberration from the center of the sphere needed.
Simulates Ronchi, Foucault and wire tests, and produces animations.
See ```design_mirror_simulate_ronchi_foucault_wire.ipynb```

Analysis of wire test in terms of surface accuracy.

To create simulations and animations of the Ronchi/Foucault/wire tests,
requires pathos for multiprocessing (https://pypi.org/project/pathos/):
```
pip install pathos
```
