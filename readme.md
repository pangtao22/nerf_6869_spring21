# 3D geometry from Neural Radiance Field (NeRF)
#### Final project of 6.869, Spring 2021.

-------
This project explores how well scene geometry can be reconstrcuted from NeRF networks when color depends only on position (<em>direction-agnostic</em>), or when color depends on both position and direction (<em>direction-dependent</em>). 

## Image synthesis
When synthesizing images from new views, a direction-agnostic NerF network appears to only miss the specular reflection effect compared with a direction-dependent NeRF network.

#### Images synthesized from a direction-angostic NeRF network.
![Alt Text](./media/direction_agnostic.gif)

#### Images synthesized from a direction-dependent NeRF network.
![Alt Text](./media/direction_dependent.gif)


## Geometry Reconstruction