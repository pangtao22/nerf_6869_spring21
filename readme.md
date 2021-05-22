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
However, a direction-agnostic NeRF may learn direction-dependent color by adding extraneous geometry to the scene, making the reconstructed geometry inaccurate, as shown in the figures below. (click on the images to view the full point cloud in 3D, which takes a while to load but looks cool!)

#### Point cloud generated from a direction-angostic NeRF network.
[![name](./media/point_cloud_position_only.png)](https://pangtao22.github.io/html_figures/nerf_6869_projects/position_only/position_only_point_cloud.html)

#### Point cloud generated from a direction-dependent NeRF network.
[![name](./media/point_cloud_position_and_orientation.png)](https://pangtao22.github.io/html_figures/nerf_6869_projects/position_and_direction/position_and_direction_point_cloud.html)

