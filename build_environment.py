from typing import *
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (MultibodyPlant, Parser, DiagramBuilder,
                         AddMultibodyPlantSceneGraph, ConnectMeshcatVisualizer,
                        RigidTransform, CameraInfo, RgbdSensor,
                        Simulator, PidController, LogOutput,
                         PiecewisePolynomial)

from pydrake.geometry.render import (
    ClippingRange,
    DepthRange,
    DepthRenderCamera,
    RenderCameraCore,
    RenderLabel,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
)

from contact_aware_control.plan_runner.plan_utils import RenderSystemWithGraphviz
from pydrake.math import RollPitchYaw


#%%
dir_path = os.path.dirname(os.path.realpath(__file__))
model_dir_path = os.path.join(dir_path, "models")
object_names = ["M", "I", "T"]
object_sdf_paths = [
    os.path.join(model_dir_path, "{}.sdf".format(name))
    for name in object_names
]

# camera intrinsics
renderer_name = "vtk_renderer"

# scene camera
scene_camera_properties = DepthRenderCamera(
    RenderCameraCore(
        renderer_name,
        CameraInfo(width=320, height=240, fov_y=np.pi/3),
        ClippingRange(0.01, 10.0),
        RigidTransform()),
    DepthRange(0.01, 5.0))


#%%
builder = DiagramBuilder()

# MultibodyPlant
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
parser = Parser(plant=plant, scene_graph=scene_graph)

# Add objects
object_models = dict()
R = RollPitchYaw(-np.pi / 2, 0, 0).ToRotationMatrix()
X_WBs = {"M": RigidTransform(R, [1.0, 0, 0]),
         "I": RigidTransform(R),
         "T": RigidTransform(R, [-1.0, -0.2, 0])}

for object_sdf_path, (object_name, X_AB) in zip(
        object_sdf_paths, X_WBs.items()):
    model = parser.AddModelFromFile(object_sdf_path, object_name)
    plant.WeldFrames(frame_on_parent_P=plant.world_frame(),
                     frame_on_child_C=plant.GetFrameByName("base_link", model),
                     X_PC=X_AB)
    object_models[object_name] = model

camera_dummy_model = parser.AddModelFromFile(
    os.path.join(model_dir_path, "dummy.sdf"))

plant.Finalize()

# Add renderer.
scene_graph.AddRenderer(
    renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))

# Add camera looking at the scene.
camera_body = plant.GetBodyByName("base_link", camera_dummy_model)
camera_frame_id = plant.GetBodyFrameIdOrThrow(camera_body.index())
scene_camera = RgbdSensor(camera_frame_id, X_PB=RigidTransform(),
                          depth_camera=scene_camera_properties)
builder.AddSystem(scene_camera)
builder.Connect(
    scene_graph.get_query_output_port(),
    scene_camera.query_object_input_port())

# meshcat visualizer.
meshcat_vis = ConnectMeshcatVisualizer(
    builder, scene_graph, open_browser=False, frames_to_draw=[camera_frame_id])


diagram = builder.Build()
RenderSystemWithGraphviz(diagram)

#%%
context = diagram.CreateDefaultContext()
context_scene_camera = diagram.GetSubsystemContext(scene_camera, context)
context_plant = diagram.GetSubsystemContext(plant, context)
context_meshcat = diagram.GetSubsystemContext(meshcat_vis, context)
#%%
sim = Simulator(diagram, context)
sim.Initialize()

#%%
data_dict = {}
prefix = ""

r = 2.8
X_WB = RigidTransform()
angle = np.random.rand() * np.pi
X_WB.set_translation(r * np.array([np.cos(angle), np.sin(angle), 0]))
X_WB.set_rotation(
    RollPitchYaw(-np.pi/2, 0, angle + np.pi/2).ToRotationMatrix())
plant.SetFreeBodyPose(context_plant, camera_body, X_WB)
meshcat_vis.DoPublish(context_meshcat, [])

scene_image = scene_camera.color_image_output_port().Eval(
    context_scene_camera).data
plt.imshow(scene_image)
plt.show()

