# How to Add Free-Moving Objects to the World

This guide explains how to add new 3D meshes (like cups, puzzle pieces, or tools) as free-moving objects placed on the table in your MuJoCo scene.

## 1. Prepare the Mesh File
Ensure your object mesh is in `.stl` format and located in the `World/` directory (e.g., `World/Cup.stl`). If you only have `.3mf` files, convert them to `.stl`, as MuJoCo natively supports `.stl` without requiring external decoders.

## 2. Load the Mesh Asset
Open the scene file, typically `panda_mujoco/world.xml`. Find the `<asset>` block and add your mesh inside it. 

*Note: Remember to apply the `scale="0.001 0.001 0.001"` attribute so the imported 3D model is correctly scaled down from millimeters to meters!*

```xml
<asset>
    <!-- ...existing assets... -->
    
    <!-- 1. Load the mesh -->
    <mesh name="cup_mesh" file="../../World/Cup.stl" scale="0.001 0.001 0.001" />
    
    <!-- 2. (Optional) Create a material with a color for it -->
    <material name="cup_mat" rgba="0.2 0.6 0.8 1" />
</asset>
```

## 3. Add the Object to the World
Next, scroll down to the `<worldbody>` section. You'll define a new `<body>` here. 

To make the object "free-moving" (able to fall with gravity, be grabbed, and collide), you **must** include a `<joint type="free" />` inside the body block.

```xml
<worldbody>
    <!-- ...existing table and robot... -->

    <!-- The 'pos' determines the spawn location -->
    <body name="cup" pos="0.4 0 0.1">
        <!-- This makes the object dynamic and disconnected from the world origin -->
        <joint type="free" />
        
        <!-- The physical geometry, linking back to the mesh and material we defined -->
        <geom type="mesh" mesh="cup_mesh" material="cup_mat" condim="3" />
    </body>
</worldbody>
```

## 4. Fine-Tune the Spawning Position
The `pos="X Y Z"` attribute in your `<body>` tag determines the object's starting coordinates:
- The base of the Panda arm is at `(0, 0, 0)`.
- The surface of the table is also around `Z = 0`.
- To drop an object safely onto the table, place its `pos` slightly above `0` (e.g., `Z = 0.05` or `Z = 0.1`). When you launch `main.py`, gravity will naturally pull the object down until it rests on the table.
