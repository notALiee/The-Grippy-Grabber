import xml.etree.ElementTree as ET
import random

xml_file = "world.xml"

tree = ET.parse(xml_file)
root = tree.getroot()

objects_to_randomize = [
    "cup1", "cup2", "cup3",
    "book1", "book2", "book3",
    "pen1", "pen2", "pen3"
]

# Find worldbody
worldbody = root.find("worldbody")

# Remove existing wall bodies (if present)
wall_names = ["wall_a", "wall_b", "wall_c"]
for wall_name in wall_names:
    for body in worldbody.findall(f"body[@name='{wall_name}']"):
        worldbody.remove(body)

# (Re)add walls with specified geometry
walls = [
    {
        "name": "wall_a",
        "pos":  "0.40 -0.20 0.20",
        "geom": {
            "type": "box",
            "size": "0.01 0.20 0.20",
            "rgba": "0.5 0.3 0.2 1",
            "group": "1",
            "contype": "1",
            "conaffinity": "1"
        }
    },
    {
        "name": "wall_b",
        "pos":  "0.75 0.05 0.25",
        "geom": {
            "type": "box",
            "size": "0.15 0.01 0.25",
            "rgba": "0.5 0.3 0.2 1",
            "group": "1",
            "contype": "1",
            "conaffinity": "1"
        }
    },
    {
        "name": "wall_c",
        "pos":  "0.20 0.10 0.18",
        "geom": {
            "type": "box",
            "size": "0.01 0.10 0.18",
            "rgba": "0.5 0.3 0.2 1",
            "group": "1",
            "contype": "1",
            "conaffinity": "1"
        }
    }
]

for wall in walls:
    body_el = ET.Element("body", name=wall["name"], pos=wall["pos"])
    geom_el = ET.Element("geom", **wall["geom"])
    body_el.append(geom_el)
    worldbody.append(body_el)

# Randomize pose and color for objects
for body in worldbody.findall("body"):
    name = body.get("name")
    if name in objects_to_randomize:
        # Random position
        x = round(random.uniform(0.2, 0.9), 3)
        y = round(random.uniform(-0.7, 0.3), 3)
        z = round(random.uniform(0.1, 0.15), 3)
        body.set("pos", f"{x} {y} {z}")

        # Set identity quaternion (no rotation)
        body.set("quat", "1 0 0 0")

        # Random color
        geom = body.find("geom")
        if geom is not None:
            r = round(random.uniform(0, 1), 3)
            g = round(random.uniform(0, 1), 3)
            b = round(random.uniform(0, 1), 3)
            geom.set("rgba", f"{r} {g} {b} 1")

            # Remove old material attribute
            if "material" in geom.attrib:
                del geom.attrib["material"]

tree.write(xml_file)
print("Scene randomized and walls generated!")
