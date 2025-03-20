# -*- coding: utf-8 -*-
import xml.etree.ElementTree as gfg
import os


class Ball_generator(object):

    def __init__(self):
        pass

    def generate(self, root='urdf/', file_name=None, ball_radius=None, ball_mass=None):
        file_name = os.path.join(root, file_name)
        root = gfg.Element("robot", name="ball")

        link = gfg.Element("link", name="ball")
        
        root.append(link)
        visual = gfg.Element("visual")
        link.append(visual)
        gfg.SubElement(visual, "origin", xyz="0 0 0")
        geometry = gfg.Element("geometry")
        visual.append(geometry)
        gfg.SubElement(geometry, "sphere", radius=str(ball_radius))
        collision = gfg.Element("collision")
        link.append(collision)
        gfg.SubElement(collision, "origin", xyz="0 0 0")
        geometry = gfg.Element("geometry")
        collision.append(geometry)
        gfg.SubElement(geometry, "sphere", radius=str(ball_radius))
        inertial = gfg.Element("inertial")
        link.append(inertial)
        gfg.SubElement(inertial, "mass", value=str(ball_mass))
        gfg.SubElement(inertial, "inertia", ixx="0.01", ixy="0.0", ixz="0.0", iyy="0.01", iyz="0.0", izz="0.01")

        self.make_file(file_name, root)

    def make_file(self, file_name, root):
        tree = gfg.ElementTree(root)
        with open(file_name, "wb") as files:
            tree.write(files)

