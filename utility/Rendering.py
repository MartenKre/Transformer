from direct.showbase.ShowBase import ShowBase
from panda3d.core import DirectionalLight, PointLight, AmbientLight, AntialiasAttrib
from panda3d.core import Vec3, LineSegs, NodePath, Material
from panda3d.core import *
import numpy as np
from utility.Transformations import T_W_Ship, LatLng2ECEF, T_ECEF_Ship, extract_RPY
from copy import deepcopy
from direct.filter.CommonFilters import CommonFilters

# Set default window size
loadPrcFileData("", "win-size 960 540")

class RenderAssociations(ShowBase):
    def __init__(self, lock, parent):
        super().__init__()
        self.lock = lock    # lock for threading
        self.parent = parent
        # panda3D settigns
        self.render.setShaderAuto()
        self.render.setAntialias(AntialiasAttrib.MMultisample)  # Enable anti-aliasing
        lens = base.cam.node().getLens()
        lens.setFar(3000)
        lens.setFov(55,40)
        #filter = CommonFilters(self.win, self.cam)
        #filter.setBloom()

        # general settings
        self.disableMouse()
        self.follow_ship = True

        #  data
        self.t = 0
        self.dt = 0.03
        self.ship_path = [] # list to store ship path
        self.ship_path_length = 0
        self.prev_idx = 0
        self.counter = 0    # frame counter
        self.preds=[]   # list to store buoy preds for current frame    
        self.buoysGT = []   # list to store buoy GTs
        self.matchingColorsGT = {}
        self.matchingColorsPreds = {}

        # transformation matrices -> need to be initialized
        self.W_T_Ship = None    # Matrix to transform ship coords to World CS (determined by init pos & heading)
        self.ECEF_T_W = None    # Matrix to transform ECEF coords to World CS (determined by init pos & heading)

        # load blender models (exported to .egg)
        self.buoy_model_green = self.loader.loadModel("utility/render_assets/buoy_green.egg")
        #self.buoy_model_red = self.loader.loadModel("utility/render_assets/buoy_red.egg")
        self.buoy_wireframe = self.loader.loadModel("utility/render_assets/buoy_wireframe.egg")

        # start basic rendering
        self.set_background_color(0.71, 0.71, 0.71, 1)  # (R, G, B, A) values (0 to 1)
        # render ship
        self.ship = self.loader.loadModel("utility/render_assets/ship.egg")
        self.ship_scale = 15
        self.ship.setScale(self.ship_scale, self.ship_scale, self.ship_scale)  # Scale the plane to make it larger
        self.ship.setPos(0, 0, 0)  # Position it at the origin
        self.ship.setHpr(0, 0, 0)
        self.ship.setShaderAuto()
        self.ship.reparentTo(self.render)  # Add to the scene

        # render data buoys
        self.pred_buoys_render = []
        self.gt_buoys_render = []
        self.buoy_lights = []
        self.line_node_path = None  # Store the node path for the line
        self.lines = LineSegs()  # Create LineSegs object once 
        self.lines.setColor(1, 0, 0, 1)    # (R, G, B, A) for red
        self.lines.set_thickness(3)
        self.floor = None

        # Add a point light
        plight = PointLight("plight")
        plight.setColor((0.3, 0.3, 0.3, 1))  # Slightly red light
        plight.attenuation = (0.3, 0, 0.00000005)
        plight.set_shadow_caster(True, 1024, 1024)  # Enable shadows with resolution
        self.plnp = self.render.attachNewNode(plight)
        self.plnp.setPos(0, 0, 100)  # Position of the point light
        self.render.setLight(self.plnp)

        # Add ambient light
        alight = AmbientLight("alight")
        alight.setColor((0.1, 0.1, 0.1, 1))  # Dim light
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        # render surface
        self.renderSurface()

        # add rendering loops to tasks
        self.taskMgr.add(self.render_loop, "render_loop")

    def render_loop(self, task):
        # called each frame
        with self.lock:
            self.updateLogic()
            self.showScreen()
            return task.cont

    def initTransformations(self, lat, lng, heading):
        # function needs to be called to initialize Transformation matrices
        x, y, z = LatLng2ECEF(lat, lng)
        self.ECEF_T_W = T_ECEF_Ship(x, y, z, 0)
        self.W_T_Ship = T_W_Ship(np.array([0, 0, -3.8]), heading)
        self.ship_path.append(np.array([0, 0, -3.8]))

    def setShipData(self, lat, lng, heading):
        # current pos and heading of ship
        x, y, z = LatLng2ECEF(lat, lng)
        p_WCS = np.linalg.pinv(self.ECEF_T_W) @ np.array([x,y,z,1])
        p_WCS[2] = -3.8
        self.ship_path.append(p_WCS[0:3])
        self.W_T_Ship = T_W_Ship(p_WCS[:3], np.radians(heading))

    def setPreds(self, preds, matching_indices={}):
        # list of buoy preds (lat&lng coords)
        self.preds = []
        for pred in preds:
            x,y,z = LatLng2ECEF(pred[0], pred[1])
            p_buoy = np.linalg.pinv(self.ECEF_T_W) @ np.array([x,y,z,1])  # buoy pred in ship CS
            self.preds.append([p_buoy[0], p_buoy[1]])
            self.matchingColorsPreds = matching_indices

    def setBuoyGT(self, buoysGT, matching_indices={}):
        # list of buoy gt data (lat & lng coords)
        # optional: dict of matched indices with color RGBA e.g. {3:(0,0,1,0), 5:{1,1,0,1}}
        self.buoysGT = []
        for buoy in buoysGT:
            x,y,z = LatLng2ECEF(buoy[0], buoy[1])
            p_buoy = np.linalg.pinv(self.ECEF_T_W) @ np.array([x,y,z,1])  # buoy pred in ship CS
            self.buoysGT.append([p_buoy[0], p_buoy[1]])
            self.matchingColorsGT = matching_indices
        
    def renderShipIcon(self):
        # render ship icon
        pos = self.W_T_Ship[:3, 3] # get current pos
        self.ship.setPos(pos[0], pos[1], pos[2])
        # get heading
        roll, pitch, yaw = extract_RPY(self.W_T_Ship[:3,:3])
        self.ship.setHpr(-90+np.rad2deg(yaw), 0, 0)

        # adjust origin of point light to be over ship
        self.plnp.setPos(pos[0], pos[1], 150)

    def renderShipPath(self):
        # render ship path
        if len(self.ship_path) > 1 and len(self.ship_path) > self.ship_path_length+5:
            self.ship_path_length = len(self.ship_path) 
            self.lines.move_to(Vec3(*self.ship_path[self.prev_idx]))
            self.lines.draw_to(Vec3(*self.ship_path[-1]))
            self.prev_idx = len(self.ship_path) -1

            line_node = self.lines.create()
            line_node_path = NodePath(line_node)
            line_node_path.setLightOff()  # Disable lighting for the line
            line_node_path.setShaderOff()
            line_node_path.reparent_to(self.render)  # Attach to the render

    def purgeBuoyRender(self):
        for light in reversed(self.buoy_lights):
            self.render.clearLight(light)
            light.removeNode()
            self.buoy_lights.pop()
        for buoy in reversed(self.gt_buoys_render):
            buoy.removeNode()
            self.gt_buoys_render.pop()
        for buoy in reversed(self.pred_buoys_render):
            buoy.removeNode()
            self.pred_buoys_render.pop()

    def renderBuoyGT(self):
        for i, (x, y) in enumerate(self.buoysGT):
            buoy = deepcopy(self.buoy_model_green)
            color = (0,1,0,1)
            if i in self.matchingColorsGT:
                color = self.matchingColorsGT[i]
                myMaterial = Material()
                myMaterial.setAmbient(color) # Make this material blue
                myMaterial.setDiffuse(color)
                buoy.setMaterial(myMaterial, priority=1)

            buoy.setScale(8, 8, 8)
            buoy.setPos(x, y, -1.8)
            p = 3 * np.sin(1.5*self.t)
            r = 1.5 * np.sin(1*self.t)
            buoy.setHpr(0, p, r)
            buoy.setShaderAuto()
            buoy.reparentTo(self.render)
            self.gt_buoys_render.append(buoy)
            plight = PointLight("plight")
            plight.setColor(color)
            plight.attenuation = (0.6, 0, 0.0005)
            plnp = buoy.attachNewNode(plight)
            self.buoy_lights.append(plnp)
            self.render.setLight(plnp)

    def renderBuoyPreds(self):
        # the predictions are expected to be in lat lon coords
        for i, (x, y) in enumerate(self.preds):
            buoy = deepcopy(self.buoy_wireframe)

            color = (1,0,0,1)
            if i in self.matchingColorsPreds:
                color = self.matchingColorsPreds[i]
                myMaterial = Material()
                myMaterial.setAmbient(color) # Make this material blue
                myMaterial.setDiffuse(color)
                buoy.setMaterial(myMaterial, priority=1)

            buoy.setScale(7, 7, 7)
            p = 3 * np.sin(1.5*self.t+1.5)
            r = 1.5 * np.sin(1*self.t+1.5)
            buoy.setHpr(0, p, r)
            buoy.setPos(x, y, -1.8)
            buoy.reparentTo(self.render)
            self.pred_buoys_render.append(buoy)

            plight = PointLight("plight")
            plight.setColor(color)
            plight.attenuation = (0.5, 0, 0.0005)
            plnp = buoy.attachNewNode(plight)
            self.buoy_lights.append(plnp)
            self.render.setLight(plnp)

    def renderSurface(self):
        self.floor  = self.loader.loadModel("utility/render_assets/floor5.egg")  # Load a plane model
        self.floor.setScale(100, 100, 1)  # Scale the plane to make it larger
        self.floor.setPos(0, 0, -4)  # Position it at the origin
        self.floor.setShaderAuto()  # Ensure correct shading
        self.floor.reparentTo(self.render)  # Add to the scene

    def setSurfacePos(self, pos):
        self.floor.setPos(*pos[0:2], -4)

    def updateLogic(self):
        # logic function, called each frame
        self.t += self.dt
        self.counter += 1

    def Camera(self):
        # cam at World CS pos & heading of 0
        self.camera.setPos(-500, 0, 200)  # X, Y, Z position
        self.camera.lookAt(0, 0, 0)  # Look at the origin (0, 0, 0)

    def shipCam(self):
        # cam that follows ship
        pos = np.array([-400, 0, 180, 1])  # camera pos in ship cs
        view_vec = np.array([600, 0, 0, 1])
        pos = self.W_T_Ship@pos
        view_vec = self.W_T_Ship@view_vec

        self.camera.setPos(*pos[:3])  # X, Y, Z position
        self.camera.lookAt(*view_vec[:3])  # Look at the origin (0, 0, 0)

        # adjust plane center to be at location of ship
        self.setSurfacePos(pos[:3])

    def camSelect(self):
        if self.follow_ship == False:
            self.Camera()
        else:
            self.shipCam()

    def showScreen(self):
        self.camSelect()
        self.purgeBuoyRender()
        self.renderBuoyPreds()
        self.renderBuoyGT()
        self.renderShipIcon()
        self.renderShipPath()

    def changeCamera(self):
        self.follow_ship = not self.follow_ship
