import os
from __main__ import vtk, qt, ctk, slicer
import EditorLib
from EditorLib.EditOptions import HelpButton
from EditorLib.EditOptions import EditOptions
from EditorLib import EditUtil
from EditorLib import LabelEffect

#
# The Editor Extension itself.
# 
# This needs to define the hooks to become an editor effect.
#

#
# GooCutOptions - see LabelEffect, EditOptions and Effect for superclasses
#

class GooCutOptions(EditorLib.LabelEffectOptions):
  """ GooCut-specfic gui
  """

  def __init__(self, parent=0):
    super(GooCutOptions,self).__init__(parent)

    # self.attributes should be tuple of options:
    # 'MouseTool' - grabs the cursor
    # 'Nonmodal' - can be applied while another is active
    # 'Disabled' - not available
    self.attributes = ('MouseTool')
    self.displayName = 'GooCut Effect'

  def __del__(self):
    super(GooCutOptions,self).__del__()

  def create(self):
    super(GooCutOptions,self).create()

    self.botButton = qt.QPushButton(self.frame)
    self.botButton.text = "Start Bot"
    self.frame.layout().addWidget(self.botButton)

    self.toleranceFrame = qt.QFrame(self.frame)
    self.toleranceFrame.setLayout(qt.QHBoxLayout())
    self.frame.layout().addWidget(self.toleranceFrame)
    self.widgets.append(self.toleranceFrame)
    self.toleranceLabel = qt.QLabel("Tolerance:", self.toleranceFrame)
    self.toleranceLabel.setToolTip("Set the tolerance of the wand in terms of background pixel values")
    self.toleranceFrame.layout().addWidget(self.toleranceLabel)
    self.widgets.append(self.toleranceLabel)
    self.toleranceSpinBox = qt.QDoubleSpinBox(self.toleranceFrame)
    self.toleranceSpinBox.setToolTip("Set the tolerance of the wand in terms of background pixel values")
    self.toleranceSpinBox.minimum = 0
    self.toleranceSpinBox.maximum = 1000
    self.toleranceSpinBox.suffix = ""
    self.toleranceFrame.layout().addWidget(self.toleranceSpinBox)
    self.widgets.append(self.toleranceSpinBox)

    self.maxPixelsFrame = qt.QFrame(self.frame)
    self.maxPixelsFrame.setLayout(qt.QHBoxLayout())
    self.frame.layout().addWidget(self.maxPixelsFrame)
    self.widgets.append(self.maxPixelsFrame)
    self.maxPixelsLabel = qt.QLabel("Max Pixels per click:", self.maxPixelsFrame)
    self.maxPixelsLabel.setToolTip("Set the maxPixels for each click")
    self.maxPixelsFrame.layout().addWidget(self.maxPixelsLabel)
    self.widgets.append(self.maxPixelsLabel)
    self.maxPixelsSpinBox = qt.QDoubleSpinBox(self.maxPixelsFrame)
    self.maxPixelsSpinBox.setToolTip("Set the maxPixels for each click")
    self.maxPixelsSpinBox.minimum = 1
    self.maxPixelsSpinBox.maximum = 1000
    self.maxPixelsSpinBox.suffix = ""
    self.maxPixelsFrame.layout().addWidget(self.maxPixelsSpinBox)
    self.widgets.append(self.maxPixelsSpinBox)

    HelpButton(self.frame, "Use this tool to label all voxels that are within a tolerance of where you click")

    self.botButton.connect('clicked()', self.onStartBot)
    self.toleranceSpinBox.connect('valueChanged(double)', self.onToleranceSpinBoxChanged)
    self.maxPixelsSpinBox.connect('valueChanged(double)', self.onMaxPixelsSpinBoxChanged)

    # Add vertical spacer
    self.frame.layout().addStretch(1)

    # TODO: the functionality for the steered volume should migrate to
    # the edit helper class when functionality is finalized.
    backgroundVolume = self.editUtil.getBackgroundVolume()
    labelVolume = self.editUtil.getLabelVolume()
    steeredName = backgroundVolume.GetName() + '-steered'
    steeredVolume = slicer.util.getNode(steeredName)
    if not steeredVolume:
      volumesLogic = slicer.modules.volumes.logic()
      steeredVolume = volumesLogic.CloneVolume(
                           slicer.mrmlScene, labelVolume, steeredName)
    compositeNodes = slicer.util.getNodes('vtkMRMLSliceCompositeNode*')
    for compositeNode in compositeNodes.values():
      compositeNode.SetForegroundVolumeID(steeredVolume.GetID())
      compositeNode.SetForegroundOpacity(0.5)


  def destroy(self):
    super(GooCutOptions,self).destroy()

  # note: this method needs to be implemented exactly as-is
  # in each leaf subclass so that "self" in the observer
  # is of the correct type 
  def updateParameterNode(self, caller, event):
    node = EditUtil.EditUtil().getParameterNode()
    if node != self.parameterNode:
      if self.parameterNode:
        node.RemoveObserver(self.parameterNodeTag)
      self.parameterNode = node
      self.parameterNodeTag = node.AddObserver("ModifiedEvent", self.updateGUIFromMRML)

  def setMRMLDefaults(self):
    super(GooCutOptions,self).setMRMLDefaults()
    disableState = self.parameterNode.GetDisableModifiedEvent()
    self.parameterNode.SetDisableModifiedEvent(1)
    defaults = (
      ("tolerance", "20"),
      ("maxPixels", "200"),
    )
    for d in defaults:
      param = "GooCut,"+d[0]
      pvalue = self.parameterNode.GetParameter(param)
      if pvalue == '':
        self.parameterNode.SetParameter(param, d[1])
    self.parameterNode.SetDisableModifiedEvent(disableState)

  def updateGUIFromMRML(self,caller,event):
    if self.updatingGUI:
      return
    params = ("tolerance",)
    params = ("maxPixels",)
    for p in params:
      if self.parameterNode.GetParameter("GooCut,"+p) == '':
        # don't update if the parameter node has not got all values yet
        return
    self.updatingGUI = True
    super(GooCutOptions,self).updateGUIFromMRML(caller,event)
    self.toleranceSpinBox.setValue( float(self.parameterNode.GetParameter("GooCut,tolerance")) )
    self.maxPixelsSpinBox.setValue( float(self.parameterNode.GetParameter("GooCut,maxPixels")) )
    self.updatingGUI = False

  def onToleranceSpinBoxChanged(self,value):
    if self.updatingGUI:
      return
    self.updateMRMLFromGUI()

  def onMaxPixelsSpinBoxChanged(self,value):
    if self.updatingGUI:
      return
    self.updateMRMLFromGUI()

  def updateMRMLFromGUI(self):
    if self.updatingGUI:
      return
    disableState = self.parameterNode.GetDisableModifiedEvent()
    self.parameterNode.SetDisableModifiedEvent(1)
    super(GooCutOptions,self).updateMRMLFromGUI()
    self.parameterNode.SetParameter( "GooCut,tolerance", str(self.toleranceSpinBox.value) )
    self.parameterNode.SetParameter( "GooCut,maxPixels", str(self.maxPixelsSpinBox.value) )
    self.parameterNode.SetDisableModifiedEvent(disableState)
    if not disableState:
      self.parameterNode.InvokePendingModifiedEvent()

  def onStartBot(self):
    """create the bot for background editing"""
    GooCutBot(self) 


#
# GooCutBot
#
 
# TODO: move the concept of a Bot into the Effect class
# to manage timer.  Also put Bot status indicator into
# an Editor interface.  Use slicer.modules.editorBot
# to enforce singleton instance for now.
#class GooCutBot(EditorLib.LabelEffectBot):
class GooCutBot(object):
  """
  Task to run in the background for this effect.
  Receives a reference to the currently active options
  so it can access tools if needed.
  """
  def __init__(self,options):
    self.sliceWidget = options.tools[0].sliceWidget
    if hasattr(slicer.modules, 'editorBot'):
      slicer.modules.editorBot.active = False
      del(slicer.modules.editorBot)
    slicer.modules.editorBot = self
    self.interval = 100
    self.active = False
    self.start()

  def start(self):
    self.active = True
    qt.QTimer.singleShot(self.interval, self.iteration)

  def stop(self):
    self.active = False

  def iteration(self):
    """Perform an iteration of the GooCut algorithm"""
    if not self.active:
      return

    import random
    sliceLogic = self.sliceWidget.sliceLogic()
    x = random.randint(0, self.sliceWidget.width-1)
    y = random.randint(0, self.sliceWidget.height-1)
    logic = GooCutLogic(sliceLogic)
    logic.apply((x,y))
    qt.QTimer.singleShot(self.interval, self.iteration)
    


#
# GooCutTool
#
 
class GooCutTool(EditorLib.LabelEffectTool):
  """
  One instance of this will be created per-view when the effect
  is selected.  It is responsible for implementing feedback and
  label map changes in response to user input.
  This class observes the editor parameter node to configure itself
  and queries the current view for background and label volume
  nodes to operate on.
  """

  def __init__(self, sliceWidget, threeDWidget=None):
    super(GooCutTool,self).__init__(sliceWidget, threeDWidget)

  def cleanup(self):
    super(GooCutTool,self).cleanup()

  def processEvent(self, caller=None, event=None):
    """
    handle events from the render window interactor
    """
    if event == "LeftButtonPressEvent":
      xy = self.interactor.GetEventPosition()
      sliceLogic = self.sliceWidget.sliceLogic()
      logic = GooCutLogic(sliceLogic)
      logic.apply(xy)
      self.abortEvent(event)
    else:
      pass


#
# GooCutLogic
#
 
class GooCutLogic(EditorLib.LabelEffectLogic):
  """
  This class contains helper methods for a given effect
  type.  It can be instanced as needed by an GooCutTool
  or GooCutOptions instance in order to compute intermediate
  results (say, for user feedback) or to implement the final 
  segmentation editing operation.  This class is split
  from the GooCutTool so that the operations can be used
  by other code without the need for a view context.
  """

  def __init__(self,sliceLogic):
    self.sliceLogic = sliceLogic

  def apply(self,xy):
    # TODO: save the undo state - not yet available for extensions
    # EditorStoreCheckPoint $_layers(label,node)
    
    #
    # get the parameters from MRML
    #
    node = EditUtil.EditUtil().getParameterNode()
    tolerance = float(node.GetParameter("GooCut,tolerance"))
    maxPixels = float(node.GetParameter("GooCut,maxPixels"))


    #
    # get the label and background volume nodes
    #
    labelLogic = self.sliceLogic.GetLabelLayer()
    labelNode = labelLogic.GetVolumeNode()
    labelNode.SetModifiedSinceRead(1)
    backgroundLogic = self.sliceLogic.GetBackgroundLayer()
    backgroundNode = backgroundLogic.GetVolumeNode()

    #
    # get the ijk location of the clicked point
    # by projecting through patient space back into index
    # space of the volume.  Result is sub-pixel, so round it
    # (note: bg and lb will be the same for volumes created
    # by the editor, but can be different if the use selected
    # different bg nodes, but that is not handled here).
    # 
    xyToIJK = labelLogic.GetXYToIJKTransform().GetMatrix()
    ijkFloat = xyToIJK.MultiplyPoint(xy+(0,1))[:3]
    ijk = []
    for element in ijkFloat:
      try:
        intElement = int(element)
      except ValueError:
        intElement = 0
      ijk.append(intElement)
    ijk.reverse()
    ijk = tuple(ijk)

    #
    # Get the numpy array for the bg and label
    #
    import vtk.util.numpy_support
    backgroundImage = backgroundNode.GetImageData()
    labelImage = labelNode.GetImageData()
    shape = list(backgroundImage.GetDimensions())
    shape.reverse()
    backgroundArray = vtk.util.numpy_support.vtk_to_numpy(backgroundImage.GetPointData().GetScalars()).reshape(shape)
    labelArray = vtk.util.numpy_support.vtk_to_numpy(labelImage.GetPointData().GetScalars()).reshape(shape)

    #
    # do a re
    value = backgroundArray[ijk]
    label = EditUtil.EditUtil().getLabel()
    lo = value - tolerance
    hi = value + tolerance
    pixelsSet = 0
    toVisit = [ijk,]
    while toVisit != []:
      location = toVisit.pop()
      try:
        l = labelArray[location]
        b = backgroundArray[location]
      except IndexError:
        continue
      if l != 0 or b < lo or b > hi:
        continue
      labelArray[location] = label
      pixelsSet += 1
      if pixelsSet > maxPixels:
        toVisit = []
      else:
        # add the 6 neighbors to the stack
        toVisit.append((location[0] - 1, location[1]    , location[2]    ))
        toVisit.append((location[0] + 1, location[1]    , location[2]    ))
        toVisit.append((location[0]    , location[1] - 1, location[2]    ))
        toVisit.append((location[0]    , location[1] + 1, location[2]    ))
        toVisit.append((location[0]    , location[1]    , location[2] - 1))
        toVisit.append((location[0]    , location[1]    , location[2] + 1))

    labelImage.Modified()
    labelNode.Modified()

#
# The GooCutExtension class definition 
#

class GooCutExtension(object):
  """Organizes the Options, Tool, and Logic classes into a single instance
  that can be managed by the EditBox
  """

  def __init__(self):
    # name is used to define the name of the icon image resource (e.g. GooCut.png)
    self.name = "GooCut"
    # tool tip is displayed on mouse hover
    self.toolTip = "GooCut: steered segmenter"

    self.options = GooCutOptions
    self.tool = GooCutTool
    self.logic = GooCutLogic

#
# GooCut
#

class GooCut:
  """
  This class is the 'hook' for slicer to detect and recognize the extension
  as a loadable scripted module
  """
  def __init__(self, parent):
    parent.title = "Editor GooCut Effect"
    parent.categories = ["Developer Tools.Editor Extensions"]
    parent.contributors = ["Steve Pieper"]
    parent.helpText = """
    Example of an editor extension.  No module interface here, only in the Editor module
    """
    parent.acknowledgementText = """
    This editor extension was developed by 
    Steve Pieper, Isomics, Inc.
    """

    # don't show this module - it only appears in the Editor module
    parent.hidden = True

    # Add this extension to the editor's list for discovery when the module
    # is created.  Since this module may be discovered before the Editor itself,
    # create the list if it doesn't already exist.
    try:
      slicer.modules.editorExtensions
    except AttributeError:
      slicer.modules.editorExtensions = {}
    slicer.modules.editorExtensions['GooCut'] = GooCutExtension

#
# GooCutWidget
#

class GooCutWidget:
  def __init__(self, parent = None):
    self.parent = parent
    
  def setup(self):
    # don't display anything for this widget - it will be hidden anyway
    pass

  def enter(self):
    pass
    
  def exit(self):
    pass


