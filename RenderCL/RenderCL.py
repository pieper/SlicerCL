import os
import unittest
from __main__ import vtk, qt, ctk, slicer

try:
  import pyopencl
  import pyopencl.array
  import numpy
except ImportError:
  raise "No OpenCL for you!\nInstall pyopencl in slicer's python installation."
import vtk.util.numpy_support

#
# RenderCL
#

class RenderCL:
  def __init__(self, parent):
    parent.title = "RenderCL"
    parent.categories = ["Work in Progress.Rendering"]
    parent.dependencies = []
    parent.contributors = ["Steve Pieper (Isomics)"]
    parent.helpText = """
Scripted module implementing a volume renderer using pyopencl.
    """
    parent.acknowledgementText = """
    This file was originally developed by Steve Pieper
and was partially funded by NIH grant P41 RR132183
""" # replace with organization, grant and thanks.
    self.parent = parent

    # Add this test to the SelfTest module's list for discovery when the module
    # is created.  Since this module may be discovered before SelfTests itself,
    # create the list if it doesn't already exist.
    try:
      slicer.selfTests
    except AttributeError:
      slicer.selfTests = {}
    slicer.selfTests['RenderCL'] = self.runTest

  def runTest(self):
    tester = RenderCLTest()
    tester.runTest()

#
# qRenderCLWidget
#

class RenderCLWidget:
  def __init__(self, parent = None):
    if not parent:
      self.parent = slicer.qMRMLWidget()
      self.parent.setLayout(qt.QVBoxLayout())
      self.parent.setMRMLScene(slicer.mrmlScene)
    else:
      self.parent = parent
    self.layout = self.parent.layout()
    if not parent:
      self.setup()
      self.parent.show()

    self.logic = None

  def setup(self):
    # Instantiate and connect widgets ...

    self.imageViewer = vtk.vtkImageViewer()

    #
    # Reload and Test area
    #
    reloadCollapsibleButton = ctk.ctkCollapsibleButton()
    reloadCollapsibleButton.text = "Reload && Test"
    self.layout.addWidget(reloadCollapsibleButton)
    reloadFormLayout = qt.QFormLayout(reloadCollapsibleButton)

    # reload button
    self.reloadButton = qt.QPushButton("Reload")
    self.reloadButton.toolTip = "Reload this module."
    reloadFormLayout.addWidget(self.reloadButton)
    self.reloadButton.connect('clicked()', self.onReload)

    # reload and test button
    # (use this during development, but remove it when delivering
    #  your module to users)
    self.reloadAndTestButton = qt.QPushButton("Reload and Test")
    self.reloadAndTestButton.toolTip = "Reload this module and then run the self tests."
    reloadFormLayout.addWidget(self.reloadAndTestButton)
    self.reloadAndTestButton.connect('clicked()', self.onReloadAndTest)

    # Collapsible button
    optionsCollapsibleButton = ctk.ctkCollapsibleButton()
    optionsCollapsibleButton.text = "Render Options"
    self.layout.addWidget(optionsCollapsibleButton)

    # Layout within the options collapsible button
    optionsFormLayout = qt.QFormLayout(optionsCollapsibleButton)

    # volume selector
    self.volumeSelector = slicer.qMRMLNodeComboBox()
    self.volumeSelector.nodeTypes = ( "vtkMRMLScalarVolumeNode", "" )
    self.volumeSelector.selectNodeUponCreation = False
    self.volumeSelector.addEnabled = False
    self.volumeSelector.noneEnabled = True
    self.volumeSelector.removeEnabled = False
    self.volumeSelector.showHidden = False
    self.volumeSelector.showChildNodeTypes = False
    self.volumeSelector.setMRMLScene( slicer.mrmlScene )
    self.volumeSelector.setToolTip( "Pick the volume to render" )
    optionsFormLayout.addRow("Render Volume:", self.volumeSelector)

    # render button
    self.renderButton = qt.QPushButton("Render")
    self.renderButton.toolTip = "Perform the OpenCL Render."
    optionsFormLayout.addWidget(self.renderButton)
    self.renderButton.connect('clicked(bool)', self.onRenderButtonClicked)

    # Add vertical spacer
    self.layout.addStretch(1)

  def onReload(self,moduleName="RenderCL"):
    """Generic reload method for any scripted module.
    ModuleWizard will subsitute correct default moduleName.
    """
    globals()[moduleName] = slicer.util.reloadScriptedModule(moduleName)

  def onReloadAndTest(self,moduleName="RenderCL"):
    try:
      self.onReload()
      evalString = 'globals()["%s"].%sTest()' % (moduleName, moduleName)
      tester = eval(evalString)
      tester.runTest()
    except Exception, e:
      import traceback
      traceback.print_exc()
      qt.QMessageBox.warning(slicer.util.mainWindow(),
          "Reload and Test", 'Exception!\n\n' + str(e) + "\n\nSee Python Console for Stack Trace")

  def enter(self):
    try:
      import pyopencl
    except ImportError:
      qt.QMessageBox.warning(slicer.util.mainWindow(), "RenderCL", "No OpenCL for you!\nInstall pyopencl in slicer's python installation.\nAnd, you'll also need to be sure you have OpenCL compatible hardware and software.")

  def onRenderButtonClicked(self):
    volumeNode = self.volumeSelector.currentNode()
    if not volumeNode:
      qt.QMessageBox.warning(slicer.util.mainWindow(), "RenderCL", "No volume selected")
      return
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    renderWindow = threeDView.renderWindow()
    renderWindowSize = tuple(renderWindow.GetSize())

    if not self.logic:
      self.logic = RenderCLLogic(volumeNode, renderSize=renderWindowSize, imageViewer=self.imageViewer)

    self.logic.render()


class CLContext(object):
  """An opencl context on a device.
  Interacts with pyopencl, but doesn't know about mrml.
  """

  def __init__(self,devicePreference='GPU'):
    """
    devicePreference is 'CPU' or 'GPU'
    """

    import os
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    #
    # Find the context
    #
    self.context = None
    for platform in pyopencl.get_platforms():
        for device in platform.get_devices():
            if pyopencl.device_type.to_string(device.type) == devicePreference:
               self.context = pyopencl.Context([device])
               print ('Using context %s' % devicePreference)
               break;

    if not self.context:
      self.context = pyopencl.create_some_context()

    if not self.context:
      raise "No OpenCL context available!"

    self.queue = pyopencl.CommandQueue(self.context)

  def compile(self,programPath,mapping=None):
    """Takes the path to an opencl program and compiles it for the context.
    If a mapping is provided, it should be a dictionary of string to string
    mappings to be applied to the program text before compiling.
    """
    if not self.context:
      return None

    fp = open(programPath)
    sourceIn = fp.read()
    fp.close()

    if mapping:
      source = sourceIn % mapping
    return pyopencl.Program(self.context, source).build()


class CLVolume(object):
  """Manage memory for a mrml volume node and all related attributes
  such as the transform and volume rendering properties"""

  def __init__(self,clContext,volumeNode):
    self.clContext = clContext
    self.volumeNode = volumeNode
    self.volumeImage_dev = None
    self.updateDevice()

  def updateDevice(self):
    """Pass currently selected volume to device.
    It goes in as an RGBA image even though it's a slicer scalar volume
    TODO: make this a one component image of the native type
    """
    volumeArray = slicer.util.array(self.volumeNode.GetID())
    num_channels = 4
    shape = volumeArray.shape
    a = numpy.zeros(shape + (num_channels,)).astype(numpy.float32)

    #print(a)
    print(a.shape)
    a[:,:,:,0] = volumeArray * .1
    a[:,:,:,1] = volumeArray * .1
    #a[:,:,:,2] = numpy.ones_like(volumeArray) * 128
    a[:,:,:,3] = numpy.ones_like(volumeArray) * 255
    #print(a)
    print(a.shape)
    print(a.max())

    self.volumeImage_dev = pyopencl.image_from_array(self.clContext.context, a, num_channels)

    rasToIJK = vtk.vtkMatrix4x4()
    self.volumeNode.GetRASToIJKMatrix(rasToIJK)

    transformNode = self.volumeNode.GetParentTransformNode()
    if transformNode:
      if transformNode.IsTransformToWorldLinear():
        rasToRAS = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToWorld(rasToRAS)
        rasToRAS.Invert()
        rasToRAS.Multiply4x4(rasToIJK, rasToRAS, rasToIJK)

    rasToIJKArray = numpy.eye(4,dtype=numpy.dtype('float32'))
    for row in range(4):
      for col in range(4):
        rasToIJKArray[row,col] = rasToIJK.GetElement(row,col)

    self.rasToIJK_dev = pyopencl.array.to_device(self.clContext.queue, rasToIJKArray)

    rasBounds = numpy.zeros(6,dtype=numpy.dtype('float32'))
    self.volumeNode.GetRASBounds(rasBounds)
    self.rasBounds_dev = pyopencl.array.to_device(self.clContext.queue, rasBounds)

class RenderCLLogic(object):

  def __init__(self,volumeNode,devicePreference='GPU',renderSize=(1024,1024), imageViewer=None):
    """
    devicePreference is 'CPU' or 'GPU'
    """
    self.volumeNode = volumeNode
    self.volumeArray = slicer.util.array(self.volumeNode.GetID())
    self.renderSize = renderSize
    self.imageViewer = imageViewer

    self.clContext = CLContext(devicePreference)
    self.clVolume = CLVolume(self.clContext, volumeNode)

    inPath = os.path.dirname(slicer.modules.rendercl.path) + "/Render.cl.in"
    mapping = { # TODO: depend on image dimensions and spacing
      'rayStepSize' : '2.f',
      'rayMaxSteps' : '5000',
    }

    self.renderProgram = self.clContext.compile(inPath, mapping)

    #
    # create a 2d array for the render buffer
    #
    self.renderArray = numpy.zeros(self.renderSize+(4,) ,dtype=numpy.dtype('ubyte'))
    self.renderArray_dev = pyopencl.array.to_device(self.clContext.queue, self.renderArray)

    self.volumeSampler = pyopencl.Sampler(self.clContext.context,
                              # normalized_coords, addressing_mode, filter_mode
                              False,
                              pyopencl.addressing_mode.NONE,
                              pyopencl.filter_mode.LINEAR)

    #
    # TODO make better 2D image of transfer function
    # for now grayscale and opacity mapped linearly from 0 to 100 (for MRHead)
    #
    num_channels = 4
    mapping = numpy.linspace(0,1,100).astype(numpy.dtype('float32'))
    transfer = numpy.transpose([mapping,]*num_channels).copy()
    self.transferFunctionImage_dev = pyopencl.image_from_array(self.clContext.context, transfer, num_channels)


    self.transferFunctionSampler = pyopencl.Sampler(self.clContext.context,
                              # normalized_coords, addressing_mode, filter_mode
                              False,
                              pyopencl.addressing_mode.NONE,
                              pyopencl.filter_mode.LINEAR)

    self.debugRendering = True
    if self.debugRendering:
      self.renderedImage = vtk.vtkImageData()
      self.renderedImage.SetDimensions(self.renderSize[0], self.renderSize[1], 1)
      self.renderedImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
      if not self.imageViewer:
        self.imageViewer = vtk.vtkImageViewer()
      self.imageViewer.SetColorWindow(255)
      self.imageViewer.SetColorLevel(128)
      self.imageViewer.SetInputData(self.renderedImage)


  def render(self):

    # get the camera parameters from default 3D window
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    renderWindow = threeDView.renderWindow()
    renderers = renderWindow.GetRenderers()
    renderer = renderers.GetItemAsObject(0)
    camera = renderer.GetActiveCamera()

    viewPosition = numpy.array(camera.GetPosition())
    focalPoint = numpy.array(camera.GetFocalPoint())
    viewDistance = numpy.linalg.norm(focalPoint - viewPosition)
    viewNormal = (focalPoint - viewPosition) / viewDistance
    viewUp = numpy.array(camera.GetViewUp())
    viewAngle = camera.GetViewAngle()
    viewRight = numpy.cross(viewNormal,viewUp)

    # make them 4 component
    viewNormal = numpy.append(viewNormal, [0,])
    viewRight = numpy.append(viewRight, [0,])
    viewUp = numpy.append(viewUp, [0,])
    viewPosition = numpy.append(viewPosition, [0,])

    # camera info as view matrix
    viewMatrix = numpy.eye(4,dtype=numpy.dtype('float32'))
    viewMatrix[0] = viewNormal
    viewMatrix[1] = viewRight
    viewMatrix[2] = viewUp
    viewMatrix[3] = viewPosition

    self.viewMatrix_dev = pyopencl.array.to_device(self.clContext.queue, viewMatrix)

    self.renderProgram.deviceRenderRayCast(self.clContext.queue, self.renderSize, None,
        self.renderArray_dev.data,
        numpy.uint32(self.renderSize[0]), numpy.uint32(self.renderSize[1]),
        numpy.float32(1.0), # density
        numpy.float32(1.0), # brightness
        numpy.float32(0.0), # transferOffset
        numpy.float32(1.0), # transferScale
        self.viewMatrix_dev.data,
        numpy.sin(numpy.deg2rad(numpy.float32(viewAngle))),
        self.clVolume.volumeImage_dev,
        self.clVolume.rasToIJK_dev.data,
        self.clVolume.rasBounds_dev.data,
        self.transferFunctionImage_dev,
        self.volumeSampler,
        self.transferFunctionSampler)

    ## Longer term, render and composite with ThreeD view

    renderedImage = self.renderArray_dev.get()

    if self.debugRendering:
      shape = list(self.renderedImage.GetDimensions())
      if shape[-1] == 1:
        shape = shape[:-1]
      components = self.renderedImage.GetNumberOfScalarComponents()
      if components > 1:
        shape.append(components)
      array = vtk.util.numpy_support.vtk_to_numpy(self.renderedImage.GetPointData().GetScalars()).reshape(shape)
      array[:] = renderedImage
      self.renderedImage.Modified()
      self.imageViewer.Render()
      pngWriter = vtk.vtkPNGWriter()
      pngWriter.SetFileName("/Users/pieper/Pictures/renderCLTests/render.png")
      pngWriter.SetInputData(self.renderedImage)
      pngWriter.Write()



class RenderCLTest(unittest.TestCase):
  """
  This is the test case for your scripted module.
  """

  def delayDisplay(self,message,msec=1000):
    """This utility method displays a small dialog and waits.
    This does two things: 1) it lets the event loop catch up
    to the state of the test so that rendering and widget updates
    have all taken place before the test continues and 2) it
    shows the user/developer/tester the state of the test
    so that we'll know when it breaks.
    """
    print(message)
    self.info = qt.QDialog()
    self.infoLayout = qt.QVBoxLayout()
    self.info.setLayout(self.infoLayout)
    self.label = qt.QLabel(message,self.info)
    self.infoLayout.addWidget(self.label)
    qt.QTimer.singleShot(msec, self.info.close)
    self.info.exec_()

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_RenderCL1()

  def test_RenderCL1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests sould exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """
    self.delayDisplay("Starting the test")

    #
    # first, get some sample data
    #
    self.delayDisplay("Get some data")
    import SampleData
    sampleDataLogic = SampleData.SampleDataLogic()
    head = sampleDataLogic.downloadMRHead()

    self.delayDisplay('Finished with download and loading\n')

    w = slicer.modules.RenderCLWidget
    w.volumeSelector.setCurrentNode(head)
    w.onRenderButtonClicked()


    self.delayDisplay('Test passed!')
