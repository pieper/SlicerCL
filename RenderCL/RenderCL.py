import os
import unittest
from __main__ import vtk, qt, ctk, slicer

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

  def setup(self):
    # Instantiate and connect widgets ...

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

  def enter():
    try:
      import pyopencl
    except ImportError:
      qt.QMessageBox.warning(slicer.util.mainWindow(), "RenderCL", "No OpenCL for you!\nInstall pyopencl in slicer's python installation.\nAnd, you'll also need to be sure you have OpenCL compatible hardware and software.")

  def onRenderButtonClicked(self):
    volumeNode = self.volumeSelector.currentNode()
    if not volumeNode:
      qt.QMessageBox.warning(slicer.util.mainWindow(), "RenderCL", "No volume selected")
      return
    self.logic = RenderCLLogic(volumeNode)
    self.logic.render()


class RenderCLLogic(object):
  def __init__(self,volumeNode,contextPreference='GPU',renderSize=(512,512)):
    self.volumeNode = volumeNode
    self.volumeArray = slicer.util.array(self.volumeNode.GetID())
    self.renderSize = renderSize

    try:
      import pyopencl
      import pyopencl.array
      import numpy
    except ImportError:
      raise "No OpenCL for you!\nInstall pyopencl in slicer's python installation."

    import os
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    self.ctx = None
    for platform in pyopencl.get_platforms():
        for device in platform.get_devices():
            if pyopencl.device_type.to_string(device.type) == contextPreference:
               self.ctx = pyopencl.Context([device])
               break;

    if not self.ctx:
      self.ctx = pyopencl.create_some_context()
    self.queue = pyopencl.CommandQueue(self.ctx)

    inPath = os.path.dirname(slicer.modules.rendercl.path) + "/Render.cl.in"
    fp = open(inPath)
    sourceIn = fp.read()
    fp.close()

    source = sourceIn % { # TODO: depend on image dimensions and spacing
        'rayStepSize' : '0.01f',
        'rayMaxSteps' : '500',
        }
    self.prg = pyopencl.Program(self.ctx, source).build()

    # camera info
    # TODO: get camera params
    invViewMatrix = numpy.eye(4,dtype=numpy.dtype('float32'))
    self.invViewMatrix_dev = pyopencl.array.to_device(self.queue, invViewMatrix)

    # pass currently selected volume to device
    num_channels = 2
    shape = self.volumeArray.shape
    a = numpy.zeros(shape + (num_channels,)).astype(numpy.float32)

    #print(a)
    print(a.shape)
    a[:,:,:,0] = numpy.ones_like(self.volumeArray) * 255
    a[:,:,:,1] = self.volumeArray
    #print(a)
    print(a.shape)
    print(a.max())

    self.volumeImage_dev = pyopencl.image_from_array(self.ctx, a, num_channels)

    # create a 2d array for the render buffer
    self.renderArray = numpy.zeros(self.renderSize+(4,) ,dtype=numpy.dtype('ubyte'))
    self.renderArray_dev = pyopencl.array.to_device(self.queue, self.renderArray)

    self.volumeSampler = pyopencl.Sampler(self.ctx,
                              # normalized_coords, addressing_mode, filter_mode
                              True,
                              pyopencl.addressing_mode.CLAMP,
                              pyopencl.filter_mode.LINEAR)

    # TODO make better 2D image of transfer function
    # for now grayscale and opacity mapped linearly from 0 to 100 (for MRHead)
    num_channels = 4
    mapping = numpy.linspace(0,1,100).astype(numpy.dtype('float32'))
    transfer = numpy.transpose([mapping,]*num_channels).copy()
    self.transferFunctionImage_dev = pyopencl.image_from_array(self.ctx, transfer, num_channels)


    self.transferFunctionSampler = pyopencl.Sampler(self.ctx,
                              # normalized_coords, addressing_mode, filter_mode
                              False,
                              pyopencl.addressing_mode.REPEAT,
                              pyopencl.filter_mode.LINEAR)

    self.debugRendering = True
    if self.debugRendering:
      self.renderedImage = vtk.vtkImageData()
      self.renderedImage.SetDimensions(self.renderSize[0], self.renderSize[1], 1)
      self.renderedImage.SetNumberOfScalarComponents(4)
      self.renderedImage.SetScalarTypeToUnsignedChar()
      self.renderedImage.AllocateScalars()
      #self.imageViewer = vtk.vtkImageViewer()
      #self.imageViewer.SetColorWindow(255)
      #self.imageViewer.SetColorLevel(128)
      #self.imageViewer.SetInput(self.renderedImage)


  def render(self):
    print("Building program...")

    import numpy
    self.prg.deviceRender(self.queue, self.renderSize, None,
        self.renderArray_dev.data,
        numpy.uint32(self.renderSize[0]), numpy.uint32(self.renderSize[1]),
        numpy.float32(1.0), # density
        numpy.float32(1.0), # brightness
        numpy.float32(0.0), # transferOffset
        numpy.float32(1.0), # transferScale
        self.invViewMatrix_dev.data,
        self.volumeImage_dev,
        self.transferFunctionImage_dev,
        self.volumeSampler,
        self.transferFunctionSampler)

    # TODO: put the renderArray into a png file for rendering
    ## Longer term, render and composite with ThreeD view

    renderedImage = self.renderArray_dev.get()

    if self.debugRendering:
      print(renderedImage.shape)
      print(renderedImage.min(), renderedImage.max())

      import vtk.util.numpy_support
      shape = list(self.renderedImage.GetDimensions())
      if shape[-1] == 1:
        shape = shape[:-1]
      shape.reverse()
      components = self.renderedImage.GetNumberOfScalarComponents()
      if components > 1:
        shape.append(components)
      array = vtk.util.numpy_support.vtk_to_numpy(self.renderedImage.GetPointData().GetScalars()).reshape(shape)
      array[:] = renderedImage
      self.renderedImage.Modified()
      #self.imageViewer.Render()
      pngWriter = vtk.vtkPNGWriter()
      pngWriter.SetFileName("/Users/pieper/Pictures/renderCLTests/render.png")
      pngWriter.SetInput(self.renderedImage)
      pngWriter.Write()

      print(self.renderedImage.GetScalarRange())


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
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
      )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        print('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        print('Loading %s...\n' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading\n')

    volumeNode = slicer.util.getNode(pattern="FA")

    self.logic = RenderCLLogic(volumeNode)
    self.logic.render()

    self.delayDisplay('Test passed!')
