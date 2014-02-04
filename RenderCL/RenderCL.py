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

    # reload button
    self.reloadButton = qt.QPushButton("Reload")
    self.reloadButton.toolTip = "Reload this module."
    self.layout.addWidget(self.reloadButton)
    self.reloadButton.connect('clicked(bool)', self.onReload)

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

  def enter():
    try:
      import pyopencl
    except ImportError:
      qt.QMessageBox.warning(slicer.util.mainWindow(), "RenderCL", "No OpenCL for you!\nInstall pyopencl in slicer's python installation.\nAnd, you'll also need to be sure you have OpenCL compatible hardware and software.")

  def onReload(self):
    import imp, sys, os
    filePath = slicer.modules.rendercl.path
    p = os.path.dirname(filePath)
    if not sys.path.__contains__(p):
      sys.path.insert(0,p)

    mod = "RenderCL"
    fp = open(filePath, "r")
    globals()[mod] = imp.load_module(mod, fp, filePath, ('.py', 'r', imp.PY_SOURCE))
    fp.close()

    globals()['r'] = r = globals()[mod].RenderCLWidget()

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
    invViewMatrix = numpy.eye(4,dtype=numpy.dtype('float32'))
    self.invViewMatrix_dev = pyopencl.array.to_device(self.queue, invViewMatrix)


    if False:
      # create a 3d image from the volume
      num_channels = 2
      volumeImage = (self.volumeArray[:], self.volumeArray[:], num_channels)
      self.volumeImage_dev = pyopencl.image_from_array(self.ctx, self.volumeArray, num_channels, mode = "r", norm_int = True)
      origin = (0,0,0)
      region = tuple(numpy.asarray(self.volumeArray.shape) - numpy.asarray( (1,1,1) ))
      #pyopencl.enqueue_fill_image(self.queue, self.volumeImage_dev, 128, 0, region, wait_for=None)
      print("max ------------------")
      print(self.volumeArray.max())

    # odd sample code that seems to work
    num_channels = 2
    shape = self.volumeArray.shape
    a = numpy.random.random(shape + (num_channels,)).astype(numpy.float32)

    print(a)
    print(a.shape)
    a[:,:,:,0] = numpy.ones_like(self.volumeArray) * 255
    a[:,:,:,1] = self.volumeArray
    print(a)
    print(a.shape)
    print(a.max())

    self.volumeImage_dev = pyopencl.image_from_array(self.ctx, a, num_channels)

    # create a 2d array for the render buffer
    self.renderArray = numpy.zeros(self.renderSize+(4,) ,dtype=numpy.dtype('ubyte'))
    self.renderArray_dev = pyopencl.array.to_device(self.queue, self.renderArray)

    self.volumeSampler = pyopencl.Sampler(self.ctx,False,
                              pyopencl.addressing_mode.CLAMP,
                              pyopencl.filter_mode.LINEAR)

    # TODO make better 2D image of transfer function
    # for now grayscale and opacity mapped linearly from 0 to 100 (for MRHead)
    num_channels = 4
    mapping = numpy.linspace(0,1,100).astype(numpy.dtype('float32'))
    transfer = numpy.transpose([mapping,]*num_channels).copy()
    self.transferFunctionImage_dev = pyopencl.image_from_array(self.ctx, transfer, num_channels)


    self.transferFunctionSampler = pyopencl.Sampler(self.ctx,False,
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
      pngWriter.SetFileName("/tmp/render.png")
      pngWriter.SetInput(self.renderedImage)
      pngWriter.Write()

      print(self.renderedImage.GetScalarRange())
