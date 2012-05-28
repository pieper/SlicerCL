"""
execfile('/Users/pieper/slicer4/latest/SlicerCL/Experiments/slicercl.py')
execfile('/Users/pieper/Dropbox/hacks/slicercl.py')
execfile('/root/Dropbox/hacks/slicercl.py')
execfile('/home/ubuntu/Dropbox/hacks/slicercl.py')
execfile('d:/Dropbox/hacks/slicercl.py')

"""




import pyopencl as cl
import pyopencl.array as cl_array
import numpy
import numpy.linalg as la

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

#headURI = 'http://www.slicer.org/slicerWiki/images/4/43/MR-head.nrrd'
#labelURI = 'http://boggs.bwh.harvard.edu/tmp/MRHead-label.nrrd'
base = '/Users/pieper/Dropbox/data/slicercl/'
headURI = base + 'MR-head-subvolume-scale_1.nrrd'
labelURI = base + 'MR-head-subvolume-scale_1-label.nrrd'

print("Starting...")
if not slicer.util.getNode('MR-head*'):
  print("Downloading...")
  vl = slicer.modules.volumes.logic()
  name = 'MR-head'
  volumeNode = vl.AddArchetypeVolume(headURI, name, 0)
  name = 'MR-head-label'
  labelNode = vl.AddArchetypeVolume(labelURI, name, 1)
  if volumeNode:
    storageNode = volumeNode.GetStorageNode()
    if storageNode:
      # Automatically select the volume to display
      appLogic = slicer.app.applicationLogic()
      selNode = appLogic.GetSelectionNode()
      selNode.SetReferenceActiveVolumeID(volumeNode.GetID())
      selNode.SetReferenceActiveLabelVolumeID(labelNode.GetID())
      appLogic.PropagateVolumeSelection(1)

node = slicer.util.getNode('MR-head')
volume = slicer.util.array('MR-head')
volumeMax = volume.max()
oneOverVolumeMax = float(1. / volume.max())
labelNode = slicer.util.getNode('MR-head-label')
labelVolume = slicer.util.array('MR-head-label')

print("Creating Context...")
ctx = None
for platform in cl.get_platforms():
    for device in platform.get_devices():
        if cl.device_type.to_string(device.type) == "GPU":
           ctx = cl.Context([device])
           print ("using: %s" % cl.device_type.to_string(device.type))
           break;

if not ctx:
  print ("preferred context not available")
  ctx = cl.create_some_context()
print("Creating Queue...")
queue = cl.CommandQueue(ctx)

print("Copying volumes...")
mf = cl.mem_flags
volume_dev = cl_array.to_device(queue, volume)
label_dev = cl.array.to_device(queue, labelVolume)
binaryLabels = numpy.logical_not(numpy.logical_not(labelVolume))
theta = float(2**15) * numpy.array(binaryLabels,dtype=numpy.dtype('float32'))
theta_dev = cl.array.to_device(queue,theta)
thetaNext = numpy.copy(numpy.array(theta, dtype=numpy.dtype('float32')))
thetaNext_dev = cl.array.to_device(queue,thetaNext)
labelNext_dev = cl_array.empty_like(label_dev)
candidates = labelVolume.copy()
candidates_dev = cl.array.to_device(queue,candidates)
candidatesNext = candidates.copy()
candidatesNext_dev = cl.array.to_device(queue,candidatesNext)
candidatesInitialized = False

print("label mean ", labelVolume.mean())
print("label_dev mean ", label_dev.get().mean())
print("labelNext_dev mean ", labelNext_dev.get().mean())
print("candidates_dev mean ", candidates_dev.get().mean())
print("theta mean ", theta.mean())
print("thetaNext_dev mean ", thetaNext_dev.get().mean())
print("candidatesNext_dev mean ", candidatesNext_dev.get().mean())

print("Building program...")
slices,rows,columns = volume.shape
source = """

    #define SLICES %(slices)d
    #define ROWS %(rows)d
    #define COLUMNS %(columns)d

    __kernel void clearShort(
                    __global short destination[SLICES][ROWS][COLUMNS] )
    {
      size_t slice = get_global_id(2);
      size_t column = get_global_id(1);
      size_t row = get_global_id(0);

      if (slice < SLICES && row < ROWS && column < COLUMNS)
      {
        destination[slice][row][column] = 0;
      }
    }

    __kernel void copyShort(
                    __global short source[SLICES][ROWS][COLUMNS],
                    __global short destination[SLICES][ROWS][COLUMNS] )
    {
      size_t slice = get_global_id(2);
      size_t column = get_global_id(1);
      size_t row = get_global_id(0);

      if (slice < SLICES && row < ROWS && column < COLUMNS)
      {
        destination[slice][row][column] = source[slice][row][column];
      }
    }

    __kernel void copyFloat(
                    __global float source[SLICES][ROWS][COLUMNS],
                    __global float destination[SLICES][ROWS][COLUMNS] )
    {
      size_t slice = get_global_id(2);
      size_t column = get_global_id(1);
      size_t row = get_global_id(0);

      if (slice < SLICES && row < ROWS && column < COLUMNS)
      {
        destination[slice][row][column] = source[slice][row][column];
      }
    }

    __kernel void copyDouble(
                    __global double source[SLICES][ROWS][COLUMNS],
                    __global double destination[SLICES][ROWS][COLUMNS] )
    {
      size_t slice = get_global_id(2);
      size_t column = get_global_id(1);
      size_t row = get_global_id(0);

      if (slice < SLICES && row < ROWS && column < COLUMNS)
      {
        destination[slice][row][column] = source[slice][row][column];
      }
    }

    static inline void setNeighbors(
            __global short volume[SLICES][ROWS][COLUMNS],
            size_t slice, size_t row, size_t column, 
            short value )
    {
      int size = 1;
      int sliceOff, rowOff, columnOff;
      unsigned int sampleSlice, sampleRow, sampleColumn;
      for (sliceOff = -size; sliceOff <= size; sliceOff++)
      {
        sampleSlice = slice + sliceOff;
        if (sampleSlice < 0 || sampleSlice >= SLICES) continue;
        for (rowOff = -size; rowOff <= size; rowOff++)
        {
        sampleRow = row + rowOff;
        if (sampleRow < 0 || sampleRow >= ROWS) continue;
        {
          for (columnOff = -size; columnOff <= size; columnOff++)
          {
            sampleColumn = column + columnOff;
            if (sampleColumn < 0 || sampleColumn >= COLUMNS) continue;
            // set the value of the volume at the neighbor location
            volume[sampleSlice][sampleRow][sampleColumn] = value;
            }
          }
        }
      }
    }

    __kernel void initialCandidates(
                    __global short labels[SLICES][ROWS][COLUMNS],
                    __global short candidates[SLICES][ROWS][COLUMNS] 
                    )
    {

      size_t slice = get_global_id(2);
      size_t column = get_global_id(1);
      size_t row = get_global_id(0);

      if (slice >= SLICES || row >= ROWS || column >= COLUMNS)
      {
        return;
      }

      if ( labels[slice][row][column] ) 
      {
        setNeighbors(candidates, slice, row, column, 1);
      }
    }

    __kernel void growCut(
                    __global short volume[SLICES][ROWS][COLUMNS],
                    __global short label[SLICES][ROWS][COLUMNS],
                    __global float theta[SLICES][ROWS][COLUMNS],
                    __global float thetaNext[SLICES][ROWS][COLUMNS],
                    __global short labelNext[SLICES][ROWS][COLUMNS],
                    __global short candidates[SLICES][ROWS][COLUMNS],
                    __global short candidatesNext[SLICES][ROWS][COLUMNS],
                    short volumeMax )
    {
      size_t slice = get_global_id(2);
      size_t column = get_global_id(1);
      size_t row = get_global_id(0);

      if (slice >= SLICES || row >= ROWS || column >= COLUMNS)
      {
        return;
      }

      int size = 1;

      int sliceOff, rowOff, columnOff;
      unsigned int sampleSlice, sampleRow, sampleColumn;

      // copy over current to Next on the assumption that nothing will change
      labelNext[slice][row][column] = label[slice][row][column];
      float thetaNow = theta[slice][row][column]; 
      thetaNext[slice][row][column] = thetaNow;
      short sample = volume[slice][row][column]; 


      if ( candidates[slice][row][column] == 0 )
      {
        return;
      }

      short otherSample, otherLabel;
      float otherTheta, sampleDiff;
      float attackStrength;

      for (sliceOff = -size; sliceOff <= size; sliceOff++)
      {
        sampleSlice = slice + sliceOff;
        if (sampleSlice < 0 || sampleSlice >= SLICES) continue;
        for (rowOff = -size; rowOff <= size; rowOff++)
        {
        sampleRow = row + rowOff;
        if (sampleRow < 0 || sampleRow >= ROWS) continue;
          for (columnOff = -size; columnOff <= size; columnOff++)
          {
            sampleColumn = column + columnOff;
            if (sampleColumn < 0 || sampleColumn >= COLUMNS) continue;

            otherLabel = label[sampleSlice][sampleRow][sampleColumn];
            if (otherLabel != 0)
            {
              otherSample = volume[sampleSlice][sampleRow][sampleColumn];
              otherTheta = theta[sampleSlice][sampleRow][sampleColumn];
              sampleDiff = sample - otherSample;
              if (sampleDiff < 0) sampleDiff *= -1;
              attackStrength = otherTheta * ( 1 - ( sampleDiff / volumeMax ) );
              if (attackStrength < 0) attackStrength = -1 * attackStrength;
              if ( attackStrength > thetaNow ) 
              {
                labelNext[slice][row][column] = otherLabel;
                thetaNext[slice][row][column] = attackStrength;
                thetaNow = attackStrength;
                setNeighbors( candidatesNext, slice, row, column, 1 );
              }
            }
          }
        }
      }
    }
    """ % {'slices' : slices, 'rows' : rows, 'columns' : columns}
prg = cl.Program(ctx, source).build()

def iterate(iterations=10):
  print("Running!")
  for iteration in xrange(iterations):
    print('---------------before----------')
    #print("labelNext_dev mean ", labelNext_dev.get().mean())
    #print("candidates_dev mean ", candidates_dev.get().mean())
    #print("thetaNext_dev mean ", thetaNext_dev.get().mean())
    #print("candidatesNext_dev mean ", candidatesNext_dev.get().mean())
    prg.growCut(queue, volume.shape, None, 
        volume_dev.data, label_dev.data, theta_dev.data, 
        thetaNext_dev.data, labelNext_dev.data, 
        candidates_dev.data, candidatesNext_dev.data,
        volumeMax).wait()
    prg.copyShort(queue, volume.shape, None, labelNext_dev.data, label_dev.data).wait()
    prg.copyShort(queue, volume.shape, None, candidatesNext_dev.data, candidates_dev.data).wait()
    prg.clearShort(queue, volume.shape, None, candidatesNext_dev.data).wait()
    prg.copyFloat(queue, theta.shape, None, thetaNext_dev.data, theta_dev.data).wait()
    #print('---------------after----------')
    #print("labelNext_dev mean ", labelNext_dev.get().mean())
    #print("candidates_dev mean ", candidates_dev.get().mean())
    #print("thetaNext_dev mean ", thetaNext_dev.get().mean())
    #print("candidatesNext_dev mean ", candidatesNext_dev.get().mean())

  print("Getting data...")
  labelVolume[:] = labelNext_dev.get()
  print("Rendering...")
  labelNode.GetImageData().Modified()
  labelNode.Modified()

  print("Done!")

def growCut(iterations=10):
  global candidatesInitialized
  if not candidatesInitialized:
    print("Initializing Candidates")
    prg.clearShort(queue, volume.shape, None, candidatesNext_dev.data).wait()
    print("candidatesNext_dev mean ", candidatesNext_dev.get().mean())
    prg.initialCandidates(queue, candidates.shape, None, label_dev.data, candidates_dev.data).wait()
    print("label_dev mean ", label_dev.get().mean())
    print("candidates_dev mean ", candidates_dev.get().mean())
    candidatesInitialized = True
  print("Candidates Ready")

  for iteration in xrange(iterations):
    iterate(1)
    slicer.app.processEvents()
    print("iteration %d" % iteration)
  print("growCut done")


# {{{

def dilate():
  #headURI = 'http://www.slicer.org/slicerWiki/images/4/43/MR-head.nrrd'
  #labelURI = 'http://boggs.bwh.harvard.edu/tmp/MRHead-label.nrrd'
  base = '/tmp/hoot/'
  headURI = base + 'MR-head.nrrd'
  labelURI = base + 'MR-head-label.nrrd'

  print("Starting...")
  if not slicer.util.getNode('MR-head*'):
    print("Downloading...")
    vl = slicer.modules.volumes.logic()
    name = 'MR-head'
    volumeNode = vl.AddArchetypeVolume(headURI, name, 0)
    name = 'MR-head-label'
    labelNode = vl.AddArchetypeVolume(labelURI, name, 1)
    if volumeNode:
      storageNode = volumeNode.GetStorageNode()
      if storageNode:
        # Automatically select the volume to display
        appLogic = slicer.app.applicationLogic()
        selNode = appLogic.GetSelectionNode()
        selNode.SetReferenceActiveVolumeID(volumeNode.GetID())
        selNode.SetReferenceActiveLabelVolumeID(labelNode.GetID())
        appLogic.PropagateVolumeSelection(1)

  node = slicer.util.getNode('MR-head')
  volume = slicer.util.array('MR-head')
  oneOverVolumeMax = 1. / volume.max()
  labelNode = slicer.util.getNode('MR-head-label')
  labelVolume = slicer.util.array('MR-head-label')

  print("Creating Context...")
  ctx = None
  for platform in cl.get_platforms():
      for device in platform.get_devices():
          print(cl.device_type.to_string(device.type))
          if cl.device_type.to_string(device.type) == "GPU":
             ctx = cl.Context([device])
             break;

  if not ctx:
    print ("no GPU context available")
    ctx = cl.create_some_context()
  print("Creating Queue...")
  queue = cl.CommandQueue(ctx)

  print("Copying volumes...")
  mf = cl.mem_flags
  volume_dev = cl_array.to_device(queue, volume)
  volume_image_dev = cl.image_from_array(ctx, volume,1)
  label_dev = cl.array.to_device(queue, labelVolume)
  theta = numpy.zeros_like(volume)
  theta_dev = cl.array.to_device(queue,theta)
  thetaNext = numpy.zeros_like(volume)
  thetaNext_dev = cl.array.to_device(queue,thetaNext)
  dest_dev = cl_array.empty_like(volume_dev)

  sampler = cl.Sampler(ctx,False,cl.addressing_mode.REPEAT,cl.filter_mode.LINEAR)

  print("Building program...")
  slices,rows,columns = volume.shape
  prg = cl.Program(ctx, """
      #pragma OPENCL EXTENSION cl_khr_fp64: enable

      __kernel void copy(
          __global short source[{slices}][{rows}][{columns}],
          __global short destination[{slices}][{rows}][{columns}])
      {{
        size_t slice = get_global_id(0);
        size_t column = get_global_id(1);
        size_t row = get_global_id(2);

        if (slice < {slices} && row < {rows} && column < {columns})
        {{
          destination[slice][row][column] = source [slice][row][column];
        }}
      }}

      __kernel void dilate(
          __read_only image3d_t volume,
          __global short label[{slices}][{rows}][{columns}],
          sampler_t volumeSampler,
          __global short dest[{slices}][{rows}][{columns}])
      {{
        size_t slice = get_global_id(0);
        size_t column = get_global_id(1);
        size_t row = get_global_id(2);

        if (slice >= {slices} || row >= {rows} || column >= {columns})
        {{
          return;
        }}

        int size = 1;

        int sliceOff, rowOff, columnOff;
        unsigned int sampleSlice, sampleRow, sampleColumn;

        short samples = 0;
        float4 samplePosition;
        for (sliceOff = -size; sliceOff <= size; sliceOff++)
        {{
          sampleSlice = slice + sliceOff;
          if (sampleSlice < 0 || sampleSlice >= {slices}) continue;
          for (rowOff = -size; rowOff <= size; rowOff++)
          {{
          sampleRow = row + rowOff;
          if (sampleRow < 0 || sampleRow >= {rows}) continue;
            for (columnOff = -size; columnOff <= size; columnOff++)
            {{
              sampleColumn = column + columnOff;
              if (sampleColumn < 0 || sampleColumn >= {columns}) continue;
              if (label[sampleSlice][sampleRow][sampleColumn] != 0)
              {{
                samples++;
              }}
            }}
          }}
        }}
        dest[slice][row][column] = samples;
      }}
      """.format(slices=slices,rows=rows,columns=columns)).build()

  def iterate(iterations=10):
    print("Running!")
    for iteration in xrange(iterations):
      prg.dilate(queue, volume.shape, None, volume_image_dev, label_dev.data, sampler, dest_dev.data)
      prg.copy(queue, volume.shape, None, dest_dev.data, label_dev.data)

    print("Getting data...")
    labelVolume[:] = dest_dev.get()
    print("Rendering...")
    labelNode.GetImageData().Modified()
    node.GetImageData().Modified()

    print("Done!")

  def grow(iterations=10):
    for iteration in xrange(iterations):
      iterate(1)
      slicer.app.processEvents()



def imageBlur():
  print("Starting...")
  if not slicer.util.getNode('MRHead*'):
    print("Downloading...")
    vl = slicer.modules.volumes.logic()
    uri = 'http://www.slicer.org/slicerWiki/images/4/43/MR-head.nrrd'
    name = 'MRHead'
    volumeNode = vl.AddArchetypeVolume(uri, name, 0)
    if volumeNode:
      storageNode = volumeNode.GetStorageNode()
      if storageNode:
        # Automatically select the volume to display
        appLogic = slicer.app.applicationLogic()
        selNode = appLogic.GetSelectionNode()
        selNode.SetReferenceActiveVolumeID(volumeNode.GetID())
        appLogic.PropagateVolumeSelection(1)

  node = slicer.util.getNode('MRHead*')
  volume = slicer.util.array('MRHead*')

  print("Creating Context...")
  ctx = None
  for platform in cl.get_platforms():
      for device in platform.get_devices():
          print(cl.device_type.to_string(device.type))
          if cl.device_type.to_string(device.type) == "GPU":
             ctx = cl.Context([device])
             break;

  if not ctx:
    print ("no GPU context available")
    ctx = cl.create_some_context()
  print("Creating Queue...")
  queue = cl.CommandQueue(ctx)

  print("Copying volume...")
  mf = cl.mem_flags
  volume_dev = cl_array.to_device(queue, volume)
  volume_image_dev = cl.image_from_array(ctx, volume,1)
  dest_dev = cl_array.empty_like(volume_dev)

  sampler = cl.Sampler(ctx,False,cl.addressing_mode.REPEAT,cl.filter_mode.LINEAR)

  print("Building program...")
  slices,rows,columns = volume.shape
  prg = cl.Program(ctx, """
      #pragma OPENCL EXTENSION cl_khr_fp64: enable
      __kernel void blur(
          __read_only image3d_t volume,
          sampler_t volumeSampler,
          __global short dest[{slices}][{rows}][{columns}])
      {{
        size_t slice = get_global_id(0);
        size_t column = get_global_id(1);
        size_t row = get_global_id(2);

        int size = 10;

        int sliceOff, rowOff, columnOff;
        unsigned int sampleSlice, sampleRow, sampleColumn;

        float sum = 0;
        unsigned int samples = 0;
        float4 samplePosition;
        int4 sample;
        for (sliceOff = -size; sliceOff <= size; sliceOff++)
        {{
          sampleSlice = slice + sliceOff;
          if (sampleSlice < 0 || sampleSlice >= {slices}) continue;
          for (rowOff = -size; rowOff <= size; rowOff++)
          {{
          sampleRow = row + rowOff;
          if (sampleRow < 0 || sampleRow >= {rows}) continue;
            for (columnOff = -size; columnOff <= size; columnOff++)
            {{
            sampleColumn = column + columnOff;
            if (sampleColumn < 0 || sampleColumn >= {columns}) continue;
            samplePosition.x = sampleColumn;
            samplePosition.y = sampleRow;
            samplePosition.z = sampleSlice;
            sample = read_imagei(volume, volumeSampler, samplePosition);
            //sum += sampleSlice+sampleRow+sampleColumn;
            sum += sample.x;
            samples++;
            }}
          }}
        }}
        dest[slice][row][column] = (short) (sum / samples);
      }}
      """.format(slices=slices,rows=rows,columns=columns)).build()

  print("Running!")
  prg.blur(queue, volume.shape, None, volume_image_dev, label_image_dev, sampler, dest_dev.data)

  print("Getting data...")
  volume[:] = dest_dev.get()
  print("Rendering...")
  node.GetImageData().Modified()

  print("Done!")

def memoryBlur():
	print("Starting...")
	if not slicer.util.getNode('MRHead*'):
	  print("Downloading...")
	  vl = slicer.modules.volumes.logic()
	  uri = 'http://www.slicer.org/slicerWiki/images/4/43/MR-head.nrrd'
	  name = 'MRHead'
	  volumeNode = vl.AddArchetypeVolume(uri, name, 0)
	  if volumeNode:
	    storageNode = volumeNode.GetStorageNode()
	    if storageNode:
	      # Automatically select the volume to display
	      appLogic = slicer.app.applicationLogic()
	      selNode = appLogic.GetSelectionNode()
	      selNode.SetReferenceActiveVolumeID(volumeNode.GetID())
	      appLogic.PropagateVolumeSelection(1)

	node = slicer.util.getNode('MRHead*')
	volume = slicer.util.array('MRHead*')

	print("Creating Context...")
	ctx = None
	for platform in cl.get_platforms():
	    for device in platform.get_devices():
		print(cl.device_type.to_string(device.type))
		if cl.device_type.to_string(device.type) == "GPU":
		   ctx = cl.Context([device])

	if not ctx:
	  print ("no GPU context available")
	  ctx = cl.create_some_context()
	print("Creating Queue...")
	queue = cl.CommandQueue(ctx)

	print("Copying volume...")
	mf = cl.mem_flags
	volume_dev = cl_array.to_device(queue, volume)
	dest_dev = cl_array.empty_like(volume_dev)

	print("Building program...")
	slices,rows,columns = volume.shape
	prg = cl.Program(ctx, """
	    #pragma OPENCL EXTENSION cl_khr_fp64: enable
	    __kernel void blur(
		__global const short volume[{slices}][{rows}][{columns}],
		__global short dest[{slices}][{rows}][{columns}])
	    {{
	      size_t slice = get_global_id(0);
	      size_t column = get_global_id(1);
	      size_t row = get_global_id(2);

	      int size = 3;

	      int sliceOff, rowOff, columnOff;
	      unsigned int sampleSlice, sampleRow, sampleColumn;

	      double sum = 0;
	      unsigned int samples = 0;
	      for (sliceOff = -size; sliceOff <= size; sliceOff++)
	      {{
		sampleSlice = slice + sliceOff;
		if (sampleSlice < 0 || sampleSlice >= {slices}) continue;
		for (rowOff = -size; rowOff <= size; rowOff++)
		{{
		sampleRow = row + rowOff;
		if (sampleRow < 0 || sampleRow >= {rows}) continue;
		  for (columnOff = -size; columnOff <= size; columnOff++)
		  {{
		  sampleColumn = column + columnOff;
		  if (sampleColumn < 0 || sampleColumn >= {columns}) continue;
		  sum += volume[sampleSlice][sampleRow][sampleColumn];
		  samples++;
		  }}
		}}
	      }}
	      dest[slice][row][column] = (short) (sum / samples);
	    }}
	    """.format(slices=slices,rows=rows,columns=columns)).build()

	print("Running!")
	prg.blur(queue, volume.shape, None, volume_dev.data, dest_dev.data)

	print("Getting data...")
	volume[:] = dest_dev.get()
	print("Rendering...")
	node.GetImageData().Modified()

	print("Done!")



def square():
  if not slicer.util.getNode('moving'):
    load_default_volume()

  a = slicer.util.array('moving').flatten()

  ctx = cl.create_some_context()
  queue = cl.CommandQueue(ctx)

  mf = cl.mem_flags
  a_dev = cl_array.to_device(queue, a)
  dest_dev = cl_array.empty_like(a_dev)

  prg = cl.Program(ctx, """
      __kernel void square(__global const short *a, __global short *c)
      {
        int gid = get_global_id(0);
        c[gid] = a[gid] * a[gid];
      }
      """).build()

  prg.square(queue, a.shape, None, a_dev.data, dest_dev.data)

  diff = ( dest_dev - (a_dev*a_dev) ).get()
  norm = la.norm(diff)
  print(norm)

# }}}

# vim: foldmethod=marker
