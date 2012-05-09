#!/bin/bash

#SLICER_SUPERBUILD="${HOME}/slicer4/latest/Slicer-superbuild"
SLICER_SUPERBUILD="${HOME}/slicer4/latest/Slicer4-superbuild"
SLICER_BUILD="${SLICER_SUPERBUILD}/Slicer-build"
PYTHONEXE="${SLICER_SUPERBUILD}/python-build/bin/python"
LAUNCH="${SLICER_BUILD}/Slicer --launcher-no-splash --launch"
PYTHON="${LAUNCH} ${PYTHONEXE}"

OPENCLINSTALL=/usr/local/cuda

tmpdir=`mktemp -d /tmp/slicercl.XXXX`
cd ${tmpdir}
wget http://pypi.python.org/packages/source/P/PyOpenGL/PyOpenGL-3.0.2a5.tar.gz#md5=22108c0ceb222143389da67ec8b2e739
tar xvfz PyOpenGL-3.0.2a5.tar.gz
cd PyOpenGL-3.0.2a5
${PYTHON} setup.py install

cd ${tmpdir}
#wget http://pypi.python.org/packages/source/p/pyopencl/pyopencl-2011.2.tar.gz#md5=db51a29aff5973f3f4a5a52d7ddc7696
#tar xvfz pyopencl-2011.2.tar.gz
#cd pyopencl-2011.2
git clone https://github.com/pieper/pyopencl.git
cd pyopencl
git submodule init
git submodule update
${PYTHON} configure.py --cl-enable-gl \
  #--cl-inc-dir=${OPENCLINSTALL}/include \
  #--cl-lib-dir=${OPENCLINSTALL}/lib \
  #--cl-libname=OpenCL \
  --no-cl-enable-device-fission

export PATH=${SLICER_SUPERBUILD}/python-build/bin:${PATH}

${LAUNCH} /usr/bin/make install
