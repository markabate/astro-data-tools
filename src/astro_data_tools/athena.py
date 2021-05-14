'''athena.py: Module with methods for handling Athena data.

Athena++ is a astrophysical magnetohydrodynamics (MHD) code written in C++. 
The code simulates an astrophysical plasma as a 3-dimensional mesh which 
evolves over time. At user-defined time-steps, the program outputs a snapshot
of the mesh as an HDF5 dataset containing the physical properties of each cell,
i.e. the density, velocity, pressure, and electric charge of the gas, as well 
as metadata, such as the time-step. Problem parameters, such as the initial and 
boundary conditions, are set in an input file written by the user. 

The input file and outputted HDF5 datasets are contained in a single directory,
which I refer to as the "Athena output directory". A computational experiment
will usually involve several simulation runs to demonstrate the effects of 
adjusting different parameters, with each run having its own directory with 
several snapshot datasets. This module simplifies the process of reading data
for these directories, so that a computational scientist can quickly access
all of the data for their experiment without worrying about the specifics of
Athena's data storage method.

The Athena++ source code can be accessed at https://github.com/PrincetonUniversity/athena-public-version
'''

import numpy as np
import h5py
import re
import os

# Module-level constants, accessed in dataset as "x[var[0]][()][var[1], mbNum]",
# where "x" is the name of the HDF5 dataset, "var" is the name of the desired
# variable (D, VX, VY, VZ, MX, MY, or MZ), and "mbNum" is the meshblock number. 
# For example, the density mesh would be accessed as "x[ D[0] ] [()] [ D[1], mbNum ]"
# (spaces are added for clarity).
D = ['prim', 0]  # density
VX = ['prim', 1]  # x velocity
VY = ['prim', 2]  # y velocity
VZ = ['prim', 3]  # z velocity
MX = ['cons', 1]  # x momentum
MY = ['cons', 2]  # y momentum
MZ = ['cons', 3]  # z momentum


class AthenaDataDirectory:
    '''Class representing an Athena output directory. All the functionality
    for this class is contained in the module-level functions, but creating 
	an AthenaDataDirectory object avoids reading the input file multiple times.

    It is implied that all functions are operating on directories. This means
    that the following functions are equivalent, but with different parameter
    requirements:
    AthenaDataDirectory.convertNumToFileName -> convertNumToFileName
    AthenaDataDirectory.loadHDF5             -> loadHDF5FromDirectory
    AthenaDataDirectory.loadFullMesh         -> loadFullMeshFromDirectory

    Attributes
    ----------
    absDirName : string
            Absolute directory name.
    shortDirName : string
            Short directory name.
    inputFileName : string
            Absolute name for athinput file.
    parameterDict : dictionary
            Contains all parameters from athinput file.
    meshblocksDim : int[3]
            List of meshblocks in x, y, and z.
    outputBaseName : string
    outputId : string
    outputType : string
    outputSuffix : string
    '''

    def __init__(self, dirName, outputType='hdf5', outputSuffix='athdf'):
        '''Initializes instance variables.'''
        self.absDirName = os.path.abspath(dirName)
        self.shortDirName = os.path.split(self.absDirName)[1]
        self.inputFileName = self.absDirName + '/' + \
            'athinput.' + self.shortDirName.replace('_', '')
        self.parameterDict = loadInputFile(self.inputFileName)
        self.meshblocksDim = getMeshblocksDim(
            self.absDirName, parameterDict=self.parameterDict)

        self.outputBaseName = self.parameterDict['job']['problem_id']
        self.outputID = 'out1'  # default to out1
        self.outputType = outputType
        self.outputSuffix = outputSuffix

        # finds number of hdf5 output
        for header in self.parameterDict.keys():
            if 'output' in header:
                if self.parameterDict[header]['file_type'] == self.outputType:
                    self.outputID = header.replace('put', '')
                    break

    def __str__(self):
        '''String representation.'''
        return "AthenaDataDirectory object: " + self.shortDirName

    def convertNumToFileName(self, fileNumList):
        '''Converts number to hdf5 filename.'''
        padding = 5

        if type(fileNumList) == list:
            fileNameList = []
            for i in fileNumList:
                fileNameList.append(self.absDirName + '/' + self.outputBaseName
                                    + '.' + self.outputID + '.'
                                    + str(i).zfill(padding) + '.' + self.outputSuffix)
            return fileNameList
        else:
            return self.absDirName + '/' + self.outputBaseName + '.' + self.outputID \
                + '.' + str(fileNumList).zfill(padding) + \
                '.' + self.outputSuffix

    def loadHDF5(self, fileNum):
        '''Loads hdf5 dataset. For a list of file numbers, use loadHDF5Iter.'''
        fileName = self.convertNumToFileName(fileNum)
        return loadHDF5(fileName)

    def loadFullMesh(self, fileNum, vars):
        '''Loads full meshes for variables in vars. For a list of file numbers, 
	use loadFullMeshIter.
	'''
        fileName = self.convertNumToFileName(fileNum)
        if type(vars[0]) == list:
            meshes = []
            for v in vars:
                meshes.append(loadFullMesh(fileName, v))
            if len(vars) == 1:
                return meshes[0]
            else:
                return meshes
        else:
            return loadFullMesh(fileName, vars)

    def loadHDF5Iter(self, fileNumList):
        '''Calls module level function.'''
        return loadHDF5FromDirectoryIter(self.absDirName, fileNumList, parameterDict=self.parameterDict)

    def loadFullMeshIter(self, fileNumList, vars):
        '''Calls module-level function.'''
        return loadFullMeshFromDirectoryIter(self.absDirName, fileNumList, vars, parameterDict=self.parameterDict)


# Module-level functions for loading data.

def loadInputFile(fileOrDirName):
    '''Reads Athena input file into a dictionary. The variable fileName can
    either be the name of a directory or an input file (the function assumes 
    names not beginning with "athinput." are directory names). By convention,
    underscores are not included in input file names. The input file associated 
    with the directory 'athena_test' would be athena_test/athinput.athenatest
    '''
    parameterDict = {}
    inputFileName = fileOrDirName

    if 'athinput.' not in fileOrDirName:
        absDirName = os.path.abspath(fileOrDirName)
        shortDirName = os.path.split(absDirName)[1]
        inputFileName = absDirName + '/' + \
            'athinput.' + shortDirName.replace('_', '')

    currentHeader = "null"
    with open(inputFileName, 'r') as file:
        for line in file:
            snippedLine = line.split('#')[0].strip('\n')
            if re.search('<.*>', snippedLine) != None:
                currentHeader = re.sub(
                    '>.*', '', re.sub('.*<', '', snippedLine))
            elif '=' in snippedLine:
                pair = snippedLine.split('=', 1)
                key = pair[0].strip()
                val = pair[1].strip()

                if currentHeader not in parameterDict.keys():
                    parameterDict[currentHeader] = {}

                try:
                    parameterDict[currentHeader][key] = float(val)
                except ValueError:
                    parameterDict[currentHeader][key] = val

    return parameterDict

def loadHDF5(fileName):
    '''Loads HDF5 file.'''
    dataset = h5py.File(fileName, 'r')
    return dataset

def loadFullMesh(fileName, var):
    '''Loads Athena mesh from file for the given variable and meshblock 
    dimensions.

    Parameters
    ----------
    var : list
            Variables to load from the HDF5 dataset. Must be list of a combination
            of the module-level variables athena.D, VX, VY, VZ, MX, MY, or MZ.
    meshblocksDim : list[3]
            Number of meshblocks in x, y, and z.
    '''
    hdf5File = h5py.File(fileName, 'r')
    fullMesh = buildFullMesh(hdf5File, var)
    hdf5File.close()
    return fullMesh

def loadHDF5FromDirectoryIter(dirName, fileNumList, parameterDict=None):
    '''Yields iterator over hdf5 datasets, corresponding to the files with
    indices in fileNumList. Doesn't build meshblocks into full grid.
    '''
    fileNames = convertNumToFileName(
        dirName, fileNumList, parameterDict=parameterDict)

    for fileName in fileNames:
        yield loadHDF5(fileName)

def loadFullMeshFromDirectoryIter(dirName, fileNumList, vars, parameterDict=None):
    '''Yields iterator over a list 3D numpy arrays, corresponding to the files
    with indices in fileNumList. Extracts the variables in vars and builds a
    full mesh. The elements of the yielded list are the separate meshes for
    each variable, in the same order they are given in vars.
    '''
    meshblocksDim = getMeshblocksDim(dirName, parameterDict=parameterDict)
    fileNames = convertNumToFileName(
        dirName, fileNumList, parameterDict=parameterDict)

    for fileName in fileNames:
        if type(vars[0]) == list:
            meshes = []
            for v in vars:
                meshes.append(loadFullMesh(fileName, v))
            if len(vars) == 1:
                yield meshes[0]
            else:
                yield meshes
        else:
            yield loadFullMesh(fileName, vars)

def getMeshblocksDim(fileOrDirName, parameterDict=None):
    '''Gets meshblocks dimensions from an input file or directory. Builds a
    parameterDict if none is passed.
    '''
    if parameterDict is None:
        parameterDict = loadInputFile(inputFile)

    meshblocksNX = int(
        parameterDict['mesh']['nx1'] / parameterDict['meshblock']['nx1'])
    meshblocksNY = int(
        parameterDict['mesh']['nx2'] / parameterDict['meshblock']['nx2'])
    meshblocksNZ = int(
        parameterDict['mesh']['nx3'] / parameterDict['meshblock']['nx3'])
    return [meshblocksNX, meshblocksNY, meshblocksNZ]

def convertNumToFileName(dirName, fileNumList, fileType='hdf5', parameterDict=None):
    '''Converts a directory name and number to the name of the corresponding
    HDF5 file. If fileNum is a list, returns a list of the HDF5 file names.
    Builds a parameterDict if none is passed.
    '''
    if parameterDict is None:
        parameterDict = loadInputFile(dirName)

    padding = 5
    absDirName = os.path.abspath(dirName)
    shortDirName = os.path.split(absDirName)[1]
    inputFile = absDirName + '/' + 'athinput.' + shortDirName.replace('_', '')
    outputID = 'out1'
    basename = parameterDict['job']['problem_id']

    for header in parameterDict.keys():
        if 'output' in header:
            if parameterDict[header]['file_type'] == fileType:
                outputID = header.replace('put', '')
                break
    else:
        raise Exception('Error in athenaConvertNumToFileName: No HDF5 output.')

    if type(fileNumList) == list:
        fileNameList = []
        for i in fileNumList:
            fileNameList.append(absDirName + '/' + basename + '.' + outputID + '.'
                                + str(i).zfill(padding) + '.athdf')
        return fileNameList
    else:
        return absDirName + '/' + basename + '.' + outputID + '.' + str(fileNumList).zfill(padding) \
            + '.athdf'


# Miscellaneous data handling.

def buildFullMeshFromHDF5(hdf5File, var):
    '''Builds full mesh using an Athena HDF5 dataset. Originally the HDF5
    dataset is segmented into separate meshblocks.
    '''
    logicalLocations = hdf5File['LogicalLocations'][()
                                                    ]  # read in local locations
    locList = list(logicalLocations)
    mbMaxIndices = list(np.amax(locList, axis=0))

    locListFull = []
    for i, x in enumerate(locList):
        # elements of form [meshblockNum, logicalLoc]
        locListFull.append([i, x])

    locListFull.sort(key=lambda x: x[1][0])
    locListFull.sort(key=lambda x: x[1][1])
    locListFull.sort(key=lambda x: x[1][2])

    xSlice = None
    ySlice = None
    fullMesh = None
    for x in locListFull:
        mbNum = x[0]
        loc = x[1]

        if xSlice is None:
            xSlice = hdf5File[var[0]][()][var[1], mbNum]
        else:
            xSlice = np.concatenate(
                (xSlice, hdf5File[var[0]][()][var[1], mbNum]), axis=2)

        if loc[0] == mbMaxIndices[0]:
            if ySlice is None:
                ySlice = xSlice
            else:
                ySlice = np.concatenate((ySlice, xSlice), axis=1)
            xSlice = None

            if loc[1] == mbMaxIndices[1]:
                if fullMesh is None:
                    fullMesh = ySlice
                else:
                    fullMesh = np.concatenate((fullMesh, ySlice), axis=0)
                ySlice = None

    return fullMesh

def buildFullMeshFromMeshBlocks(meshblocks, logicalLocations):
    '''Builds full mesh using the meshblocks and logical locations.'''
    locList = list(logicalLocations)
    mbMaxIndices = list(np.amax(locList, axis=0))

    locListFull = []
    for i, x in enumerate(locList):
        # elements of form [meshblockNum, logicalLoc]
        locListFull.append([i, x])

    locListFull.sort(key=lambda x: x[1][0])
    locListFull.sort(key=lambda x: x[1][1])
    locListFull.sort(key=lambda x: x[1][2])

    xSlice = None
    ySlice = None
    fullMesh = None
    for x in locListFull:
        mbNum = x[0]
        loc = x[1]

        if xSlice is None:
            xSlice = meshblocks[mbNum]
        else:
            xSlice = np.concatenate((xSlice, meshblocks[mbNum]), axis=2)

        if loc[0] == mbMaxIndices[0]:
            if ySlice is None:
                ySlice = xSlice
            else:
                ySlice = np.concatenate((ySlice, xSlice), axis=1)
            xSlice = None

            if loc[1] == mbMaxIndices[1]:
                if fullMesh is None:
                    fullMesh = ySlice
                else:
                    fullMesh = np.concatenate((fullMesh, ySlice), axis=0)
                ySlice = None

    return fullMesh

def meshCoords(N, L=1.0):
    '''Volume centered coordinates for a square mesh centered at (0,0,0) with
    side length L and size NxNxN.
    '''
    return np.linspace(-0.5*L + L*0.5/float(N), 0.5*L - L*0.5/float(N), N)


# Space for module testing.

def main():
    pass

if __name__ == "__main__":
    main()
