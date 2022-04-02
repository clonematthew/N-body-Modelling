# Importing libraries
import numpy as np
import matplotlib.pyplot as plt

# Defining constants
G = 6.67e-11

# Function to generate cluster properties
def generateCluster(numberOfStars, clusterRadius, alpha, driftVelocity):

    # Generating a number of random values of x, y and z in a square of radius R
    xCandidates = 2*clusterRadius*np.random.random(numberOfStars*5) - clusterRadius
    yCandidates = 2*clusterRadius*np.random.random(numberOfStars*5) - clusterRadius
    zCandidates = 2*clusterRadius*np.random.random(numberOfStars*5) - clusterRadius

    # Creating empty array for star r positions
    rValues = np.array([], dtype=np.float64)
    xValues = np.array([], dtype=np.float64)
    yValues = np.array([], dtype=np.float64)
    zValues = np.array([], dtype=np.float64)

    # Turning these into array elements
    for i in range(len(xCandidates)):
        # Truth testing if star lies in right range
        if np.sqrt(xCandidates[i]**2+yCandidates[i]**2+zCandidates[i]**2) < clusterRadius:
            # Adding the successful values to the arrays
            rValues = np.append(rValues,[xCandidates[i], yCandidates[i], zCandidates[i]])
            xValues = np.append(xValues, xCandidates[i])
            yValues = np.append(yValues, yCandidates[i])
            zValues = np.append(zValues, zCandidates[i])
        else:
            pass

    # Only taking the first N values of the array
    xValues = xValues[:numberOfStars]
    yValues = yValues[:numberOfStars]
    zValues = zValues[:numberOfStars]

    # Generating the masses of the cluster componenets
    masses = np.random.lognormal(np.log(0.079),0.69, numberOfStars) * 1.989e30
    #masses = np.ones(numberOfStars)

    # Finding the centre of mass of the system
    totalMass = np.sum(masses)
    comX = np.sum(xValues*masses)/totalMass
    comY = np.sum(yValues*masses)/totalMass
    comZ = np.sum(zValues*masses)/totalMass

    # Correcting so that Centre of Masses lies at 0,0,0
    xValuesCentered = xValues - comX
    yValuesCentered = yValues - comY
    zValuesCentered = zValues - comZ

    # Determining GPE of the system
    GPE = 0

    for i in range(numberOfStars):
        ri = np.array([xValuesCentered[i], yValuesCentered[i], zValuesCentered[i]])
        for j in range(numberOfStars):
            if i==j:
                pass
            else:
                rj = np.array([xValuesCentered[j], yValuesCentered[j], zValuesCentered[j]])
                rij = np.linalg.norm(ri-rj)
                GPE += masses[i]*masses[j]*G/rij

    # Generating some random velocities
    vx = 2*np.random.random(numberOfStars) - 1
    vy = 2*np.random.random(numberOfStars) - 1
    vz = 2*np.random.random(numberOfStars) - 1

    # Determining the centre of velocity
    comVX = np.sum(vx*masses)/totalMass
    comVY = np.sum(vy*masses)/totalMass
    comVZ = np.sum(vz*masses)/totalMass

    # Centering the velocities
    vxCentered = vx - comVX
    vyCentered = vy - comVY
    vzCentered = vz - comVZ

    # Determining the KE
    v = vxCentered**2 + vyCentered**2 + vzCentered**2
    KE = np.sum(0.5*masses*v)

    # Finding how KE and G are related
    KEscale = GPE/(alpha*KE)
    
    # Scaling the velocities
    vxCorrected = vx*np.sqrt(KEscale)
    vyCorrected = vy*np.sqrt(KEscale)
    vzCorrected = vz*np.sqrt(KEscale)

    # Working out the veloicty dispersion
    vmag = vxCorrected*vxCorrected + vyCorrected*vyCorrected + vzCorrected*vzCorrected
    vsig = np.sqrt(np.sum(vmag) / numberOfStars)

    # Calculating the crossing time
    tcross = clusterRadius / vsig

    # Adding on a random drift velocity in some direction
    driftX = driftVelocity*(2*np.random.random(1)-1)
    driftY = driftVelocity*(2*np.random.random(1)-1)
    driftZ = driftVelocity*(2*np.random.random(1)-1)

    vxCentered = vxCentered + driftX
    vyCentered = vyCentered + driftY
    vzCentered = vzCentered + driftZ

    # Returning the values
    return xValuesCentered, yValuesCentered, zValuesCentered, vxCorrected, vyCorrected, vzCorrected, masses, tcross

def outputICs(time, maxTime, xpositions, ypositions, zpositions, xvelocities, yvelocities, zvelocities, masses):
    # Defining the name of the file
    fname = "cluster.txt"

    # Opening the file
    with open(fname, "w") as file:
        
        # Writing the headers
        file.write("Time MaxTime Xpos Ypos Zpos Xvel Yvel Zvel Mass \n")

        numberOfObjects = len(xpositions)

        # Looping through each object 
        for i in range(numberOfObjects):
            file.write(str(time) + " " + str(maxTime) + " ")
            file.write(str(xpositions[i]) + " " + str(ypositions[i]) + " " + str(zpositions[i]) + " ")
            file.write(str(xvelocities[i]) + " " + str(yvelocities[i]) + " " + str(zvelocities[i]) + " ")
            file.write(str(masses[i]) + "\n") 

# Generating a line of clusters
def fillamentGenerator(numberOfClusters, clusterSeparation, clusterStarNumber, clusterRadius, alpha, driftVelocity):

    # Generating the offset direction
    clusterDirectionX = 2*np.random.random(numberOfClusters)-1
    clusterDirectionY = 2*np.random.random(numberOfClusters)-1
    clusterDirectionZ = 2*np.random.random(numberOfClusters)-1
    clusterDirections = []

    # Creating the master offset 
    offsets = np.arange(1, numberOfClusters+1, 1)
    offsets = offsets * clusterSeparation

    for i in range(numberOfClusters):
        normalisationConstant = np.sqrt(clusterDirectionX[i]**2 + clusterDirectionY[i]**2 +  clusterDirectionZ[i]**2)
        offset = normalisationConstant * offsets[i]
        clusterDirections.append([offset*clusterDirectionX[i], offset*clusterDirectionY[i], offset*clusterDirectionZ[i]])

    # Defining arrays
    x = []; y = []; z = []; vx = []; vy = []; vz = []; m = []

    # Generating the clusters and offsetting them
    for i in range(numberOfClusters):
        xpos, ypos, zpos, vxs, vys, vzs, ms, _ = generateCluster(clusterStarNumber, clusterRadius, alpha, driftVelocity)
        adjuster = clusterDirections[i]
        for j in range(len(xpos)):
            x.append(xpos[j]+adjuster[0])
            y.append(ypos[j]+adjuster[1])
            z.append(zpos[j]+adjuster[2])
        
            vx.append(vxs[i])
            vy.append(vys[i])
            vz.append(vzs[i])
            m.append(ms[i])

    return x, y, z, vx, vy, vz, m

def generateCylindricalFilament(numberOfClusters, clusterSeparation, clyinderRadius, clusterStarNumber, clusterRadius, alpha, driftVelocity):

    # Setting the central axis values
    clusterPositions = np.arange(0, numberOfClusters, 1)
    zValues = clusterPositions * clusterSeparation

    # Calculating the x and y values
    xValues = 2*np.random.random(numberOfClusters*2) - 1
    yValues = 2*np.random.random(numberOfClusters*2) - 1

    # Getting the ones that lie in the radius we want
    xValuesList = xValues[np.sqrt(xValues**2 + yValues**2) < 1]
    yValuesList = yValues[np.sqrt(xValues**2 + yValues**2) < 1]
    
    # Getting the right number of values
    xValuesChosen = xValuesList * clyinderRadius
    yValuesChosen = yValuesList * clyinderRadius

    # Setting the offsets
    clusterOffsets = []
    for i in range(numberOfClusters):
        clusterOffsets.append([xValuesChosen[i],yValuesChosen[i],zValues[i]])

    # Generating the right number of clusters and positioning them
    x = []; y = []; z = []; vx = []; vy = []; vz = []; m = []; t = []

    for i in range(numberOfClusters):
        xpos, ypos, zpos, vxs, vys, vzs, ms, tc = generateCluster(clusterStarNumber, clusterRadius, alpha, driftVelocity)
        adjuster = clusterOffsets[i]
        t.append(tc)

        for j in range(len(xpos)):
            x.append(xpos[j]+adjuster[0])
            y.append(ypos[j]+adjuster[1])
            z.append(zpos[j]+adjuster[2])
        
            vx.append(vxs[i])
            vy.append(vys[i])
            vz.append(vzs[i])
            m.append(ms[i])

    return x, y, z, vx, vy, vz, m, max(t)
