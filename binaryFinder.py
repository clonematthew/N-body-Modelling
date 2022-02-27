# Function that looks for binaries in the outputs from the hermite integrator

# Importing libraries
import numpy as np

G = np.float64(6.67e-11)    # Newton's Gravitational constant
eCutoff = np.float64(3)     # Energy Ratio for a Binary

# Definng the binary finder function
def binaryFinder(x, y, z, vx, vy, vz, m, n):
    # Allocating memory for important arrays
    binarySeparations = np.zeros((n, n), dtype=np.float64) 
    massCombinations = np.zeros((n, n), dtype=np.float64)
    gravEnergies = np.zeros((n, n), dtype=np.float64)
    energyRatio = np.zeros((n, n), dtype=np.float64)
    kintEnergies = np.zeros(n, dtype=np.float64)
    
    # Defining arrays to store the indicies of binary pairs
    binaryRows = []
    binaryColumns = []

    # Calculating kinetic energies
    kintEnergies = 0.5 * m * (vx*vx + vy*vy + vz*vz)

    # List for looping
    nlist = np.arange(0, n, 1, dtype=np.int64)
    nlist = np.append(nlist, nlist)

    # Looping through every body pair
    for i in range(n):
        for j in nlist[i:i+n]:
            # Calcualting separations
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]

            # Determining magnitude of distance
            r = np.sqrt(dx*dx + dy*dy + dz*dz)

            # Assigning this to the 2D array
            binarySeparations[i][j] = r
            massCombinations[i][j] = m[i] * m[j]

            # Calculating gravtiational potential for this pair
            gravEnergies[i][j] = G * massCombinations[i][j] / binarySeparations[i][j]
        
            # Calculating the energy ratio of this pair
            energyRatio[i][j] = gravEnergies[i][j] / (kintEnergies[i])

            # Determining if this is a binary, and if so adding to the lists
            if energyRatio[i][j] > eCutoff:
                binaryRows.append(i)
                binaryColumns.append(j)

    return binaryRows, binaryColumns

def binaryPropertyDeterminer(binaryRows, binaryColumns, x, y, z, vx, vy, vz, m):
    # The number of binaries
    n = len(binaryRows)

    # Creating the lists for the binary properties
    bindingEnergies = np.zeros(n)
    reducedMasses = np.zeros(n)
    semiMajorAxes = np.zeros(n)
    systemMasses = np.zeros(n)
    periods = np.zeros(n)
    eccentricities = np.zeros(n)
    orbitalAM = np.zeros(n)

    # Looping through all the binaries
    for k in range(n):
        # Getting the relative positions and velocities
        dx = x[binaryRows[k]] - x[binaryColumns[k]]
        dy = y[binaryRows[k]] - y[binaryColumns[k]]      
        dz = z[binaryRows[k]] - z[binaryColumns[k]]

        dvx = vx[binaryRows[k]] - vx[binaryColumns[k]]
        dvy = vy[binaryRows[k]] - vy[binaryColumns[k]]
        dvz = vz[binaryRows[k]] - vz[binaryColumns[k]]

        # Finding the magnitude 
        dv = np.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)

        # Finding the relative distance 
        dr = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Finding the dot product between the relative r and v
        rdv = dx*dvx + dy*dvy + dz*dvz

        # Calculating the reduced mass
        reducedMasses[k] = (m[binaryRows[k]] * m[binaryColumns[k]])/(m[binaryRows[k]] + m[binaryColumns[k]])
    
        # Calculating the binding energy
        bindingEnergies[k] = 0.5 * reducedMasses[k] * dv*dv - G * m[binaryRows[k]] * m[binaryColumns[k]] / dr

        # Calculating the semi-major axis
        semiMajorAxes[k] = -G * m[binaryRows[k]] * m[binaryColumns[k]] / (2*bindingEnergies[k])

        # Calculating the system mass
        systemMasses[k] = m[binaryRows[k]] + m[binaryColumns[k]]

        # Calculating the period of the orbit
        periods[k] = 2 * np.pi * np.sqrt(semiMajorAxes[k]**3/(G*systemMasses[k])) 

        # Calculating the eccentricity
        eccentricities[k] = np.sqrt((1 - dr/semiMajorAxes[k])**2 + rdv**2/(semiMajorAxes[k]*G*systemMasses[k]))

        # Calculatinf the orbital angular momentum
        orbitalAM[k] = np.sqrt(G*semiMajorAxes[k]*(1-eccentricities[k]**2)/systemMasses[k])*m[binaryRows[k]]*m[binaryColumns[k]]

    # Returning the calculated values
    return bindingEnergies, semiMajorAxes, systemMasses, periods, eccentricities, orbitalAM

t, _, x, y, z, vx, vy, vz, m = np.loadtxt("1.00.txt", delimiter=" ", skiprows=1, unpack=True, dtype=np.float64)
n = len(x)

rows, columns = binaryFinder(x, y, z, vx, vy, vz, m, n)
    
binaryPropertyDeterminer(rows, columns, x, y, z, vx, vy, vz, m)
