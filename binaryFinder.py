# Function that looks for binaries in the outputs from the hermite integrator

# Importing libraries
import numpy as np

G = np.float64(6.67e-11)    # Newton's Gravitational constant
eCutoff = np.float64(3)     # Energy Ratio for a Binary

# Definng the binary finder function
def binaryFinder(x, y, z, vx, vy, vz, m, n):
    # Allocating memory for important arrays
    binarySeparations = np.zeros((n, n), dtype=np.float64) 
    reducedMasses = np.zeros((n, n), dtype=np.float64)
    bindingEnergies = np.zeros((n, n), dtype=np.float64)
    comx = np.zeros(n)
    comy = np.zeros(n)
    comz = np.zeros(n)
    
    # Defining arrays to store the indicies of binary pairs
    binaryRows = []
    binaryColumns = []

    # List for looping
    nlist = np.arange(0, n, 1, dtype=np.int64)
    nlist = np.append(nlist, nlist)

    # Looping through every body pair
    for i in range(n):
        for j in nlist[i+1:n]:
            # Calcualting separations
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]

            # Calculating relative velociteis
            dvx = vx[j] - vx[i]
            dvy = vy[j] - vy[i]
            dvz = vz[j] - vz[i]

            # Determining magnitude of distance and velocity
            r = np.sqrt(dx*dx + dy*dy + dz*dz)
            v = np.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)

            # Assigning this to the 2D array
            binarySeparations[i][j] = r

            # Calculating the reduced masses and mass combint
            reducedMasses[i][j] = (m[i] * m[j])/(m[i] + m[j])

            # Calculating the binding energy of the binary
            bindingEnergies[i][j] = 0.5 * reducedMasses[i][j] * v * v - G * m[i] * m[j] / r

            # Determining if this is a binary, and if so adding to the lists
            if bindingEnergies[i][j] < -1e34:
                binaryRows.append(i)
                binaryColumns.append(j)

    # Looping through all the binaries we have found
    for p in range(len(binaryRows)):
        # Finding the CoM of the binary
        comx[p] = (x[binaryRows[p]] * m[binaryRows[p]] + x[binaryColumns[p]] * m[binaryColumns[p]]) / (m[binaryRows[p]] + m[binaryColumns[p]])
        comy[p] = (y[binaryRows[p]] * m[binaryRows[p]] + y[binaryColumns[p]] * m[binaryColumns[p]]) / (m[binaryRows[p]] + m[binaryColumns[p]])
        comz[p] = (z[binaryRows[p]] * m[binaryRows[p]] + z[binaryColumns[p]] * m[binaryColumns[p]]) / (m[binaryRows[p]] + m[binaryColumns[p]])

        # Calculating the distance to every other day 
        dStars = np.sqrt((x - comx[p])**2 + (y - comy[p])**2 + (z - comz[p])**2)

        # Taking the closest 3 stars 
        dStars =  np.sort(dStars)

        # Calculating the distance to the 3rd closest star
        d = dStars[2]

        # Calculating the volume 
        volStars = (4*np.pi*d**3)/3

        # Calculating density of nearest 3 neighbours
        densityClust = 3 / volStars

        # Calculating the density of the binary
        volBinary = (4*np.pi*binarySeparations[binaryRows[p]][binaryColumns[p]]**3)
        densityBin = 2/ volBinary

        # Checking if binary is much greater in density than the surrounding cluster
        if densityBin < 2*densityClust:
            # Deleting binaries from the list that dont meet this criteria
            binaryRows = np.delete(binaryRows, binaryRows[p])
            binaryColumns = np.delete(binaryColumns, binaryColumns[p])

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
    massRatios = np.zeros(n)

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

        # Calculating the mass ratio (always biggest over smallest)
        if m[binaryRows[k]] > m[binaryColumns[k]]:
            massRatios[k] = m[binaryRows[k]] / m[binaryColumns[k]]
        else:
            massRatios[k] = m[binaryColumns[k]] / m[binaryRows[k]]

    print(bindingEnergies, semiMajorAxes, systemMasses, periods, eccentricities, orbitalAM, massRatios)

    # Returning the calculated values
    return bindingEnergies, semiMajorAxes, systemMasses, periods, eccentricities, orbitalAM, massRatios

t, _, x, y, z, vx, vy, vz, m = np.loadtxt("0.75.txt", delimiter=" ", skiprows=1, unpack=True, dtype=np.float64)
n = len(x)

rows, columns = binaryFinder(x, y, z, vx, vy, vz, m, n)
    
binaryPropertyDeterminer(rows, columns, x, y, z, vx, vy, vz, m)
