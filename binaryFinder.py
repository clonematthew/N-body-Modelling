# Function that looks for binaries in the outputs from the hermite integrator

# Importing libraries
import numpy as np

G = np.float64(6.67e-11)    # Newton's Gravitational constant

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
    bestPartner = []
    bestPartnerEnergy = []

    # Looping through every body pair
    for i in range(n):
        # Creating variables for the current best binary index and energy
        currentBestEnergy = 0
        currentBestIndex = n+1

        for j in range(n):
            if i == j:
                pass
            else:
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

                # Testing if this binding energy is bigger than the current lowest
                if bindingEnergies[i][j] < currentBestEnergy:
                    currentBestEnergy = bindingEnergies[i][j]
                    currentBestIndex = j
            
        # Setting the best partner in the list
        bestPartner.append(currentBestIndex)
        bestPartnerEnergy.append(currentBestEnergy)

    # Getting lists to store the best binary and numbers used
    binaryRows = []
    binaryColumns = []
    numbersUsed = []

    # Looping through every body again
    for i in range(n):
        # Checking if the binaries binding energy is negative
        if bestPartnerEnergy[i] < 0:
            # Checking that the best partner of the other body is me as well
            if bestPartner[bestPartner[i]] == i:
                newPairing = True

                # Checking we havent already saved this pair
                for j in range(len(numbersUsed)):
                    # If any of the indicies match those previously used, don't add to binary list
                    if i == numbersUsed[j] or bestPartner[i] == numbersUsed[j]:
                        newPairing = False
                
                # Only adding the binary if this pairing hasn't been added previously
                if newPairing:
                    binaryRows.append(i)
                    binaryColumns.append(bestPartner[i])
                    numbersUsed.append(i)
                    numbersUsed.append(bestPartner[i])

    # Setting up a list for the stars not in a binary
    unpairedList = []
    for u in range(n):
        unpairedList.append(u)

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
        densityBin = 2 / volBinary

        # Checking if binary is much greater in density than the surrounding cluster
        if densityBin < 2*densityClust:
            # Marking under dense binaries for deletion
            binaryRows[p] = "n"
            binaryColumns[p] = "n"
        else:
            unpairedList.remove(binaryRows[p])
            unpairedList.remove(binaryColumns[p])

    # Removing the under dense binaries
    while "n" in binaryRows: 
        binaryRows.remove("n")
        binaryColumns.remove("n")

    return binaryRows, binaryColumns, unpairedList  

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

    # Returning the calculated values
    return bindingEnergies, semiMajorAxes, systemMasses, periods, eccentricities, orbitalAM, massRatios

#t, _, x, y, z, vx, vy, vz, m = np.loadtxt("0.25.txt", delimiter=" ", skiprows=1, unpack=True, dtype=np.float64)
#n = len(x)

def binaryCheck(x, y, z, vx, vy, vz, m, n):
    # First run through of the binary finder to find binaries
    rows, cols, ul = binaryFinder(x, y, z, vx, vy, vz, m, n)

    # Getting the properties of these binaries
    bE, bA, bM, bP, be, bO, bR = binaryPropertyDeterminer(rows, cols, x, y, z, vx, vy, vz, m)

    # New lists to contain the x, y, z etc of starts
    tx = []; ty = []; tz = []; tvx = []; tvy = []; tvz = []; tm = []; tt = []; ti = []

    # Looping through the unpaired object list and adding them to the new list
    for b in ul:
        tx.append(x[b])
        ty.append(y[b])
        tz.append(z[b])
        tvx.append(vx[b])
        tvy.append(vy[b])
        tvz.append(vz[b])
        tm.append(m[b])
        tt.append("s")
        ti.append(b)

    # Creating new stars for the binaries
    for b in range(len(rows)):
        # Centre of mass
        tx.append((x[rows[b]] * m[rows[b]] + x[cols[b]] * m[cols[b]]) / (m[rows[b]] + m[cols[b]]))
        ty.append((y[rows[b]] * m[rows[b]] + y[cols[b]] * m[cols[b]]) / (m[rows[b]] + m[cols[b]]))
        tz.append((z[rows[b]] * m[rows[b]] + z[cols[b]] * m[cols[b]]) / (m[rows[b]] + m[cols[b]]))

        # Centre of velocity
        tvx.append((vx[rows[b]] * m[rows[b]] + vx[cols[b]] * m[cols[b]]) / (m[rows[b]] + m[cols[b]]))
        tvy.append((vy[rows[b]] * m[rows[b]] + vy[cols[b]] * m[cols[b]]) / (m[rows[b]] + m[cols[b]]))
        tvz.append((vz[rows[b]] * m[rows[b]] + vz[cols[b]] * m[cols[b]]) / (m[rows[b]] + m[cols[b]])) 

        # System mass
        tm.append(m[rows[b]] + m[cols[b]])

        # Stating that this is a binary
        tt.append("b")
        ti.append(b)

    # Running the binary finder again but this time using the new list
    rt, ct, _ = binaryFinder(tx, ty, tz, tvx, tvy, tvz, tm, len(tx))

    # Getting the properties of these triples
    if len(rt) > 0:
        tE, tA, tM, tP, te, tO, tR = binaryPropertyDeterminer(rt, ct, tx, ty, tz, tvx, tvy, tvz, tm)
    else:
        tE = tA = tM = tP = te = tO = tR = []

    # Creating a list for the triples
    tripIndex = []
    tripType = []
    for i in range(len(rows)):
        tripIndex.append("N/A")
        tripType.append("N/A")

    # Binaries used in a quadruple system
    bUsed = []

    # Looping through all the binaries
    for t in range(len(rt)):
        # Checking if any member of this binary is already a binary
        if tt[rt[t]] == "b":
            if tt[ct[t]] == "s":
                tripIndex[ti[rt[t]]] = ti[ct[t]]
                tripType[ti[rt[t]]] = "Binary-Star"
            else:
                bi = str(rows[ti[ct[t]]]) + str(cols[ti[ct[t]]])
                tripIndex[ti[rt[t]]] = bi
                tripType[ti[rt[t]]] = "Binary-Binary"
                
        if tt[ct[t]] == "b":
            if tt[rt[t]] == "s":
                tripIndex[ti[ct[t]]] = ti[rt[t]]
                tripType[ti[ct[t]]] = "Binary-Star"
            else:
                bi = str(rows[ti[rt[t]]]) + str(cols[ti[rt[t]]])
                tripIndex[ti[ct[t]]] = bi
                tripType[ti[ct[t]]] = "Binary-Binary"



    # Returning indicies and triple type
    return rows, cols, tripIndex, tripType, bE, bA, bM, bP, be, bO, bR, tE, tA, tM, tP, te, tO, tR
