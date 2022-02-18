# 4th Order Hermite Integrator for 3rd Year Physics Project
# Simulates the orbital dynamics of stars, planets etc

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm

# Defining constants
G = np.float64(6.67e-11)    # Newton's Gravitational constant
s = np.float64(6e7)         # Softening, gravity "blurring"
e = np.float64(0.05)        # Eta, timestep accuracy parameter
a = np.float64(1.3)         # Arseth stability criterion
sixth = np.float64(1.0/6.0) # One over Six, for speed

# Function that determines accelerations
@jit(nopython=True)
def accelerations(x, y, z, vx, vy, vz, m, n):
    # Resetting and allocating memory
    ax = np.zeros(n, dtype=np.float64)
    ay = np.zeros(n, dtype=np.float64)
    az = np.zeros(n, dtype=np.float64)
    adx = np.zeros(n, dtype=np.float64)
    ady = np.zeros(n, dtype=np.float64)
    adz = np.zeros(n, dtype=np.float64)
    gpe = np.zeros(n, dtype=np.float64)

    # First loop through every body
    for i in range(n):
        # Creating a list of body indicies that doesn't include i
        blist = np.arange(0, n, 1, dtype=np.int64)
        blist = np.delete(blist, i)

        # Setting GPE for this body to zero
        gpei = np.float64(0)

        # Looping through all other bodies
        for j in blist:
            # Resetting gpe
            gpeij = np.float64(0)

            # Allocating variable memory
            dx = np.float64(0)
            dy = np.float64(0)
            dz = np.float64(0)
            dvx = np.float64(0)
            dvy = np.float64(0)
            dvz = np.float64(0)
            r = np.float64(0)
            vdr = np.float64(0)
            dterm3 = np.float64(0)
            dterm5 = np.float64(0)

            # Calculating position and velocity vectors
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz = z[j] - z[i]

            dvx = vx[j] - vx[i]
            dvy = vy[j] - vy[i]
            dvz = vz[j] - vz[i]

            # Finding rij and rij dot vij (squared)
            r = np.sqrt(dx*dx + dy*dy + dz*dz + s*s)
            vdr = dx*dvx + dy*dvy + dz*dvz

            # Calculating divison terms
            dterm3 = 1.0/(r*r*r)
            dterm5 = 1.0/(r*r*r*r*r)

            # Calculating the accelerations
            ax[i] = ax[i] + (G * m[j]) * (dx*dterm3)
            ay[i] = ay[i] + (G * m[j]) * (dy*dterm3)
            az[i] = az[i] + (G * m[j]) * (dz*dterm3)

            adx[i] = adx[i] + (G * m[j]) * ((dvx*dterm3) + 3.0*(vdr*dx*dterm5))
            ady[i] = ady[i] + (G * m[j]) * ((dvy*dterm3) + 3.0*(vdr*dy*dterm5))
            adz[i] = adz[i] + (G * m[j]) * ((dvz*dterm3) + 3.0*(vdr*dz*dterm5))

            # Calculating gravitational potential
            gpeij = -1.0 * G * m[i] * m[j] / r
            gpei = gpei + gpeij

        # Setting the gpe of this body after the loop
        gpe[i] = gpei

    # Returning acceleration etc
    return ax, ay, az, adx, ady, adz, gpe

# Function to calculate the timestep
@jit(nopython=True)
def nextTimeStep(ax, ay, az, adx, ady, adz, addx, addy, addz, adddx, adddy, adddz, n, dt):
    # List for timesteps & setting old and new dt values
    dtList = []
    newdt = np.float64(0)
    dt = np.float64(dt)

    # Looping through all bodies to determine their timestep
    for b in range(n):
        # Calculating values of a, ad, add, addd
        a = np.sqrt(ax[b]*ax[b] + ay[b]*ay[b] + az[b]*az[b])
        ad = np.sqrt(adx[b]*adx[b] + ady[b]*ady[b] + adz[b]*adz[b])
        add = np.sqrt(addx[b]*addx[b] + addy[b]*addy[b] + addz[b]*addz[b])
        addd = np.sqrt(adddx[b]*adddx[b] + adddy[b]*adddy[b] + adddz[b]*adddz[b])

        # Calculating timestep
        dtList.append(np.sqrt(e * ((a*add + ad**2)/(ad*addd + add**2))))

    # Preventing too large timestep changes
    if min(dtList)/dt > a:
        newdt = dt*a
    else:
        newdt = min(dtList)

    # Returning the new timestep
    return newdt

# Function that initialises the integrator 
def init(dataFile):
    # Importing the ICs from the file
    t, _, x, y, z, vx, vy, vz, m = np.loadtxt(dataFile, delimiter=" ", skiprows=1, unpack=True, dtype=np.float64)

    # Determining the number of bodies
    n = len(x)

    # Setting up the arrays for acceleration
    axp = ayp = azp = adxp = adyp = adzp = np.zeros(n, dtype=np.float64)
    xadd = yadd = zadd = xaddd = yaddd = zaddd = np.zeros(n, dtype=np.float64)
    axc = ayc = azc = adxc = adyc = adzc = np.zeros(n, dtype=np.float64)

    # Setting up arrays for position and velocity
    xp = yp = zp = vxp = vyp = vzp = np.zeros(n, dtype=np.float64)
    xc = yc = zc = vxc = vyc = vzc = np.zeros(n, dtype=np.float64)

    # Setting up arrays for plotting # TODO: Delete
    xarr = yarr = zarr = np.zeros((n, 1), dtype=np.float64)
    tarr = dtarr = np.zeros(1, dtype=np.float64)
    K = np.zeros((n, 1), dtype=np.float64)
    G = np.zeros((n, 1), dtype=np.float64)

    # Only choosing the 1st value from the time array
    t = t[0]

    # Returning all the arrays
    return t, x, y, z, vx, vy, vz, m, n, axp, ayp, azp, adxp, adyp, adzp, xadd, yadd, zadd, xaddd, yaddd, zaddd, axc, ayc, azc, adxc, adyc, adzc, xp, yp, zp, vxp, vyp, vzp, xc, yc, zc, vxc, vyc, vzc, xarr, yarr, zarr, tarr, dtarr, K, G

# Function that actually runs the integrator
def hermiteIntegrator(dt, dynamicTimestep, outputNumber, dataFile, tmax):
    # Initialising
    t, x, y, z, vx, vy, vz, m, n, axp, ayp, azp, adxp, adyp, adzp, xadd, yadd, zadd, xaddd, yaddd, zaddd, axc, ayc, azc, adxc, adyc, adzc, xp, yp, zp, vxp, vyp, vzp, xc, yc, zc, vxc, vyc, vzc, xarr, yarr, zarr, tarr, dtarr, K, G = init(dataFile)

    # Setting up counters and progress bar etc
    outputInterval = np.float64(tmax / outputNumber)
    outputCounter = 0.0
    progBar = tqdm(total=tmax)

    # Setting number of times to predict correct
    pecOrder = 3

    # The main while loop of the function
    while t < tmax:
        # Calculating dt powers 
        dt = np.float64(dt)
        dt2 = dt*dt
        dt3 = dt*dt*dt

        # Prediction step
        axp, ayp, azp, adxp, adyp, adzp, _ = accelerations(x, y, z, vx, vy, vz, m, n)

        xp = x + vx*dt + axp*dt2*0.5 + adxp*dt3*sixth
        yp = y + vy*dt + ayp*dt2*0.5 + adyp*dt3*sixth
        zp = z + vz*dt + azp*dt2*0.5 + adzp*dt3*sixth

        vxp = vx + dt*axp + 0.5*adxp*dt2
        vyp = vy + dt*ayp + 0.5*adyp*dt2
        vzp = vz + dt*azp + 0.5*adzp*dt2

        # Start the loop through for correcting
        for it in range(pecOrder):
            # Using predicited p and v for first loop, corrected for all else
            if it == 0:
                axc, ayc, azc, adxc, adyc, adzc, gp = accelerations(xp, yp, zp, vxp, vyp, vzp, m, n)
            else:
                axc, ayc, azc, adxc, adyc, adzc, gp = accelerations(xc, yc, zc, vxc, vyc, vzc, m, n)

            # Determining higher order corrections 
            xadd = np.float64(-6.0)*(axp - axc) - dt*(np.float64(4.0)*adxp + np.float64(2.0)*adxc)
            yadd = np.float64(-6.0)*(ayp - ayc) - dt*(np.float64(4.0)*adyp + np.float64(2.0)*adyc)
            zadd = np.float64(-6.0)*(azp - azc) - dt*(np.float64(4.0)*adzp + np.float64(2.0)*adzc)

            xaddd = np.float64(12.0)*(axp - axc) + np.float64(6.0)*dt*(adxp + adxc)
            yaddd = np.float64(12.0)*(ayp - ayc) + np.float64(6.0)*dt*(adyp + adyc)
            zaddd = np.float64(12.0)*(azp - azc) + np.float64(6.0)*dt*(adzp + adzc)

            # Adding the corrections 
            xc = xp + dt2*xadd/np.float64(24.0) + dt2*xaddd/np.float64(120.0)
            yc = yp + dt2*yadd/np.float64(24.0) + dt2*yaddd/np.float64(120.0)
            zc = zp + dt2*zadd/np.float64(24.0) + dt2*zaddd/np.float64(120.0)

            vxc = vxp + dt*xadd/np.float64(6.0) + dt*xaddd/np.float64(24.0)
            vyc = vyp + dt*yadd/np.float64(6.0) + dt*yaddd/np.float64(24.0)
            vzc = vzp + dt*zadd/np.float64(6.0) + dt*zaddd/np.float64(24.0)

        # Updating the positions and velocities with the final solution
        x = xc; y = yc; z = zc; vx = vxc; vy = vyc; vz = vzc

        # Calculating the KE and GPE
        ke = np.float64(0.5) * m * (vx*vx + vy*vy + vz*vz)
        K = np.append(K, np.array_split(ke, n), axis=1)
        G = np.append(G, np.array_split(gp, n), axis=1)

        # Updating the timestep
        if dynamicTimestep:
            dt = nextTimeStep(axc, ayc, azc, adxc, adyc, adzc, xadd, yadd, zadd, xaddd, yaddd, zaddd, n, dt)
        
        # Ticking the clock
        t = t + dt
        outputCounter = outputCounter + dt

        # Outputing based on the given time interval
        if outputCounter > outputInterval:
            # Update progress bar
            progBar.update(outputCounter)

            # Resetting counter
            outputCounter = 0

            # Appendng to arrays
            xarr = np.append(xarr, np.array_split(x, n), axis=1)
            yarr = np.append(yarr, np.array_split(y, n), axis=1)
            zarr = np.append(zarr, np.array_split(z, n), axis=1)
            dtarr = np.append(dtarr, dt)
            tarr = np.append(tarr, t)

    # Closing the progress bar when complete
    progBar.close()

    # Returning the arrays
    return G, K, xarr, yarr, zarr, dtarr, tarr

''' Main Program Call '''

# Asking for simulation parameters
textFile = input("Name of the data file: ")
initialdt = input("Time-step: ")
dynamicTimeStep = input("Dynamic Time Step? (Y or N): ")
outputNumber = input("Number of Outputs: ")

if dynamicTimeStep == "Y":
    dyn = True
else:
    dyn = False

print("Maximum Time Selection \n")
timeUnit = input("Unit of Time (Gyear, Myear, kYear, Year, Month, Week, Day): ")
timeNumber = input("Number of Time Units: ")

if timeUnit == "Gyear":
    timeUnit = 1e9 * 365*24*60*60
elif timeUnit == "Myear":
    timeUnit = 1e6 * 365*24*60*60
elif timeUnit == "kYear":
    timeUnit = 1e3 * 365*24*60*60
elif timeUnit == "Year":
    timeUnit = 365*24*60*60
elif timeUnit == "Month":
    timeUnit = 30*24*60*60
elif timeUnit == "Week":
    timeUnit = 7*24*60*60
else:
    timeUnit = 24*60*60*365

maximumTime = np.float64(timeUnit) * np.float64(timeNumber)

# Calling the hermite function
g, k, x, y, z, dts, t = hermiteIntegrator(np.int64(initialdt), dyn, np.int64(outputNumber), str(textFile), np.int64(maximumTime))

# Getting the shape of the arrays
arrayShape = g.shape
bodyAmount = arrayShape[0]
valuesAmount = arrayShape[1]

# Summing energies
totalKinetic = np.zeros(valuesAmount)
totalGravity = np.zeros(valuesAmount)

# Looping through to sum
for i in range(bodyAmount):
    totalKinetic = totalKinetic + k[i]
    totalGravity = totalGravity + g[i]

# Getting the total energy
totalEnergy = totalKinetic + totalGravity

# Getting the initial energy
initialEnergy = totalEnergy[1]

# Finding the % change
percentChange = (totalEnergy-initialEnergy)/initialEnergy

# Plotting the energy evolution
plt.figure(figsize=(10,10))
plt.plot(percentChange[1:])
plt.show()

plt.figure(figsize=(10,10))
plt.plot(x[0][1:], y[0][1:])
plt.show()

plt.figure(figsize=(10,10))
plt.plot(dts[1:])
plt.show()