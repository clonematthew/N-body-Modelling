''' A fully automatic integrator for my physics project '''

# External dependencies
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Defining important constants
G = np.float64(6.67e-11)
softening = np.float64(6e8)
eta = np.float64(0.1)

# Calculating useful fractions
half = np.float64(1./2.)
sixth = np.float64(1./6.)
twentyFourth = np.float64(1./24.)
oneHundredAndTwentyth = np.float64(1./120.)

# Function to determine accelerations
@jit(nopython=True)
def accelerations(x, y, z, vx, vy, vz, m, n):
    # Allocating memory for the accelerations and adots
    ax = np.zeros(n, dtype=np.float64)
    ay = np.zeros(n, dtype=np.float64)
    az = np.zeros(n, dtype=np.float64)

    adx = np.zeros(n, dtype=np.float64)
    ady = np.zeros(n, dtype=np.float64)
    adz = np.zeros(n, dtype=np.float64)

    # ALlocating memory for the GPE
    GPEs = np.zeros(n, dtype=np.float64)

    # Beginning the first loop to go through every body in the simulation
    for bodyIndex in range(n):
        # Creating a list of body indicies that doesn't include the current bodies index
        bodyIndicies = np.arange(0, n, 1, np.int64)
        bodyIndicies = np.delete(bodyIndicies, bodyIndex)

        # Resetting the GPE
        GPEi = np.float64(0)

        # Looping through every body that isn't this one
        for otherBodyIndex in bodyIndicies:
            # Resetting the GPE
            GPEij = np.float64(0)

            # Calculating the position and velocity vectors
            rijx = x[bodyIndex] - x[otherBodyIndex]
            rijy = y[bodyIndex] - y[otherBodyIndex]
            rijz = z[bodyIndex] - z[otherBodyIndex]

            vijx = vx[bodyIndex] - vx[otherBodyIndex]
            vijy = vy[bodyIndex] - vy[otherBodyIndex]
            vijz = vz[bodyIndex] - vz[otherBodyIndex]

            # Calculating the magnitudes of the vectors
            rij = np.sqrt(rijx*rijx + rijy*rijy + rijz*rijz)

            # Finding the dot product of the velocity and position vectors
            rijvijdot = rijx*vijx + rijy*vijy + rijz*vijz

            # Calculating the divison terms used to calculate a and adot
            divisonTerm = rij*rij + softening*softening
            divisonTermThreeTwo = divisonTerm**(3/2)
            divisonTermFiveTwo = divisonTerm**(5/2)

            # Calculating acceleration
            ax[bodyIndex] = ax[bodyIndex] + (-G * m[otherBodyIndex]) * (rijx/divisonTermThreeTwo)
            ay[bodyIndex] = ay[bodyIndex] + (-G * m[otherBodyIndex]) * (rijy/divisonTermThreeTwo)
            az[bodyIndex] = az[bodyIndex] + (-G * m[otherBodyIndex]) * (rijz/divisonTermThreeTwo)

            # Calculating adot
            adx[bodyIndex] = adx[bodyIndex] + (G * m[otherBodyIndex]) * ((vijx/divisonTermThreeTwo) + 3*(rijvijdot*rijx/divisonTermFiveTwo))
            ady[bodyIndex] = ady[bodyIndex] + (G * m[otherBodyIndex]) * ((vijy/divisonTermThreeTwo) + 3*(rijvijdot*rijy/divisonTermFiveTwo))
            ady[bodyIndex] = adz[bodyIndex] + (G * m[otherBodyIndex]) * ((vijz/divisonTermThreeTwo) + 3*(rijvijdot*rijz/divisonTermFiveTwo))

            # Calculating the gravitational potential
            GPEij = -G * m[bodyIndex] * m[otherBodyIndex] / rij

            # Adding this to the total GPE for the current body
            GPEi = GPEi + GPEij

        # Allocating the final GPE for this body
        GPEs[bodyIndex] = GPEi

    # Returning all the calculated values
    return ax, ay, az, adx, ady, adz, GPEs

# Function to set the timestep according to the Hermite Scheme
@jit(nopython=True)
def hermiteTimeStep(xacel, yacel, zacel, xadot, yadot, zadot, xa2dot, ya2dot, za2dot, xa3dot, ya3dot, za3dot, numberOfBodies):
    # Creating an empty list for the prospective timesteps
    dtList = []

    # Looping through each body to determine the timestep for that body
    for body in range(numberOfBodies):

        # Calculating the scalar for a, adot, a2dot and a3dot
        a = np.sqrt(xacel[body]*xacel[body] + yacel[body]*yacel[body] + zacel[body]*zacel[body])
        ad = np.sqrt(xadot[body]*xadot[body] + yadot[body]*yadot[body] + zadot[body]*zadot[body])
        add = np.sqrt(xa2dot[body]*xa2dot[body] + ya2dot[body]*ya2dot[body] + za2dot[body]*za2dot[body])
        addd = np.sqrt(xa3dot[body]*xa3dot[body] + ya3dot[body]*ya3dot[body] + za3dot[body]*za3dot[body])

        # Calculating the timestep
        dtList.append(np.sqrt(eta * ((a*add + ad**2)/(ad*addd + add**2))))

    # Returning the smallest timestep for usage
    return min(dtList)

# The function that initialises the integrator from a data file input
def initIntegrator(dataFile):
    # Importing initial conditions
    timeKeeper, maxTime, xpos, ypos, zpos, xvel, yvel, zvel, mass = np.loadtxt(dataFile, delimiter=" ", skiprows=1, unpack=True, dtype=np.float64)

    # Getting the number of bodies
    numberOfBodies = len(xpos)

    # Setting up arrays
    xacel = np.zeros(numberOfBodies, dtype=np.float64)
    yacel = np.zeros(numberOfBodies, dtype=np.float64)
    zacel = np.zeros(numberOfBodies, dtype=np.float64)

    xadot = np.zeros(numberOfBodies, dtype=np.float64)
    yadot = np.zeros(numberOfBodies, dtype=np.float64)
    zadot = np.zeros(numberOfBodies, dtype=np.float64)

    xa2dot = np.zeros(numberOfBodies, dtype=np.float64)
    ya2dot = np.zeros(numberOfBodies, dtype=np.float64)
    za2dot = np.zeros(numberOfBodies, dtype=np.float64)

    xa3dot = np.zeros(numberOfBodies, dtype=np.float64)
    ya3dot = np.zeros(numberOfBodies, dtype=np.float64)
    za3dot = np.zeros(numberOfBodies, dtype=np.float64)

    xacel0 = np.zeros(numberOfBodies, dtype=np.float64)
    yacel0 = np.zeros(numberOfBodies, dtype=np.float64)
    zacel0 = np.zeros(numberOfBodies, dtype=np.float64)

    xadot0 = np.zeros(numberOfBodies, dtype=np.float64)
    yadot0 = np.zeros(numberOfBodies, dtype=np.float64)
    zadot0 = np.zeros(numberOfBodies, dtype=np.float64)

    # Arrays for plotting
    xarrs = np.zeros((numberOfBodies,1), dtype=np.float64)
    yarrs = np.zeros((numberOfBodies,1), dtype=np.float64)
    zarrs = np.zeros((numberOfBodies,1), dtype=np.float64)

    # Setting up the values for time etc
    timeKeeper = timeKeeper[0]
    maxTime = maxTime[0]

    # Setting up arrays to determine the kinetic and gravitational potential energy
    KE = np.zeros((numberOfBodies,1), dtype=np.float64)
    GP = np.zeros((numberOfBodies,1), dtype=np.float64)

    return timeKeeper, maxTime, xpos, ypos, zpos, xvel, yvel, zvel, mass, xacel, yacel, zacel, xadot, yadot, zadot, xa2dot, ya2dot, za3dot, xa3dot, ya3dot, za3dot, numberOfBodies, xarrs, yarrs, zarrs, KE, GP, xacel0, yacel0, zacel0, xadot0, yadot0, zadot0

# Function for outputting to a file
def output(time, xpositions, ypositions, zpositions, xvelocities, yvelocities, zvelocities, masses):
    # Defining the name of the file
    fname = "output" + str(time) + ".txt"

    # Opening the file
    with open(fname, "w") as file:
        
        # Writing the headers
        file.write("Time Xpos Ypos Zpos Xvel Yvel Zvel Mass \n")

        numberOfObjects = len(xpositions)

        # Looping through each object 
        for i in range(numberOfObjects):
            file.write(str(time))
            file.write(str(xpositions[i]) + " " + str(ypositions[i]) + " " + str(zpositions[i]) + " ")
            file.write(str(xvelocities[i]) + " " + str(yvelocities[i]) + " " + str(zvelocities[i]) + " ")
            file.write(str(masses[i]) + "\n") 

# The function to start the integrator running
def hermiteIntegrator(dt, dynamicTimeStep, outputInterval, dataFile, maxTime):
    # Initialising the integrator
    timeKeeper, _, xpos, ypos, zpos, xvel, yvel, zvel, mass, xacel, yacel, zacel, xadot, yadot, zadot, xa2dot, ya2dot, za3dot, xa3dot, ya3dot, za3dot, numberOfBodies, xarrs, yarrs, zarrs, KE, GP, xacel0, yacel0, zacel0, xadot0, yadot0, zadot0 = initIntegrator(dataFile)
    
    # Starting a counter for the outputting
    timeElapsedSinceLastOutput = 0

    # The main program loop
    while timeKeeper < maxTime:

        # Calculating the powers of dt
        dt2 = (dt**2)
        dt3 = (dt**3)
        dt4 = (dt**4)
        dt5 = (dt**5)

        # Calculating the initial acceleration values
        xacel0, yacel0, zacel0, xadot0, yadot0, zadot0, _ = accelerations(xpos, ypos, zpos, xvel, yvel, zvel, mass, numberOfBodies)

        # Predicting the objects positions and velocities based on these accelerations
        xpos = np.float64(dt3*sixth*xadot0 + dt2*half*xacel0 + dt*xvel + xpos)
        ypos = np.float64(dt3*sixth*yadot0 + dt2*half*yacel0 + dt*yvel + ypos)
        zpos = np.float64(dt3*sixth*zadot0 + dt2*half*zacel0 + dt*zvel + zpos)

        xvel = np.float64(dt2*half*xadot0 + dt*xacel0 + xvel)
        yvel = np.float64(dt2*half*yadot0 + dt*yacel0 + yvel)
        zvel = np.float64(dt2*half*zadot0 + dt*zacel0 + zvel)

        # Using these predictions to determine the accelerations again
        xacel, yacel, zacel, xadot, yadot, zadot, gravitationalPotentials = accelerations(xpos, ypos, zpos, xvel, yvel, zvel, mass, numberOfBodies)

        # Using these values to determine second and third derivative of acceleration
        xa2dot = np.float64(-6*(xacel0 - xacel)/dt2 - (4*xadot0 + 2*xadot)/dt)
        ya2dot = np.float64(-6*(yacel0 - yacel)/dt2 - (4*yadot0 + 2*yadot)/dt)
        za2dot = np.float64(-6*(zacel0 - zacel)/dt2 - (4*zadot0 + 2*zadot)/dt)

        xa3dot = np.float64(12*(xacel0 - xacel)/dt3 + 6*(xadot0 + xadot)/dt2)
        ya3dot = np.float64(12*(yacel0 - yacel)/dt3 + 6*(yadot0 + yadot)/dt2)
        za3dot = np.float64(12*(zacel0 - zacel)/dt3 + 6*(zadot0 + zadot)/dt2)

        # Using the derivaties to correct the position and velocity values
        xpos = np.float64(xpos + (xa2dot*twentyFourth*dt4) + (xa3dot*oneHundredAndTwentyth*dt5))
        ypos = np.float64(ypos + (ya2dot*twentyFourth*dt4) + (ya3dot*oneHundredAndTwentyth*dt5))
        zpos = np.float64(zpos + (za2dot*twentyFourth*dt4) + (za3dot*oneHundredAndTwentyth*dt5))

        xvel = np.float64(xvel + (xa2dot*sixth*dt3) + (xa3dot*twentyFourth*dt4))
        yvel = np.float64(yvel + (ya2dot*sixth*dt3) + (ya3dot*twentyFourth*dt4))
        zvel = np.float64(zvel + (za2dot*sixth*dt3) + (za3dot*twentyFourth*dt4))

        # Calculating the KE and GPE
        kineticEnergy = half * mass * (xvel*xvel + yvel*yvel + zvel*zvel)
        KE = np.append(KE, np.array_split(kineticEnergy,numberOfBodies), axis=1)
        GP = np.append(GP, np.array_split(gravitationalPotentials, numberOfBodies), axis=1)

        # Updating the timestep based on the accelerations felt
        if dynamicTimeStep == True:
            dt = hermiteTimeStep(xacel, yacel, zacel, xadot, yadot, zadot, xa2dot, ya2dot, za2dot, xa3dot, ya3dot, za3dot, numberOfBodies)

        # Ticking the clock
        timeKeeper = timeKeeper + dt

        # Outputting based on the given time interval
        timeElapsedSinceLastOutput = timeElapsedSinceLastOutput + dt
        if timeElapsedSinceLastOutput > outputInterval:
            # Resetting the clock
            timeElapsedSinceLastOutput = 0

            # Appending to the position arrays
            xarrs = np.append(xarrs, np.array_split(xpos, numberOfBodies), axis=1)
            yarrs = np.append(yarrs, np.array_split(ypos, numberOfBodies), axis=1)
            zarrs = np.append(zarrs, np.array_split(zpos, numberOfBodies), axis=1)

            # Outputting the current data to a file
            output(timeKeeper, xpos, ypos, zpos, xvel, yvel, zvel, mass)

    return GP, KE, xarrs, yarrs, zarrs     

# Asking for simulaiton parameters
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
elif timeUnit == "kyear":
    timeUnit = 1e3 * 365*24*60*60
elif timeUnit == "Year":
    timeUnit = 365*24*60*60
elif timeUnit == "Month":
    timeUnit = 30*24*60*60
elif timeUnit == "Week":
    timeUnit = 7*24*60*60
else:
    timeUnit = 24*60*60

maximumTime = np.float64(timeUnit) * np.float64(timeNumber)

outputInterval = maximumTime / np.float64(outputNumber)

# Calling the hermite function
g, k, x, y, z = hermiteIntegrator(np.int64(initialdt), dyn, np.float64(outputInterval), str(textFile), np.int64(maximumTime))

plt.figure(figsize=(10,10))
plt.plot(x[1][1:], y[1][1:], "r")
plt.xlabel("x, m")
plt.ylabel("y, m")
plt.show()