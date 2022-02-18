''' A fully automatic integrator for my physics project '''

# External dependencies
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import tqdm

# Defining important constants
G = np.float64(6.67e-11)
softening = np.float64(6e7)
eta = np.float64(0.05)

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
            rijx =  x[otherBodyIndex] - x[bodyIndex]
            rijy =  y[otherBodyIndex] - y[bodyIndex]
            rijz =  z[otherBodyIndex] - z[bodyIndex]

            vijx =  vx[otherBodyIndex] - vx[bodyIndex] 
            vijy =  vy[otherBodyIndex] - vy[bodyIndex] 
            vijz =  vz[otherBodyIndex] - vz[bodyIndex]

            # Calculating the magnitudes of the vectors
            rij = np.sqrt(rijx*rijx + rijy*rijy + rijz*rijz + softening*softening)

            # Finding the dot product of the velocity and position vectors
            rijvijdot = rijx*vijx + rijy*vijy + rijz*vijz

            # Calculating the divison terms used to calculate a and adot
            divisonTermThree = 1.0/rij**(3)
            divisonTermFive = 1.0/rij**(5)

            # Calculating acceleration
            ax[bodyIndex] = ax[bodyIndex] + (G * m[otherBodyIndex]) * (rijx*divisonTermThree)
            ay[bodyIndex] = ay[bodyIndex] + (G * m[otherBodyIndex]) * (rijy*divisonTermThree)
            az[bodyIndex] = az[bodyIndex] + (G * m[otherBodyIndex]) * (rijz*divisonTermThree)

            # Calculating adot
            adx[bodyIndex] = adx[bodyIndex] + (G * m[otherBodyIndex]) * ((vijx*divisonTermThree) + 3.0*(rijvijdot*rijx*divisonTermFive))
            ady[bodyIndex] = ady[bodyIndex] + (G * m[otherBodyIndex]) * ((vijy*divisonTermThree) + 3.0*(rijvijdot*rijy*divisonTermFive))
            ady[bodyIndex] = adz[bodyIndex] + (G * m[otherBodyIndex]) * ((vijz*divisonTermThree) + 3.0*(rijvijdot*rijz*divisonTermFive))

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
def hermiteTimeStep(xacel, yacel, zacel, xadot, yadot, zadot, xadotdot, yadotdot, zadotdot, xadotdotdot, yadotdotdot, zadotdotdot, numberOfBodies, dt):
    # Creating an empty list for the prospective timesteps
    dtList = []
    dt = float(dt)
    newdt = float(0)
    arsethStabilityCriterion = float(1.3)

    # Looping through each body to determine the timestep for that body
    for body in range(numberOfBodies):

        # Calculating the scalar for a, adot, adotdot and adotdotdot
        a = np.sqrt(xacel[body]*xacel[body] + yacel[body]*yacel[body] + zacel[body]*zacel[body])
        ad = np.sqrt(xadot[body]*xadot[body] + yadot[body]*yadot[body] + zadot[body]*zadot[body])
        add = np.sqrt(xadotdot[body]*xadotdot[body] + yadotdot[body]*yadotdot[body] + zadotdot[body]*zadotdot[body])
        addd = np.sqrt(xadotdotdot[body]*xadotdotdot[body] + yadotdotdot[body]*yadotdotdot[body] + zadotdotdot[body]*zadotdotdot[body])

        # Calculating the timestep
        dtList.append(np.sqrt(eta * ((a*add + ad**2)/(ad*addd + add**2))))

    # Checking the timestep jump isn't to big
    if min(dtList)/dt > arsethStabilityCriterion:
        newdt = dt*arsethStabilityCriterion
    else:
        newdt = min(dtList)

    # Returning the smallest timestep for usage
    return newdt

# The function that initialises the integrator from a data file input
def initIntegrator(dataFile):
    # Importing initial conditions
    timeKeeper, maxTime, xpos, ypos, zpos, xvel, yvel, zvel, mass = np.loadtxt(dataFile, delimiter=" ", skiprows=1, unpack=True, dtype=np.float64)

    # Getting the number of bodies
    numberOfBodies = len(xpos)

    # Setting up arrays
    xacelP = np.zeros(numberOfBodies, dtype=np.float64)
    yacelP = np.zeros(numberOfBodies, dtype=np.float64)
    zacelP = np.zeros(numberOfBodies, dtype=np.float64)

    xadotP = np.zeros(numberOfBodies, dtype=np.float64)
    yadotP = np.zeros(numberOfBodies, dtype=np.float64)
    zadotP = np.zeros(numberOfBodies, dtype=np.float64)

    xadotdot = np.zeros(numberOfBodies, dtype=np.float64)
    yadotdot = np.zeros(numberOfBodies, dtype=np.float64)
    zadotdot = np.zeros(numberOfBodies, dtype=np.float64)

    xadotdotdot = np.zeros(numberOfBodies, dtype=np.float64)
    yadotdotdot = np.zeros(numberOfBodies, dtype=np.float64)
    zadotdotdot = np.zeros(numberOfBodies, dtype=np.float64)

    xAcelC = np.zeros(numberOfBodies, dtype=np.float64)
    yAcelC = np.zeros(numberOfBodies, dtype=np.float64)
    zAcelC = np.zeros(numberOfBodies, dtype=np.float64)

    xAdotC= np.zeros(numberOfBodies, dtype=np.float64)
    yAdotC = np.zeros(numberOfBodies, dtype=np.float64)
    zAdotC = np.zeros(numberOfBodies, dtype=np.float64)

    xAcelCT = np.zeros(numberOfBodies, dtype=np.float64)
    yAcelCT = np.zeros(numberOfBodies, dtype=np.float64)
    zAcelCT = np.zeros(numberOfBodies, dtype=np.float64)

    xAdotCT = np.zeros(numberOfBodies, dtype=np.float64)
    yAdotCT = np.zeros(numberOfBodies, dtype=np.float64)
    zAdotCT = np.zeros(numberOfBodies, dtype=np.float64)

    # Arrays for plotting
    xarrs = np.zeros((numberOfBodies,1), dtype=np.float64)
    yarrs = np.zeros((numberOfBodies,1), dtype=np.float64)
    zarrs = np.zeros((numberOfBodies,1), dtype=np.float64)
    tarrs = np.zeros(1, dtype=np.float64)
    dtarrs = np.zeros(1, dtype=np.float64)

    # Setting up the values for time etc
    timeKeeper = timeKeeper[0]
    maxTime = maxTime[0]

    # Setting up arrays to determine the kinetic and gravitational potential energy
    KE = np.zeros((numberOfBodies,1), dtype=np.float64)
    GP = np.zeros((numberOfBodies,1), dtype=np.float64)

    return timeKeeper, maxTime, xpos, ypos, zpos, xvel, yvel, zvel, mass, xacelP, yacelP, zacelP, xadotP, yadotP, zadotP, xadotdot, yadotdot, zadotdotdot, xadotdotdot, yadotdotdot, zadotdotdot, numberOfBodies, xarrs, yarrs, zarrs, KE, GP, xAcelC, yAcelC, zAcelC, xAdotC, yAdotC, zAdotC, xAcelCT, yAcelCT, zAcelCT, xAdotCT, yAdotCT, zAdotCT, dtarrs, tarrs

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
def hermiteIntegrator(dt, dynamicTimeStep, outputNumber, dataFile, maxTime):
    # Initialising the integrator
    timeKeeper, _, xpos, ypos, zpos, xvel, yvel, zvel, mass, xacelP, yacelP, zacelP, xadotP, yadotP, zadotP, xadotdot, yadotdot, zadotdotdot, xadotdotdot, yadotdotdot, zadotdotdot, numberOfBodies, xarrs, yarrs, zarrs, KE, GP, xAcelC, yAcelC, zAcelC, xAdotC, yAdotC, zAdotC, xAcelCT, yAcelCT, zAcelCT, xAdotCT, yAdotCT, zAdotCT, # dtarrs, tarrs = initIntegrator(dataFile)
    
    # Starting a counter for the outputting
    outputInterval = maxTime / np.float64(outputNumber)
    timeElapsedSinceLastOutput = 0

    # Starting the progress bar
    progressBar = tqdm(total=maxTime)
    
    # Setting the order of the scheme
    pecOrder = 2

    # The main program loop
    while timeKeeper < maxTime:

        # Calculating the powers of dt
        dt = np.float64(dt)
        dt2 = np.float64(dt**2)
        dt3 = np.float64(dt**3)

        # Prediction Step
        xacelP, yacelP, zacelP, xadotP, yadotP, zadotP, _ = accelerations(xpos, ypos, zpos, xvel, yvel, zvel, mass, numberOfBodies)

        xPosPrediction = np.float64(dt3*sixth*xadotP + dt2*half*xacelP + dt*xvel + xpos)
        yPosPrediction = np.float64(dt3*sixth*yadotP + dt2*half*yacelP + dt*yvel + ypos)
        zPosPrediction = np.float64(dt3*sixth*zadotP + dt2*half*zacelP + dt*zvel + zpos)

        xVelPrediction = np.float64(dt2*half*xadotP + dt*xacelP + xvel)
        yVelPrediction = np.float64(dt2*half*yadotP + dt*yacelP + yvel)
        zVelPrediction = np.float64(dt2*half*zadotP + dt*zacelP + zvel)

        # First Correction Step
        xAcelC, yAcelC, zAcelC, xAdotC, yAdotC, zAdotC, gravitationalPotentials = accelerations(xPosPrediction, yPosPrediction, zPosPrediction, xVelPrediction, yVelPrediction, zVelPrediction, mass, numberOfBodies)

        xadotdot = np.float64(-6.0*(xacelP - xAcelC) - (4.0*xadotP + 2.0*xAdotC)*dt)
        yadotdot = np.float64(-6.0*(yacelP - yAcelC) - (4.0*yadotP + 2.0*yAdotC)*dt)
        zadotdot = np.float64(-6.0*(zacelP - zAcelC) - (4.0*zadotP + 2.0*zAdotC)*dt)

        xadotdotdot = np.float64(12.0*(xacelP - xAcelC) + 6.0*(xadotP + xAdotC)*dt)
        yadotdotdot = np.float64(12.0*(yacelP - yAcelC) + 6.0*(yadotP + yAdotC)*dt)
        zadotdotdot = np.float64(12.0*(zacelP - zAcelC) + 6.0*(zadotP + zAdotC)*dt)

        xPosCorrectionOne = np.float64(xPosPrediction + (xadotdot*twentyFourth*dt2) + (xadotdotdot*oneHundredAndTwentyth*dt2))
        yPosCorrectionOne = np.float64(yPosPrediction + (yadotdot*twentyFourth*dt2) + (yadotdotdot*oneHundredAndTwentyth*dt2))
        zPosCorrectionOne = np.float64(zPosPrediction + (zadotdot*twentyFourth*dt2) + (zadotdotdot*oneHundredAndTwentyth*dt2))

        xVelCorrectionOne = np.float64(xVelPrediction + (xadotdot*sixth*dt) + (xadotdotdot*twentyFourth*dt))
        yVelCorrectionOne = np.float64(yVelPrediction + (yadotdot*sixth*dt) + (yadotdotdot*twentyFourth*dt))
        zVelCorrectionOne = np.float64(zVelPrediction + (zadotdot*sixth*dt) + (zadotdotdot*twentyFourth*dt))

        # Second Correction Step This entire step is currently commented out in the real code as including this produces even greater energy errors
        xAcelCT, yAcelCT, zAcelCT, xAdotCT, yAdotCT, zAdotCT, gravitationalPotentials = accelerations(xPosCorrectionOne, yPosCorrectionOne, zPosCorrectionOne, xVelCorrectionOne, yVelCorrectionOne, zVelCorrectionOne, mass, numberOfBodies)

        xadotdot = np.float64(-6.0*(xAcelC - xAcelCT) - (4.0*xAdotC + 2.0*xAdotCT)*dt)
        yadotdot = np.float64(-6.0*(yAcelC - yAcelCT) - (4.0*yAdotC + 2.0*yAdotCT)*dt)
        zadotdot = np.float64(-6.0*(zAcelC - zAcelCT) - (4.0*zAdotC + 2.0*zAdotCT)*dt)

        xadotdotdot = np.float64(12.0*(xAcelC - xAcelCT) + 6.0*(xAdotC + xAdotCT)*dt)
        yadotdotdot = np.float64(12.0*(yAcelC - yAcelCT) + 6.0*(yAdotC + yAdotCT)*dt)
        zadotdotdot = np.float64(12.0*(zAcelC - zAcelCT) + 6.0*(zAdotC + zAdotCT)*dt)

        xPosCorrectionTwo = np.float64(xPosCorrectionOne + (xadotdot*twentyFourth*dt2) + (xadotdotdot*oneHundredAndTwentyth*dt2))
        yPosCorrectionTwo = np.float64(yPosCorrectionOne + (yadotdot*twentyFourth*dt2) + (yadotdotdot*oneHundredAndTwentyth*dt2))
        zPosCorrectionTwo = np.float64(zPosCorrectionOne + (zadotdot*twentyFourth*dt2) + (zadotdotdot*oneHundredAndTwentyth*dt2))

        xVelCorrectionTwo = np.float64(xVelCorrectionOne + (xadotdot*sixth*dt) + (xadotdotdot*twentyFourth*dt))
        yVelCorrectionTwo = np.float64(yVelCorrectionOne + (yadotdot*sixth*dt) + (yadotdotdot*twentyFourth*dt))
        zVelCorrectionTwo = np.float64(zVelCorrectionOne + (zadotdot*sixth*dt) + (zadotdotdot*twentyFourth*dt))

        xpos = xPosCorrectionTwo
        ypos = yPosCorrectionTwo
        zpos = zPosCorrectionTwo

        xvel = xVelCorrectionTwo
        yvel = yVelCorrectionTwo
        zvel = zVelCorrectionTwo

        # Calculating the KE and GPE
        kineticEnergy = half * mass * (xvel*xvel + yvel*yvel + zvel*zvel)
        KE = np.append(KE, np.array_split(kineticEnergy,numberOfBodies), axis=1)
        GP = np.append(GP, np.array_split(gravitationalPotentials, numberOfBodies), axis=1)

        # Updating the timestep based on the accelerations felt
        if dynamicTimeStep == True:
            dt = hermiteTimeStep(xac, yac, zac, xadc, yadc, zadc, xadotdot, yadotdot, zadotdot, xadotdotdot, yadotdotdot, zadotdotdot, numberOfBodies, dt)

        # Ticking the clock
        timeKeeper = timeKeeper + int(dt)

        # Outputting based on the given time interval
        timeElapsedSinceLastOutput = timeElapsedSinceLastOutput + dt
        if timeElapsedSinceLastOutput > outputInterval:
            # Updating the progress bar
            progressBar.update(timeElapsedSinceLastOutput)

            # Resetting the clock
            timeElapsedSinceLastOutput = 0

            # Appending to the position arrays
            xarrs = np.append(xarrs, np.array_split(xpos, numberOfBodies), axis=1)
            yarrs = np.append(yarrs, np.array_split(ypos, numberOfBodies), axis=1)
            zarrs = np.append(zarrs, np.array_split(zpos, numberOfBodies), axis=1)
            dtarrs = np.append(dtarrs, dt)
            tarrs = np.append(tarrs, timeKeeper)

            # Outputting the current data to a file
            #output(timeKeeper, xpos, ypos, zpos, xvel, yvel, zvel, mass)

    # Closing the progress bar
    progressBar.close()

    return GP, KE, xarrs, yarrs, zarrs, dtarrs   

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
g, k, x, y, z, t = hermiteIntegrator(np.int64(initialdt), dyn, np.int64(outputNumber), str(textFile), np.int64(maximumTime))

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
plt.plot(x[1][1:], y[1][1:])
plt.show()

plt.figure(figsize=(10,10))
plt.plot(t[1:])
plt.show()