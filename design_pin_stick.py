import numpy as np

########################################################################
# Generate the radii, in mm, at which to put a pin 
# on the pin stick, to be used for the wire test,
# or Foucault test.

########################################################################
# Telescope specs

# unit conversion
inchToMm = 2.54 * 10.

# Blank diameter [mm]
D = 10. * inchToMm



########################################################################
# Pin stick with 5 zones,
# chosen from the weird Stellafane calculator:
# https://stellafane.org/tm/atm/test/zone_calc.html
'''
RMeas = D/2. * np.array([0.316, 0.548, 0.707, 0.837, 0.949])   # [mm]
'''

########################################################################
# Pin stick with 9 zones, for final testing,
# with my calculation,
# allowing to specify inner and outer pins
rMin = 1.3 * inchToMm   # [mm] radius of the obstruction by secondary
rMax = 117  # [mm] outer radius such that the wire test line is readable for my 10inch
nZones = 9

I = np.arange(1, nZones+1) 
RMeas = np.sqrt(rMin**2 + (rMax**2-rMin**2) * (I-1.)/(nZones-1.))

# round to nearest millimeter, to simplify making the pin stick
RMeas = np.round(RMeas)

print("Radii [mm] at which to put a pin on the stick,")
print("for the wire/Foucault test:")
print(RMeas)
