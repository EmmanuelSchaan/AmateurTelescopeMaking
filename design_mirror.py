import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# unit conversion
inchToMm = 2.54 * 10.

########################################################################
# Telescope specs

# Blank diameter [mm]
D = 10. * inchToMm

# Focal length
fNumber = 5.   # "f number"
lf = D*fNumber

# Depth of the blank
depth = D / 6.


print("Primary diameter =", D, "mm =", D / inchToMm, "inch")
print("Aperture area =", np.round(np.pi*(0.1*D/2.)**2, 1), "cm^2,")
print("ie ", np.int((D/7.)**2), "times the area of fully dilated eye pupil (7mm)")
print("ie limiting apparent magnitude is", np.round(6 + 2.512*np.log10((D/7.)**2), 1))
print("compared to 6 for the naked eye.")

resBlueArcsec = 1.22 * 400.e-9/(D*1.e-3) * (180.*3600./np.pi)
resRedArcsec = 1.22 * 800.e-9/(D*1.e-3) * (180.*3600./np.pi)
print("Diffraction limited Airy disk radius (Rayleigh criterion):")
print(np.round(resBlueArcsec, 2), "arcsec in blue and ", np.round(resRedArcsec, 2), "arcsec in red")
print("Diffraction limited Airy disk diameter:")
print(np.round(2.*resBlueArcsec, 2), "arcsec in blue and ", np.round(2.*resRedArcsec, 2), "arcsec in red")

print("")
print("f number =", fNumber)
print("focal length =", lf, "mm =", lf / inchToMm, "inch")
print("Min useful magnification, for fully dilated eye pupil (7mm) =", np.round(D/7.,1))
print("ie max useful eyepiece focal length =", np.round(7. * fNumber,1), "mm")
print("Max useful magnification is 30 * (D/1inch) = ", np.round(30. * D/inchToMm, 1))
print("ie min useful eyepiece focal length =", np.round(lf / (30. * D/inchToMm),1), "mm")
print("Magnification =", np.round(lf/5.,1), np.round(lf/10.,1), np.round(lf/20.,1), np.round(lf/35.,1), "for 5 10 20 30mm eyepiece focal length")




########################################################################
# Equations for the circle and parabola, for a given circle radius Rc

def zCircCenter(Rc):
   """Height of the circle center [mm]
   Rc: circle radius [mm]
   """
   return np.sqrt(Rc**2 - (D/2.)**2)


def zCirc(r, Rc):
   """Equation for the circle.
   Rc: circle radius [mm]
   r: polar coordinate [mm]
   Output: height z [mm]
   """
   result = zCircCenter(Rc)
   result -= np.sqrt(Rc**2 - r**2)
   return result


def rContact(Rc):
   """Polar radius [mm] of the contact point between the circle and parabola.
   """
   return np.sqrt(Rc**2 - 4.*lf**2)

def sagittaCirc(Rc):
   """Maximum depth of the sphere [mm],
   given the radius Rc [mm]
   """
   result = Rc - zCircCenter(Rc)
   return result

########################################################################

def zParaOffset(Rc):
   """Vertical offset [mm] for the parabola,
   to be tangent to the circle at the desired polar radius.
   Rc: circle radius [mm]
   """
   rC = rContact(Rc)
   result = zCircCenter(Rc)
   result -= np.sqrt(Rc**2 - rC**2)
   result -= rC**2 / (4.*lf)
   return result


def zPara(r, Rc):
   """Equation for the parabola,
   with vertical offset determined to be tangent to the circle of radius Rc
   """
   result = zParaOffset(Rc)
   result += r**2 / (4. * lf)
   return result


########################################################################
# Find the circle radius Rc, to have the desired rContact = alpha * D/2
# Wisdom says alpha = 0.7

def RcForContact(alpha):
   """Find Rc needed for contact at rContact = alpha * D/2
   """
   result = (alpha*D/2.)**2
   result += 4. * lf**2
   return np.sqrt(result)

# Typical wisdom is to have the contact point at 70%
# of the radius of the blank
Rc70 = RcForContact(0.7)


########################################################################
# Find the circle radius that minimizes the volume between the parabola and the circle

def dVolumedRc(Rc):
   """Quantity proportional to the derivative of the volume between circle and parabola,
   with respect to the circle radius.
   Find the root of this function to minimize the volume between the sphere
   and paraboloid.
   """
   beta = Rc / (D/2.)
   result = 1. - np.sqrt(1. - 1./beta**2)
   result -= 1. / (8.*beta*fNumber)
   return result


# guess for the circle radius: twice the focal length
RcGuess = 2. * D * fNumber
RcBest = optimize.brentq(dVolumedRc , 0.9*RcGuess, 1.1*RcGuess)
rContactBest = rContact(RcBest)
alphaBest = rContactBest/(D/2.)


print("")
print("To minimize the volume between sphere and paraboloid,")
print("the contact point should be at a fraction", np.round(alphaBest, 4), "of the blank radius.")
print("The wisdom is 0.7.")

print("Hence the best circle radius =", np.round(RcBest, 3), "mm =", np.round(RcBest/inchToMm, 3), "inch")
print("The guess would be twice the focal length, ie", RcGuess, "mm =", np.round(RcGuess/inchToMm, 3), "inch")

print("Circle sagitta =", np.round(sagittaCirc(RcBest),5), "mm =", np.round(sagittaCirc(RcBest) / inchToMm,5), "inch")


########################################################################
# plot the mirror curve

R = np.linspace(-D/2., D/2., 501)
ZCirc = zCirc(R, RcBest)
ZPara = zPara(R, RcBest)


fig=plt.figure(0)
ax=fig.add_subplot(111)
#
# Circle
ax.plot(R*0.1, ZCirc, label=r'Circle')
# Parabola
ax.plot(R*0.1, ZPara, label=r'Parabola')
#
# Contact points
ax.plot(np.array([-rContactBest, rContactBest])*0.1, [zCirc(rContactBest, RcBest), zCirc(rContactBest, RcBest)], 'ko')
#
# Blank
ax.plot(R*0.1, np.zeros_like(R), 'k--')
ax.plot(R*0.1, -depth * np.ones_like(R), 'k--')
ax.plot(np.array([-D/2., -D/2.])*0.1, [-depth, 0.], 'k--')
ax.plot(np.array([D/2., D/2.])*0.1, [-depth, 0.], 'k--')
#
ax.legend(loc=3)
ax.set_xlabel(r'$r$ [cm]')
ax.set_ylabel(r'$z$ [mm]')
#
fig.savefig("./figures/circle.pdf", bbox_inches='tight')



fig=plt.figure(1)
ax=fig.add_subplot(111)
#
# Show tolerances for red, green and blue wavelengths:
# should be wavelength/8 to reach the Rayleigh criterion for a miror,
# and wavelength/4 for a lens
tol = 8.
ax.fill_between(R*0.1, -800.e-3/tol, 800.e-3/tol, edgecolor=None, facecolor='r', alpha=0.2)
ax.fill_between(R*0.1, -600.e-3/tol, 600.e-3/tol, edgecolor=None, facecolor='g', alpha=0.2)
ax.fill_between(R*0.1, -400.e-3/tol, 400.e-3/tol, edgecolor=None, facecolor='b', alpha=0.2)
#
# Compare circle and parabola
ax.plot(R*0.1, 0.*R, 'k--', label=r'Circle')
ax.plot(R*0.1, (ZPara - ZCirc)*1.e3, 'k', label=r'Parabola')
#
# Contact points
ax.plot(np.array([-rContactBest, rContactBest])*0.1, [0., 0.], 'ko')
#
ax.legend(loc=2, fontsize='x-small')
ax.set_xlabel(r'$r$ [cm]')
ax.set_ylabel(r'$z_\text{Parabola} - z_\text{Circle}$ [$\mu$m]')
#
fig.savefig("./figures/parabola_vs_circle.pdf", bbox_inches='tight')

plt.show()
