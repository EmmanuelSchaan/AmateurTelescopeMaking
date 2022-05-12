import numpy as np
import matplotlib.pyplot as plt


# zones measured
RMeas = np.array([ 40.132,  69.596,  89.789, 106.299, 120.523])   # [mm]


# 2020/11/21
zMirror = np.array([85., 140., 140., 125., 85.])  # [nm]
# 2020/11/22, after 3min of polishing
zMirror2 = np.array([45., 73., 65., 57., 48.]) # [nm]

# variation per minute [nm/minute]
zSpeed = (zMirror2-zMirror) / 3.


fig=plt.figure(0)
ax=fig.add_subplot(111)
#
tol = 16.
ax.fill_between(RMeas, -800./tol, 800./tol, edgecolor=None, facecolor='r', alpha=0.2)
ax.fill_between(RMeas, -600./tol, 600./tol, edgecolor=None, facecolor='g', alpha=0.2)
ax.fill_between(RMeas, -400./tol, 400./tol, edgecolor=None, facecolor='b', alpha=0.2, label=r'$\pm1/16$-wave')
#
ax.plot(RMeas, zMirror, label=r'2020/11/21')
ax.plot(RMeas, zMirror2, label=r'2020/11/22')
ax.plot(RMeas, zMirror2 + zSpeed*2., '--', label=r'2min')
ax.plot(RMeas, zMirror2 + zSpeed*2.5, '--', label=r'2.5min')
ax.plot(RMeas, zMirror2 + zSpeed*2.75, '--', label=r'2.75min')
ax.plot(RMeas, zMirror2 + zSpeed*3., '--', label=r'3min')
#
ax.axhline(0.)
#
ax.legend(loc=1, fontsize='x-small', labelspacing=0.1)
ax.set_xlabel(r'$r$ [mm]')
ax.set_ylabel(r'Offset from parabola [nm]')
#
fig.savefig('./figures/figuring_speed.pdf', bbox_inches='tight')

plt.show()
