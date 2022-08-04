import numpy as np
from scipy.interpolate import interp1d

# Read data file into array
# Transpose array to allow for standard indexing
paramT = np.loadtxt('DIXEFT-Parameterization.dat')
param  = paramT.transpose()

# Compute interpolating spline
AEp    = interp1d(param[0], param[1],  kind='cubic')
AEn    = interp1d(param[0], param[2],  kind='cubic')
BE     = interp1d(param[0], param[3],  kind='cubic')
BbarE  = interp1d(param[0], param[4],  kind='cubic')
AMp    = interp1d(param[0], param[5],  kind='cubic')
BMp    = interp1d(param[0], param[6],  kind='cubic')
BbarMp = interp1d(param[0], param[7],  kind='cubic')
AMn    = interp1d(param[0], param[8],  kind='cubic')
BbarMn = interp1d(param[0], param[9],  kind='cubic')
BMn    = interp1d(param[0], param[10], kind='cubic')





# Proton Electric Form Factor
def GEp(Q2,r2Ep,r2En):
    return AEp(Q2) + r2Ep*BE(Q2)  + r2En*BbarE(Q2)

# Neutron Magnetic Form Factors
def GEn(Q2,r2En,r2Ep):
    return AEn(Q2) + r2En*BE(Q2)  + r2Ep*BbarE(Q2)

# Proton Magnetic Form Factors
def GMp(Q2,r2Mp,r2Mn):
    return AMp(Q2) + r2Mp*BMp(Q2) + r2Mn*BbarMp(Q2)

# Neutron Magnetic Form Factors
def GMn(Q2,r2Mn,r2Mp):
    return AMn(Q2) + r2Mn*BMn(Q2) + r2Mp*BbarMn(Q2)






# Set default values of squared radii
# All values in fm^2 units
r2Ep =  0.842**2
r2Mp =  0.850**2
r2En = -0.116
r2Mn =  0.864**2









import matplotlib.pyplot as plt

Q2list = np.linspace(0, 1.5, 301)

# Plot proton electric form factor for various values of the proton electric radius
# The neutron electric radius is kept at its default value defined above

plt.figure(dpi=100, figsize=[5,4])

plt.plot(Q2list,
         GEp(Q2list, 0.80**2, r2En),
         ':',  label='0.80 fm', color='black')
plt.plot(Q2list,
         GEp(Q2list, 0.84**2, r2En),
         '-',  label='0.84 fm', color='black')
plt.plot(Q2list,
         GEp(Q2list, 0.88**2, r2En),
         '--', label='0.88 fm', color='black')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tick_params(axis='y', which='both', tick2On=True, tickdir='in')
plt.tick_params(axis='x', which='both', tick2On=True, tickdir='in')
plt.xlabel('$Q^2$ [GeV$^2$]', fontsize=14)
plt.ylabel('$G_E^p$', fontsize=14)
plt.legend(frameon=False, fontsize=12, loc=1)
plt.show()


# Plot proton magnetic form factor for various values of the proton magnetic radius
# The neutron magnetic radius is kept at its default value defined above

plt.figure(dpi=100, figsize=[5,4])

plt.plot(Q2list,
         GMp(Q2list, 0.80**2, r2Mn),
         ':',  label='0.80 fm', color='black')
plt.plot(Q2list,
         GMp(Q2list, 0.84**2, r2Mn),
         '-',  label='0.84 fm', color='black')
plt.plot(Q2list,
         GMp(Q2list, 0.88**2, r2Mn),
         '--', label='0.88 fm', color='black')

plt.xlim(0,1)
plt.ylim(0,3)
plt.tick_params(axis='y', which='both', tick2On=True, tickdir='in')
plt.tick_params(axis='x', which='both', tick2On=True, tickdir='in')
plt.xlabel('$Q^2$ [GeV$^2$]', fontsize=14)
plt.ylabel('$G_M^p$', fontsize=14)
plt.legend(frameon=False, fontsize=12, loc=1)

plt.show()


# Load Mainz spline fit
DataMainz = np.loadtxt('Spline.dat')

# Plot proton electric form factor
plt.figure(dpi=100, figsize=[5,4])
plt.plot(Q2list, GEp(Q2list, 0.843**2, r2En),'--',
         label='DIXEFT, $r_E^p$ = 0.84 fm',color='black',lw=1)
plt.plot(Q2list, GEp(Q2list, 0.88**2, r2En),'--',
         label='DIXEFT, $r_E^p$ = 0.88 fm',color='grey',lw=1)
plt.fill_between(DataMainz[:,0],DataMainz[:,1]+DataMainz[:,2],DataMainz[:,1]-DataMainz[:,2],
                 color='cyan',label='Mainz Spline')
plt.legend(frameon=False,fontsize=12)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('$Q^2$ [GeV$^2$]',fontsize=14)
plt.ylabel('$G_E^p$',fontsize=14)
plt.show()