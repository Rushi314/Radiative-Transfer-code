import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

h=6.626e-34 #Planck constant
c=3.0e8 #Velocity of light in vaccum
k=1.38e-23 #Boltzmann constant
G=6.673e-11 #Gravitational constant

T_ex=10 #Exitation temperature
T_kin=10 #Kinetic temperature
T_bg=2.7 #CMB temperature

amu=1.66e-27 #Atomic mass unit
m=2*amu #of Hydrogen molecule
mass_number=28 #of CO tracer
mass=mass_number*amu

nu0=115.3e9 #J1->J0 line frequency of CO

XCO=1.0e-4 #fraction of CO
A10=7.67e-8
BA10=A10/(8*np.pi*h*nu0**3/c**3) #Einstein coefficients for emmission
g1=3 #degeneracy of upper level
g0=1 #degeneracy of lower level
BA01=(g1/g0)*BA10 #Einstein coefficients for absorption
L=24*3.0e16 #2 pc diameter
R=L/(2*3.0e16) #Radius of molecular cloud in pc

vc=0 #m/s
dopp=np.sqrt(1-(vc/c)**2)/(1-(vc/c)) #Doppler

#print dopp

def nr(i,x): #Gaussian number density profile
    x=x/3.0e16 #m to pc
    d=np.sqrt(x**2)+yoffset**2+zoffset**2)
    d=d*3.0e16 #pc to m
    return n0H2[i]*(n0H2[i]/1e6)**(-4*d**2/L**2)
'''
def Nr(i,x):
    N=lambda x: nr(i,x)*4*np.pi*x**2
    if x>=0:
       #print x/3e16
       return integrate.quad(N,0,x)[0]
    if x<0:
       #print x/3e16
       return -integrate.quad(N,0,x)[0]
'''
def num(i,x): 
    return XCO*nr(i,x)

'''
def Nr(i,x):#derivative of numberdensity #uniform density
    if x>=0:
       return (4/3)*n0H2[i]*np.pi*x**3
    if x<0:
       return -(4/3)*n0H2[i]*np.pi*x**3

def vel(i,x): #Velocity profile for gravitational collapse
    vr=lambda x: -2*G*m*Nr(i,x)/x**2
    if x>0:
       return np.sqrt(-integrate.quad(vr, 0, x)[0])
    if x<0:
       return -np.sqrt(integrate.quad(vr, 0, x)[0])
'''
'''
def vel(i,x):
    #print x/3e16
    if x==0:
       return 0
    vr=2*G*m*Nr(i,x)/(3*x) #gravitation and drag
    if x>0:
       #print x/3e16, vr
       return -np.sqrt(vr)
    if x<0:
       #print x/3e16, vr
       return np.sqrt(-vr)
'''

def vel(i,x): #Velocity profile(Considering rotation of cloud) 
    x=x/3.0e16  #m to pc
    d=np.sqrt(x**2)+yoffset**2+zoffset**2)
    if x==0 and d==0:
       return -(3**i)*(1000/R)*d/np.exp((10/R)*d)
    if x>=0 and d!=0:
       return -(3**i)*(1000/R)*d/np.exp((10/R)*d)*(x/d)-omega*yoffset*3.0e16
    if x<0:
       d=-d
       return -(3**i)*(1000/R)*d/np.exp(-(10/R)*d)*(x/d)-omega*yoffset*3.0e16

'''
def rot(i,x):
    if x==0:
       return 0
    vt=G*m*Nr(i,x)/x-vel(i,x)**2
    if x>0:
       #print x/3e16, vr
       return np.sqrt(vt)
    if x<0:
       #print x/3e16, vr
       return -np.sqrt(-vt)
'''
def Temp(x): #Temperature profile for hot thick collapsing core and quasi static envelope
    x=x/3.0e16  #m to pc
    d=np.sqrt(x**2)#+yoffset**2+zoffset**2)
    if x>=0:
       return 10+5/(1+(00.001/R)*np.exp((50/R)*d))
    if x<0:
       d=-d
       return 10+5/(1+(00.001/R)*np.exp(-(50/R)*d))

'''
def Temp(x): # Gradually varing temperature profile across cloud 
    x=x/3.0e16  #m to pc
    if x>=0:
       return 15-8.64e-5*np.exp(10.966*x)
       #return 10+8.64e-5*np.exp(10.966*x)
    if x<0:
       return 15-8.64e-5*np.exp(-10.966*x)
       #return 10+8.64e-5*np.exp(-10.966*x)
'''
def lp(nu,nu0,x): #Doppler widened(Gaussian) spectral line profile
    #print x/3e16
    sigma=nudisp(x)
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(nu-nu0)**2/(2*sigma**2)) #normalized normal distribution

def opac(nu,i,x): #Opacity function
    Temper=Temp(x)
    Number=num(i,x)
    num1=Number/(1+(g0/g1)*np.exp(h*nu0/(k*Temper)))
    num0=Number/(1+(g1/g0)*np.exp(-h*nu0/(k*Temper)))
    return (num0*BA01-num1*BA10)*(h*nu0/c)*lp(nu,(nu0+delnu(i,x)),x) #delnu(i,x) takes time; this gives the wedge shape

def delnu(i,x): 
    return -nu0*vel(i,x)/c

def vdisp(x): #Velocity dispersion
    return np.sqrt(k*Temp(x)/mass)

def nudisp(x): #Frequency dispersion
    return nu0*vdisp(x)/c

def J(T,nu): 
    """ Return Planck function """
    return (h/k)*nu/(np.exp((h/k)*(nu/T))-1)

def J_dopp(T,nu):
    """ Return Doppler Planck function """
    return (h/k)*nu*dopp/(np.exp((h/k)*(nu/T)/dopp)-1)

def taufix(tau0,nu,nu_disp):
    """ Return optical depth """
    return tau0*np.exp(-(nu-nu0)**2/(2*nu_disp**2))

nH2=[1e7,1e8,1e9,1e10] #number density of H2 in m-3
n0H2=[1.97824e7,2.66935e8,3.20622e9,3.65764e10] #number density of OH2 in m-3

ntotal=np.zeros([len(nH2)])
for i in range(len(nH2)):
    ntotal[i]=4/3*np.pi*(L/2)**3*nH2[i]
#print ntotal

nu_start=nu0-0.005e9 #original=0.002e9
nu_stop=nu0+0.005e9

v_disp=np.sqrt(k*T_kin/mass)
nu_disp=v_disp*nu0/c

taufix0=[i*((h*nu0/c)*BA01)*(1/nu_disp)*XCO*L for i in nH2]

taunu=np.zeros([4])

numsteps = 100

resolution=(nu_stop-nu_start)/numsteps
deltanu=(vc/c)*nu0
nushift=np.linspace(nu_start-deltanu, nu_stop-deltanu, numsteps)

def tauget(i,x,q): #Optical Depth from x to the edge of the cloud
    #taunux=np.linspace(-hc,hc,numsteps)
    #for j in range(numsteps): #j loop (inner)
    ns=lambda s: opac(nu[q],i,s)
    taunux=integrate.quad(ns,-hc,x)[0] #gives tau at a point along LOS in cloud
    #at the end of j loop, we get tau profile (array) at a point along LOS in cloud
    return taunux

def rt(i,nu): #Radiative transfer equation
    #delnu=np.zeros([numsteps])
    radtemp1=np.zeros([numsteps])
    radtemp2=np.zeros([numsteps])
    for q in range(numsteps): #q loop (outer)
        #taunu=np.zeros([numsteps])
        nr=lambda r: J(Temp(r),nu[q])*np.exp(-tauget(i,r,q))*opac(nu[q],i,r) #shift this whole thing?
        radtemp2[q]=integrate.quad(nr,-hc,hc)[0] #how to specify number of steps
        radtemp1[q]=J(T_bg,nu[q])*np.exp(-tauget(i,hc,q))+radtemp2[q]
    #print radtemp1[numsteps/2]
    return radtemp1
#position=np.linspace(-L/2,L/2,numsteps)
#for i in range(4): #check various line profiles
#    radvel=np.zeros([numsteps])
#    numcum=np.zeros([numsteps])
#    numden=np.zeros([numsteps])
#    tp=np.zeros([numsteps])
#    acc=np.zeros([numsteps])
#    for l in range(numsteps):
        #radvel[l]=vel(i,position[l])
        #numcum[l]=Nr(i,position[l])
#        numden[l]=nr(i,position[l])
#        tp[l]=Temp(position[l])
        #acc[l]=grav(i,position[l])-buoy(i,position[l])
    #plt.semilogy(position/3.0e16,radvel,'k')
    #plt.plot(position/3.0e16,radvel,'k')
    #plt.plot(position/3.0e16,acc,'k')
    #plt.plot(position/3.0e16,numcum/1e49,'k')
    #plt.plot(position/3.0e16,numden,'k')
#    plt.plot(position/3.0e16,tp,'k')

#plt.xlabel('position')
#plt.ylabel('radial velocity of layer')

yslices=5 #Equal width slices of molecular cloud in y-axis
ygap=1./yslices
zslices=5 #Equal width slices of molecular cloud in z-axis
zgap=1./zslices

for i in [3]:
    ztotal=0
    total_yslices=0
    omega=1e-13
    T_mb_ysum=np.zeros([numsteps])
    T_mb_zsum=np.zeros([numsteps])
    for zp in range(zslices):
        ytotal=0
        yslices=int(round(np.sqrt(1-(zp*zgap)**2)/ygap))
        #total_yslices=total_yslices+yslices
        #print yslices
        #print total_yslices
        los_zoffset=zp*zgap #fraction of radius L/2
        zoffset=los_zoffset*(L/2)/3.0e16 #in pc
        for yp in range(yslices):
            for w in range(2):
                print(zp,yp,w)
                los_yoffset=((-1)**w)*yp*ygap #fraction of radius L/2
                yoffset=los_yoffset*(L/2)/3.0e16 #in pc
                hc=L/2*np.cos(np.arcsin(los_zoffset))*np.cos(np.arcsin(los_yoffset/np.cos(np.arcsin(los_zoffset)))) #half-chord (semi-chord)
                #possteps=int(2*hc*L/numsteps)
                #position=np.linspace(-hc,hc,possteps) #possteps depend on offset?
                #print hc,offset
                #opdep=np.zeros([numsteps])
                T_mb=np.zeros([numsteps])
                Tfix_mb=np.zeros([numsteps])
                nu=np.linspace(nu_start,nu_stop,numsteps)
                #opdep=tau(i,nu)
                T_mb=rt(i,nu)-J(T_bg,nu) #try with plus; removing cmb
                #yweight=((yp+1)*ygap)**2-(yp*ygap)**2 #or use different weighing scheme
                #yweight=np.sqrt(1-(yp*ygap)**2)-np.sqrt(1-((yp+1)*ygap)**2)
                #yweight=ygap*(np.arcsin((yp+1)*ygap)-np.arcsin(yp*ygap))
                yweight=1
                ytotal=ytotal+yweight
                T_mb_ysum=T_mb_ysum+T_mb*yweight
                #Tfix_mb=(J(T_ex,nu)-J(T_bg,nu))*(1-np.exp(-taufix(taufix0[i],nu,nu_disp)))
        T_mb_ysum=T_mb_ysum/ytotal
        zweight=(np.arcsin((zp+1)*zgap)+((zp+1)*zgap)*np.sqrt(1-(zp*zgap)**2))-(np.arcsin(zp*zgap)+(zp*zgap)*np.sqrt(1-(zp*zgap)**2))
        ztotal=ztotal+zweight
        T_mb_zsum=T_mb_zsum+T_mb_ysum*zweight
    T_mb_zsum=T_mb_zsum/ztotal
    plt.subplot(2,2,i+1)
    #plt.plot(nushift/10000,opdep,'k-',linewidth=.5,label=r'nH2=%.0f cm$^{-3}$'%(nH2[i]/1e6))
    #plt.plot(nushift/10000,tauget(i,nu),'k-',linewidth=.5,label=r'nH2=%.0f cm$^{-3}$'%(nH2[i]/1e6))
    #plt.plot(nu/10000,Tfix_mb,'b--',linewidth=.5)
    plt.plot(nu/10000,T_mb_zsum,'k-',linewidth=.5,label=r'nH2=%.0f cm$^{-3}$'%(nH2[i]/1e6)) #for m^3 format: r'nH2=%.1e m$^{-3}$'%nH2[i]
    plt.legend(loc=1,frameon=False)
    plt.xlabel('frequency ($10^4$ Hz)')
    plt.ylabel('radiation temperature (K)')

plt.show()
