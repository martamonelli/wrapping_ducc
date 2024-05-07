import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import mueller_convolver
import ducc0
import scanning_mod as scanning

fsamp = 19.1
ndays = 1
nsamp = ndays*24*3600*fsamp

ctim0 = 0.
ctime = ctim0 + np.arange(nsamp)/fsamp
theta, phi, psi = scanning.ctime2bore(ctime)

ptg = np.empty((nsamp,3))
ptg[:,0] = theta
ptg[:,1] = phi
ptg[:,2] = psi

hwp_angles = np.random.uniform(0,2*np.pi,size=(nsamp,))  # alpha

'''
pixs = hp.ang2pix(32, theta, phi)

hits = np.empty(12*32**2)

for p in np.arange(12*32**2):
    hits[p] = np.count_nonzero(pixs == p)
        
hp.mollview(hits)
plt.show()
'''

def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)

def make_full_random_alm(lmax, mmax, rng):
    res = rng.uniform(-1., 1., (4, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (4, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    # components 1 and 2 are spin-2, fix them accordingly
    spin=2
    for s in range(spin):
        res[1:3, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    return res

def get_gauss_beam_from_beamconv(fwhm, lmax):
    import beamconv
    blmT, blmm2 = beamconv.tools.gauss_blm(fwhm*180*60/np.pi, lmax, pol=True)
    res = np.zeros((4,blmT.shape[0]), dtype=np.complex128)
    blmE, blmB = beamconv.tools.spin2eb(blmm2, blmm2*0, spin=2)
    res[0] = blmT
    res[1] = blmE
    res[2] = blmB
    res[3] = blmT  # correct?
    return res

def blm_gauss_new(fwhm, lmax, pol=False):
    fwhm = float(fwhm)
    lmax = int(lmax)
    pol = bool(pol)
    mmax = 2 if pol else 0
    ncomp = 3 if pol else 1
    nval = hp.Alm.getsize(lmax, mmax)

    if mmax > lmax:
        raise ValueError("lmax value too small")

    blm = np.zeros((ncomp, nval), dtype=np.complex128)
    sigmasq = fwhm * fwhm / (8 * np.log(2.0))

    for l in range(lmax+1):
        blm[0, hp.Alm.getidx(lmax, l, 0)] = np.exp(-0.5*sigmasq*l*(l+1))

    if pol:
        for l in range(2, lmax+1):
            blm[1, hp.Alm.getidx(lmax, l, 2)] = np.exp(-0.5 * sigmasq * (l*(l+1)-4))
        blm[2] = 1j * blm[1]

    return blm

# blm_gauss_new times sqrt((2*l+1)/(4pi))
def Blm_gauss_new(fwhm, lmax, pol=False):
    blm = blm_gauss_new(fwhm, lmax, pol)
    for l in range(lmax+1):
        blm[0, hp.Alm.getidx(lmax, l, 0)] *= np.sqrt((2*l+1) / (4*np.pi))

    if pol:
        for l in range(2, lmax+1):
            blm[1:3, hp.Alm.getidx(lmax, l, 2)] *= np.sqrt((2*l+1) / (4*np.pi))

    return blm
    
np.random.seed(10)
rng = np.random.default_rng(np.random.SeedSequence(42))
lmax = 30
kmax = 18

# completely random sky
slm =make_full_random_alm(lmax, lmax, rng)

# completely random Mueller matrix
mueller = np.random.uniform(-1,1,size=(4,4))
#mueller[1:3,0]=mueller[1:3,-1] = 0
#mueller[0,2]=mueller[2,0] = 0

# completely random beam
blm = make_full_random_alm(lmax, kmax, rng)
# ... or use a Gauss beam
#blmtmp = blm_gauss_new(np.radians(10.), lmax, True)
#blm *= 0
#blm [0:3, 0:blmtmp.shape[1]] = blmtmp

# Now do the same thing with MuellerConvolver
fullconv = mueller_convolver.MuellerConvolver(
    lmax=lmax,
    kmax=kmax,
    slm=slm,
    blm=blm,
    mueller=mueller,
    single_precision=False,
    epsilon=1e-7,
    nthreads=1,
)
signal_muellerconvolver = fullconv.signal(ptg=ptg, alpha=hwp_angles)


plt.plot(signal_muellerconvolver, label="MuellerConvolver")
plt.legend()
plt.show()
