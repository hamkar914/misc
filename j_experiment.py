'''August 2021, written by Hampus Karlsson, hamka@chalmers.se/hkarlsson914@gmail.com.
   Small Python program for doing some spin dynamics calculations and quickly investigate
   and test some NMR pulse sequences. Based on:

   1. Levitt, M.H. Spin Dynamics, Basics of Nucelar Magnetic Resonance (2001) John Wiley & Sons.
   2. Cavanagh, J. et al. Protein NMR Spectroscopy: Principles and Practice 2nd ed. (2007) Elsevier.
   3. Allard, P., Helgstrand, M. and Hard, T. (1998) Journal of Magnetic Resonance 134, pp.7-16.
   4. Sorensen, O.W. et al. (1983) Progress in NMR Spectroscopy 16, pp. 163-192.

   Probably also inspired by GAMMA,SIMPSON,SpinDynamica etcetera...'''

import numpy as np
from scipy.linalg import expm


# Spin matrices
uni = 0.5 * np.array([[1., 0.],
                      [0., 1.]])

Ix = 0.5 * np.array([[0., 1.],
                     [1., 0.]])

Iy = 0.5j * np.array([[ 0., -1.],
                      [1.,   0.]])

Iz = 0.5 * np.array([[1., 0.],
                     [0.,-1.]])


# Product operators from Pauli spin matrices
puni = 2 * np.kron(uni,uni)
pIz = 2 * np.kron(Iz,uni)
pSz = 2 * np.kron(uni,Iz)
pIzSz = 2 * np.kron(Iz,Iz)

pIx = 2 * np.kron(Ix,uni)
pIy = 2 * np.kron(Iy,uni)
pIxSz = 2 * np.kron(Ix,Iz)
pIySz = 2 * np.kron(Iy,Iz)

pSx = 2 * np.kron(uni,Ix)
pSy = 2 * np.kron(uni,Iy)
pIzSx = 2 * np.kron(Iz,Ix)
pIzSy = 2 * np.kron(Iz,Iy)

pIxSx = 2 * np.kron(Ix,Ix)
pIySy = 2 * np.kron(Iy,Iy)
pIxSy = 2 * np.kron(Ix,Iy)
pIySx = 2 * np.kron(Iy,Ix)


class spin_system:

    '''Class describing a spin system'''

    def __init__(self, prod_op, hard_p90, J):

        self.prod_op = prod_op
        self.hard_p90 = hard_p90
        self.J = J

    def set_operator(self,op):

        self.prod_op = op

    def j_evolve_weak(self, t):

        # J-coupling evolution under weak Hamiltoninan for time, t
        Hj = np.pi*self.J*pIzSz
        self.prod_op = np.dot(expm(-1j*Hj*t),np.dot(self.prod_op,expm(1j*Hj*t)))

    def j_evolve_strong(self, t):

        # J-coupling evolution under strong Hamiltoninan for time, t
        Hj = np.pi * self.J *(pIxSx+pIySy+pIzSz)
        self.prod_op = np.dot(expm(-1j * Hj * t), np.dot(self.prod_op, expm(1j * Hj * t)))

    def pulse_90(self, ph):

        # A perfect 90 degree pulse on both spins, of phase "ph" 0,1,2,3 = x,y,-x,-y
        w1 = 1.0/(4.0*self.hard_p90)
        Hrf = np.cos(ph*(np.pi/2.0)) * 2.0 * np.pi*w1*(pIx+pSx) + np.sin(ph*(np.pi/2.0)) * 2.0 * np.pi*w1*(pIy+pSy)
        self.prod_op = np.dot(expm(-1j*Hrf*self.hard_p90), np.dot(self.prod_op,expm(1j*Hrf*self.hard_p90)))

    def spin1_pulse_90(self, ph):

        w1 = 1.0 / (4.0 * self.hard_p90)
        Hrf = np.cos(ph * (np.pi / 2.0)) * 2.0 * np.pi * w1 * (pIx) + np.sin(ph * (np.pi / 2.0)) * 2.0 * np.pi * w1 * (pIy)
        self.prod_op = np.dot(expm(-1j * Hrf * self.hard_p90), np.dot(self.prod_op, expm(1j * Hrf * self.hard_p90)))

    def spin2_pulse_90(self, ph):

        w1 = 1.0 / (4.0 * self.hard_p90)
        Hrf = np.cos(ph * (np.pi / 2.0)) * 2.0 * np.pi * w1 * (pSx) + np.sin(ph * (np.pi / 2.0)) * 2.0 * np.pi * w1 * (pSy)
        self.prod_op = np.dot(expm(-1j * Hrf * self.hard_p90), np.dot(self.prod_op, expm(1j * Hrf * self.hard_p90)))

    def pulse_180(self, ph):

        # non-selective 180 on both spins
        w1 = 1.0/(4.0*self.hard_p90)
        Hrf = np.cos(ph*(np.pi/2.0)) * 2.0 * np.pi*w1*(pIx+pSx) + np.sin(ph*(np.pi/2.0)) * 2.0 * np.pi*w1*(pIy+pSy)
        self.prod_op = np.dot(expm(-1j*Hrf*2*self.hard_p90),np.dot(self.prod_op,expm(1j*Hrf*2*self.hard_p90)))

    def pulse_dur(self, ph,t):

        # non-selective pulse of duration,t at field strenth 1/(4*p90)
        w1 = 1.0/(4.0*self.hard_p90)
        Hrf = np.cos(ph*(np.pi/2.0)) * 2.0 * np.pi*w1*(pIx+pSx) + np.sin(ph*(np.pi/2.0)) * 2.0 * np.pi*w1*(pIy+pSy)
        self.prod_op = np.dot(expm(-1j*Hrf*t),np.dot(self.prod_op,expm(1j*Hrf*t)))

    def cp(self, ph1H, ph13C,t):

        # Do CP-transfer
        wIx = np.cos(ph1H*0.5*np.pi)*2*np.pi*100.0
        wIy = np.sin(ph1H*0.5*np.pi)*2*np.pi*100.0
        wSx = np.cos(ph13C *0.5* np.pi) * 2 * np.pi * 100.0
        wSy = np.sin(ph13C *0.5* np.pi) * 2 * np.pi * 100.0
        Hrf = np.pi * self.J * (pIzSz) + wIx * pIx + wSx * pSx + wIy * pIy + wSy * pSy
        self.prod_op = np.dot(expm(-1j * Hrf * (t)), np.dot(self.prod_op, expm(1j * Hrf * (t))))

    def display(self):

        # print out the "density matrix"
        print(self.prod_op)

    def check_content(self):

        # Check product operator content of density matrix and print out in readable manner
        poperators = [puni,pIz,pSz,pIzSz,pIx,pIy,
                      pIxSz,pIySz,pSx,pSy,pIzSx,pIzSy,
                      pIxSx,pIySy,pIxSy,pIySx]

        mag_poperator_real = []
        mag_poperator_imag = []

        for pop in poperators:
            mag_poperator_real.append(round(np.trace(np.dot(pop, self.prod_op)).real,3))
            mag_poperator_imag.append(round(np.trace(np.dot(pop, self.prod_op)).imag, 3))

        out_str_real = "real = "
        out_str_imag = "imag = "

        for re in mag_poperator_real:
            out_str_real+="\t"+"{:6.2f}".format(re)

        for im in mag_poperator_imag:
            out_str_imag+="\t"+str("{:6.2f}".format(im))

        print(10*" "+"   E      Iz      Sz    IzSz      Ix      Iy    IxSz    IySz      Sx      Sy    IzSx    IzSy    IxSx    IySy    IxSy    IySx")
        print(out_str_real)
        print(out_str_imag)

    def disp_part_op(self,op):

        # give an operator, calc expectation value for that operator
        out_str=""
        r_part = round(np.trace(np.dot(op, self.prod_op)).real, 3)
        i_part = round(np.trace(np.dot(op, self.prod_op)).imag, 3)
        out_str += "\t" + str("{:6.2f}".format(r_part))
        out_str += "\t" + str("{:6.2f}".format(i_part))
        return out_str

    def content(self):

        # return lists with expectation values for different operators
        poperators = [puni, pIz, pSz, pIzSz, pIx, pIy,
                      pIxSz, pIySz, pSx, pSy, pIzSx, pIzSy,
                      pIxSx, pIySy, pIxSy, pIySx]

        mag_poperator_real = []
        mag_poperator_imag = []

        for pop in poperators:
            mag_poperator_real.append(round(np.trace(np.dot(pop, self.prod_op)).real, 3))
            mag_poperator_imag.append(round(np.trace(np.dot(pop, self.prod_op)).imag, 3))

        return (mag_poperator_real, mag_poperator_imag)

    def content_tuple(self):

        #Returns a tuple containing strings of operators of relevant magnitude.
        poperators = [puni,pIz,pSz,pIzSz,pIx,pIy,
                      pIxSz,pIySz,pSx,pSy,pIzSx,pIzSy,
                      pIxSx,pIySy,pIxSy,pIySx]

        pop_strs = ["uni", "Iz", "Sz", "IzSz", "Ix", "Iy",
                      "IxSz", "IySz", "Sx", "Sy", "IzSx", "IzSy",
                      "IxSx", "IySy", "IxSy", "IySx"]

        mag_poperators_real = []

        for p in range(len(poperators)):
            magreal = round(np.trace(np.dot(poperators[p], self.prod_op)).real,3)
            if  0.05 < magreal or magreal < -0.05:
                mag_poperators_real.append(str(magreal)+pop_strs[p])

        return tuple(mag_poperators_real)


# Define some parameters
# --------------------------
J = 50.0      # Hz
p90 = 10E-6   # s
tau = 0.005   # s
NS = 4

# some phase programs
ph1= np.asarray([0,0,2,2])
ph2= np.asarray([1,1,1,1])

# a start spin system object
spin_sys = spin_system(np.copy(pIz),p90,J)

# some starting operators
ops = [-pIy,-pIy,-pIy,-pIy]

pops_all_scan = []

# Do an actual NMR experiment
# -----------------------------
for scan in range(NS):

    print("Scan "+str(scan+1))

    # change to wanted start operator for this scan
    spin_sys.set_operator(ops[scan])

    # 1. first spin echo
    spin_sys.j_evolve_weak(tau)
    spin_sys.pulse_dur(ph1[scan],19E-6)
    spin_sys.j_evolve_weak(tau)
    print("\nAfter first SE")
    spin_sys.check_content()


    # 90 degree pulse
    spin_sys.pulse_dur(ph2[scan],9.5E-6)
    print("\nAfter p90")
    spin_sys.check_content()


    # second spin echo
    spin_sys.j_evolve_weak(tau)
    spin_sys.pulse_dur(ph1[scan],19E-6)
    spin_sys.j_evolve_weak(tau)
    print("\nafter 2nd SE")
    spin_sys.check_content()
    print("\n\n\n")

    pops_all_scan.append(spin_sys.content()[0])

print("Operators all scans")
print([x for x in np.sum(np.asarray(pops_all_scan),axis=0)/NS])


