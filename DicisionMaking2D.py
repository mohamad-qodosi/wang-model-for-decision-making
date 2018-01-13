import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Introduction
# This must be run as a def -- use the debugger to step through

# Summary: I examine the dynamic model of Wong ang Wang (2006) for decision
# making in a reaction time discrimination task. I exmine the phase space
# and demonstrate how to extract fixed points

# Wong and Wang present a model of LIP activity during the reaction time
# motion direction choice task (Wong and Wang, 2005). This model begins
# with a network of leaky integrate-and-fire neurons in four populations:
#
# 1: Excitatory neurons recieving leftward motion signals
# 2: Excitatory neurons recieving rightward motion signals
# NS: Excitatory neurons recieving constant, nonselective signals
# I: Inhibitory interneurons
#
# L and R recieve motion-related inputs from MT and feed back on themselves
# NS neurons are active at a background level. I neurons are excited by L,
# R, and NS neurons and project inhibitory connections onto all groups
# (including themselves.)
#
# Wong and Wang then employ a series of approximations to reduce their
# model to just two dynamical variables, the mean firing rates of 1 and 2
# populations, plus noise variables. The 2-variable set of differential
# equations they present is:
#
# dS1/dt = -S1/ts + (1-S1)*g*H(x1)
# dS2/dt = -S2/ts + (1-S2)*g*H(x2)
#
# with:
#
# x1 = JN11*S1 - JN12 * S2 + I0 + I1 + Inoise1
# x2 = JN22*S2 - JN21 * S2 + I0 + I2 + Inoise2, and
#
# H(x) = (ax - b)/(1-exp(-d*ax+b))
#
# [these are the further simplified equations presented in the Appendix
# -- I have been unable to make the model presented in the main paper behave.]
#
# I will describe the variables while casting them into MATLAB defs.
# defs & Parameters
# S1, S2 are the dynamical variables, which are the average NMDA gating
# proportion in each population. The NMDA currents are chosen as the model
# parameters because firing rates and AMPA / GABA currents reach
# equilibrium values much faster they are assumed to be near equilibrium
# with the slowly changing NMDA gating.


def dS1(S1, S2, I0, I1, I2, Inoise1, Inoise2):
    # d = dS1 / dt
    d = -S1/tS + (1-S1) * g * H1(Isyn1(S1,S2, I0, I1, Inoise1), Isyn2(S1, S2, I0, I2, Inoise2))
    return d


def dS2(S1, S2, I0, I1, I2, Inoise1, Inoise2):
    # d = dS2 / dt
    d = -S2/tS + (1-S2) * g * H2(Isyn1(S1,S2, I0, I1, Inoise1), Isyn2(S1, S2, I0, I2, Inoise2))
    return d

# The constants JNxy denotes the strength of NMDA synapses from population x
# onto population y:
JN11 = 0.2609 # nA
JN22 = JN11
JN12 = 0.0497 # nA
JN21 = JN12

# I0 gives the background current:
I0 = 0.3255 # nA


# tS is an NMDA-associated time constant,
tS = 0.1 # s #tau_NMDA

# g is a constant relating NMDA gate opening to the presynaptic
# firing rate (eq. 8),
g = 0.641 #unitless #gamma

# Isyn1 and Isyn2 are intermediate values, in units of current, used to
# calculate the firing rate of cells given the dynamical variables of
# stimulus strength (I1, I1) and NMDA gating(S1, S2). (Eqns. 16-19,
# Appendix D of Wong & Wang)

def Isyn1(S1, S2, I0, I1, Inoise1):
    x = JN11 * S1 - JN12 * S2 + I0 + I1 + Inoise1
    return x


def Isyn2(S1, S2, I0, I2, Inoise2):
    x = JN22*S2 - JN21*S1 + I0 + I2 + Inoise2
    return x


# The def H approximates the firing rate of each population in
# terms of the intermediate variables x (Wong & Wang Appendix, Supp. D.)
# FI curve parameters ###################################################
JA_vec = [0, 0.0005, 0.0010, 0.0015, 0.0020]
JA11 = JA_vec[0]
JA22 = JA11
JA12 = 0.1*JA11
JA21 = JA12


def H1(x1, x2):
    a = 270 + 239400 * JA11
    b = 108 + 97000*JA11
    d = 0.1540 - 30*JA11 # Parameters for excitatory cells

    fA1 = JA12 * (-276 * x2 + 106) * (np.sign(x2 - 0.4) + 1) / 2

    h = (a * x1 - b) / (1 - np.exp(-d * (a * x1 - fA1 - b)))
    # To ensure firing rates are always positive (noise may cause negative)
    h[h < 0] = 0
    return h


def H2(x1, x2):
    a = 270 + 239400 * JA22
    b = 108 + 97000 * JA22
    d = 0.1540 - 30 * JA22 # Parameters for excitatory cells

    fA1 = JA21 * (-276 * x1 + 106) * (np.sign(x1 - 0.4) + 1)/2

    h = (a * x2 - b) / (1 - np.exp(-d * (a * x2 - fA1 - b)))
    # To ensure firing rates are always positive (noise may cause negative)
    h[h < 0] = 0
    return h

# The primary source of variability in this model is a noise term that is
# added to each synaptic current:


def dInoise(Inoise):
    d = 1 / tA * -(Inoise + np.random.randn(Inoise.shape[0])
                   * np.sqrt(tA / dt * snoise ** 2))
    return d

#where tA is the AMPA-related time constant and snoise is the amplitude of
#the noise.
tA = 0.002 # s #tau_AMPA
snoise = 0.02 # nA
dt = 0.0005 #s
# Finally we describe the mapping from motion coherence to currents. It is
# linear around a null value:
# coherence Should be in its percentage form (coh = coherence/100).


def stimulus(coh,mu0,ONOFF):
    I1 = JAext * mu0 * (1 + coh) * ONOFF
    I2 = JAext * mu0 * (1 - coh) * ONOFF
    return I1, I2

#where
JAext = 0.2243E-3 # nC, is the AMPA synaptic coupling of the external input
# JAext = 0.00052 #from Wang
mu0 = 30  # Hz, the baseline external input firing rate.

#Finally here is a def to do Euler integration:


def euler(Func, var0, dt, time0, time, skip):
    # var0 = S1
    # var1 = S2
    # var2 = Inoise1
    # var3 = Inoise2

    #The initial contitions have a row for every trial to simulate (done in
    #parallel.)

    #skip is sampling time step

    #The history is returned with seperate trials in rows, each
    #variable in a column, and history along the third dimension.
    if skip in globals():
        skip = 1


    t = np.arange(time0, time + dt * skip, dt * skip)

    history = np.zeros([var0.shape[0], var0.shape[1], t.shape[0]])
    history[:, :, 0] = var0

    for i in range(1, t.shape[0] * skip):
        var0 = var0 + Func(var0) * dt
        if i % skip == 0:
            history[:, :, (i / skip)] = var0

    return t, history


#and a combined def to calculate the step.
def step(x):
    # x = var0
    # dx is a 4 column matrix that holds dS1/dt , dS2/dt, dInoise1/dt , dInoise2/dt
    dx = np.zeros((x.shape[0], x.shape[1]))
    dx[:, 0] = dS1(x[:, 0], x[:, 1], I0, I1, I2, x[:, 2], x[:, 3]) # dS1 / dt
    dx[:, 1] = dS2(x[:, 0], x[:, 1], I0, I1, I2, x[:, 2], x[:, 3]) # dS2 / dt
    dx[:, 2] = dInoise(x[:, 2]) # dInoise1 / dt
    dx[:, 3] = dInoise(x[:, 3]) # dInoise2 / dt
    return dx

# --------------------------------------------------------------------------
# Plot Phase Plane
plt.figure(1)
plt.clf()
plt.subplot(1, 2, 1)

# Let's take a look at the dynamics, in a quiver plot
[S1, S2] = np.meshgrid(np.linspace(0, 0.8, 25), np.linspace(0, 0.8, 25))
cohs = [0, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1]
# ~~> we decide to use a loop in this part
print 'choose coheriance level from'
print cohs
coherLevel = int(input())
coher = cohs[coherLevel]
NFF = 0
if coher != 0:
    NFF = 1

# here I1 and I2 are constant during a trial
# we decide to make it dynamic
[I1, I2] = stimulus(coher,mu0,NFF)

plt.axis('equal')
plt.axis([0,0.8,0,0.8])
plt.xlabel('NMDA gating (right)')
plt.ylabel('NMDA gating (left)')

plt.quiver(S1, S2, dS1(S1, S2, I0, I1, I2, 0, 0), dS2(S1, S2, I0, I1, I2, 0, 0))
plt.show(block=False)

# The three fixed points (a saddle point on the maindiagonal and attractors
# off diagonal) should be apparent. We will return to these later.

# pause
# Plot time trajectories
thresh = .5

# Before delving into analysis, let's show that the model
# works (qualitatively). The basic demonstration is to run a single trial
# and show the evolution of activity over time.
I1, I2 = stimulus(0, 0, 0)
t1, history1 = euler(step, np.array([[0, 0, 0, 0]]), dt, 0, 1, 10)

I1, I2 = stimulus(coher, mu0, NFF)
t2, history2 = euler(step, history1[:, :, -1], dt, t1[-1], 3, 10)

I1, I2 = stimulus(0, 0, 0)
t3, history3 = euler(step, history2[:, :, -1], dt, t2[-1], 4, 10)

t = np.hstack((t1, t2, t3))

# concats history1 and 2 and 3 along the third dimension ~> equal to cat in matlab
history = np.concatenate((history1, history2, history3), axis=2)

# External input time series
I1, I2 = stimulus(coher, mu0, NFF)

I1_vec = I1 * np.hstack((np.zeros((1, len(t1))), np.ones((1, len(t2))), np.zeros((1, len(t3))))) * 10
I2_vec = I2 * np.hstack((np.zeros((1, len(t1))), np.ones((1, len(t2))), np.zeros((1, len(t3))))) * 10


#Here is a representative trace for a single decision trial at the specified coherence:
plt.subplot(1, 2, 2)
plt.ylim([0, 1])
#plt.axis('equal')
# history[0,0,-1] is S1 at end of euler function
if history[0, 0, -1] > thresh:
    plt.plot(t, (history[0, 0, :]).squeeze(), 'r')
else:
    plt.plot(t, (history[0, 0, :]).squeeze(), 'r--')


# history[0,1,-1] is S2 at end of euler function
if history[0, 1, -1] > thresh:
    plt.plot(t, (history[0, 1, :]).squeeze(), 'b--')
else:
    plt.plot(t, (history[0, 1, :]).squeeze(), 'b')

plt.plot(t, I1_vec.transpose(), 'm')
plt.plot(t, I2_vec.transpose(), 'c')
plt.xlabel('t')
plt.ylabel('NMDA gating')
plt.legend(['R', 'L'])
plt.show(block=False)


# Here is its trajectory in phase space:
plt.subplot(1, 2, 1)
tindx = np.where(t<t3[0])[0]
if history[0, 0, -1] > thresh:
    plt.plot((history[0, 0, tindx]).squeeze(), (history[0, 1, tindx]).squeeze(), 'r')
else:
    plt.plot((history[0, 0, tindx]).squeeze(), (history[0, 1, tindx]).squeeze(), 'b')

#plt.axis('equal')
plt.ylim([0, 0.8])
plt.show(block=False)

# pause

## Itretion
#Now I'll compute and plot 9 more trials to show some of the variability.
I1, I2 = stimulus(0, 0, 0)
t1, history1 = euler(step, np.zeros((9,4)), dt, 0, 1, 10)

I1, I2 = stimulus(coher, mu0, NFF)
t2, history2 = euler(step, history1[:, :, -1], dt, t1[-1], 3, 10)

I1, I2 = stimulus(0, 0, 0)
t3, history3 = euler(step, history2[:, :, -1], dt, t2[-1], 4, 10)

t = np.hstack((t1, t2, t3))

history = np.concatenate((history1, history2, history3), axis=2)

for i in range(0, 9):
    a = plt.subplot(1, 2, 2)

    if history[i, 1, -1] > thresh:
        plt.plot(t, (history[i, 0, :]).squeeze(), 'r')
    else:
        plt.plot(t, (history[i, 0, :]).squeeze(), 'r--')

    if history[i, 1, -1] > thresh:
        plt.plot(t, np.squeeze(history[i, 1, :]), 'b--')
    else:
        plt.plot(t, np.squeeze(history[i, 1, :]), 'b')

#    plt.axis('equal')
    a.set_ylim([0, 0.8])

    plt.subplot(1, 2, 1)
    if history[i, 0, -1] > thresh:
        plt.plot(np.squeeze(history[i, 0, tindx]), np.squeeze(history[i, 1, tindx]), 'r')
    else:
        plt.plot(np.squeeze(history[i, 0, tindx]), np.squeeze(history[i, 1, tindx]), 'b')

plt.show(block=False)

# pause
## Plot nullclines and fixed points
# As we can see, the state is first attracted towards the saddle point and
# then it is 'pushed' out towards either attractor.
# Also note the variability of the decision times.

# Now let's examine the behavior ly linearizing around the decision points.

# Start with a vector-to-vector version of the differential relaton:
I1, I2 = stimulus(coher, mu0, NFF)

dS = lambda x: np.array([dS1(np.array([x[0]]), np.array([x[1]]), I0, I1, I2, 0, 0),
                         dS2(np.array([x[0]]), np.array([x[1]]), I0, I1, I2, 0, 0)])[:, 0]


#Let's examine the fixed points.
#First we find out precisely where they fixed points are (starting with reasonable
#guesses):

s0 = [0 , 0 , 0]
s0[0] = fsolve(dS, x0=np.array([0.2, 0.2]))
s0[1] = fsolve(dS, x0=np.array([0.1, 0.7]))
s0[2] = fsolve(dS, x0=np.array([0.7, 0.1]))

s0 = np.array(s0)

#Plot these fixed points on the graph.
plt.scatter(s0[:, 0], s0[:, 1], 100)


#And also plot the nullclines.
S = np.arange(0, 0.8 + 0.01, 0.01)
S10 = np.zeros((len(S), 2))
S20 = np.zeros((len(S), 2))
k = 0
for s in S:
    f1 = lambda x:dS1(s, x, I0, I1, I2, 0, 0)
    f2 = lambda x:dS2(x, s, I0, I1, I2, 0, 0)
    S20[k,:]= fsolve(f1, 0)
    S10[k,:]= fsolve(f2, 0)
    k = k + 1

plt.plot(S10[:, 0], S)
plt.plot(S, S20[:, 0])
plt.show(block=False)

# pause
## Plot the eigen vectors
#The behavior around the fixed points is important. The model gives us a
#first-order, nonlinear equation of the form 
#
# dS/dt = f(S)
#
# where F(S) is quite nonlinear. In the neighborhood of fixed points it
# should behave linearly that is, we want to approximate the above
# equation as:
#
# dS/dT = A*(S-s0)
#
# in the neighborhood of some point s0. A is then the gradient of f.
# Here is a def for a simple finite difference approximation to the
# gradient:

def grad(f, x, delta):
    # Approximate gradient of f at a point x (x being a row vector)
    # Returns a matrix where each row is the derivative along one component
    # of x.
    #
    # That is, f(x) + d * grad(f, x) ~= f(x + d) for small row vectors d.
    
    fx = f(x)
    
    d = np.zeros((x.shape[0], x.shape[0]))
    for i in range(len(x)):
        delt = np.zeros(x.shape)
        delt[i] = delta / 2

        d[i][:] = (f(x + delt) - f(x - delt)) / delta
    return d

# Now we will compute the gradient of dS/dt at every fixed point.

def showEigSys(point, V, D, scale=0.05):
    # shows the eigensystem around a point. THe eigenvectors are in the
    # *rows* of V.
    if scale != None:
        scale = 0.05
    scale = 0.005

    for i in range(D.shape[0]):
        # plot positive eigenvalues with the arrows pointing out in
        # green, negative eigenvalues with the arrows pointing inward in
        # blue.
        vec = V[i][:]
        val = D[i][i]

        if val > 0:
            plt.quiver(np.array([point[0], point[0]]), np.array([point[1], point[1]]),
                       np.array([vec[0], -vec[0]]) * val * scale,
                       np.array([vec[1], -vec[1]]) * val * scale,
                       color='g')#, 'AutoScale', 'off', 'LineWidth', 2)
        else:
            plt.quiver(np.array([point[0], point[0]]) + np.array([-vec[0], vec[0]]) * val * scale,
                       np.array([point[1], point[1]]) + np.array([-vec[1], vec[1]]) * val * scale,
                       np.array([vec[0], -vec[0]]) * val * scale,
                       np.array([vec[1], -vec[1]]) * val * scale,
                       color='b')  # , 'AutoScale', 'off', 'LineWidth', 2)

for point in s0:
    deldF = grad(dS, point, dt)

    #this gives us a matrix. Its eigenvalues diagnose the type of fixed
    #point it is.

    D, V = np.linalg.eig(deldF.transpose()) #here we have set up the components in rows,
                          #thus the transpose
    D = np.diag(D)
    V = V.transpose()

#     point
#     V #for visual inspection
#     D
    
    #plot the eigensystem
    showEigSys(point, V, D) #showEigSys is defined at the bottom of the file.
plt.show(block=False)

#pause    
## Sychometric ...
#-------------------------------------------------------------------------

# Now let's explore the predicitons of the model. In the reaction-time
# dots task, we presume that the animal is comparing time-varying evidence
# for leftward motion with time-varying evidence for rightward motion. We
# believe that a statistically optimal solution for two targets is
# accumulation to a bound, where the bound gradually decreases. This makes
# intuitive sense: as more evidence is accumulated, more of the decision
# variable is due to noise, so that the probability of a correct decision
# reduces over time. Eventually it reduces enough that it is not worth
# collecting more evidence. The optimal shape of the bound would be
# determined by the timing and reward structure of the task as well as the
# reliability of the evidence reaching the point where it is accumulated.
#
# But we can see that the Wong & Wang model follows different dynamics. The
# activity starts out on a 'hill' and once the state of the system is
# perturbed off that hill, the system will fall down into one of two
# attractor basins (Wong & Wang Fig. 4D, also A. Roxin SFN abstract, 2006).
#
# This would predict some strange consequences in the data it would be
# more difficult for the decision variable to move back to the other side
# once it has started to move off the saddle point.

# Let's generate some data: 500 trials at each of 7 coherence levels (this
# takes a while):
n = 500
# 500 trials for each coherency level => 3500 trails
cohs = np.array([0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1])
coh = np.tile(cohs, (n, 1))
group = np.tile(np.arange(0, len(cohs)), (n, 1))
coh = coh.flatten()
group = group.flatten()
I1, I2 = stimulus(coh, mu0, 1)

t, history = euler(step, np.zeros((len(coh), 4)), dt, 0, 4, 10)
#plot a sampling of of them, to see what's going on. Here the colors range
#from red for low coherence trials to blue for high coherence trials.
plt.figure(3)
plt.clf()
for i in range(0, coh.shape[0], n / 20):
    c = max(np.log(coh[i]), np.log(0.016)) / np.log(0.016)

    plt.plot(t, history[i, 0, :], color=(c, 0, 1 - c))

plt.xlabel('time (s)')
plt.ylabel('S')
plt.show(block=False)

#pause


##
# Now let's plot psychometric and chronometric defs.
# We will set the decision threshold to be when either S1 or S2 crosses 0.5.
# thresh    = 15        # Decision threshold
# thresh_s  = (gamma*Tnmda*thresh/1000)/(1+gamma*Tnmda*thresh/1000) # Threshold in s-space

thresh = 0.5

def decisiontimes(t, history, thresh):
    # each row gets the NMDA variable or each trial

    # Rin is activity of spiking group A per time
    Rin = np.squeeze(history[:, 0, :])
    #Rout is activity of spiking group B per time
    Rout = np.squeeze(history[:, 1, :])

    # _in is true for when the decision of group A is selected and false otherwise
    _in = (Rin > thresh)
    # _out is true for when the decision of group B is selected and false otherwise
    _out = (Rout > thresh)

    # tin is decision time for trials in which group A win
    tin = np.min(_in * np.tile(t, (_in.shape[0], 1)) + max(t) * np.invert(_in), 1)
    # iin is index of tin
    iin = np.argmin(_in * np.tile(t, (_in.shape[0], 1)) + max(t) * np.invert(_in), 1)
    # tout is decision time for trials in which group B win
    tout = np.min(_out * np.tile(t, (_out.shape[0], 1)) + max(t) * np.invert(_out), 1)
    # iout is index of iout
    iout = np.argmin(_out * np.tile(t, (_out.shape[0], 1)) + max(t) * np.invert(_out), 1)

    # if group A win , t for group B is 4
    # choice is true when group A win and false when group B win
    choice = tin < tout
    # min (decision time, 4) => decision time
    # time is decision time  regarding selected choice
    time = np.min(np.array([tin, tout]), 0)
    # iter is decision time index regarding selected choice
    iter = iin * choice + iout * np.invert(choice)
    return choice, time, iter

#calculate choice and decision time for each trial
choice, time, iter = decisiontimes(t, history, thresh)

#Assort the choices and decision times by coherence level

p = np.array([0. for i in range(len(cohs))])
tin_ = np.array([0. for i in range(len(cohs))])
tin_stdev = np.array([0. for i in range(len(cohs))])
tout_ = np.array([0. for i in range(len(cohs))])
tout_stdev = np.array([0. for i in range(len(cohs))])

for i in range(len(cohs)):
    p[i] = np.mean(choice[group == i])
    tin_[i] = np.mean(time[(group == i) & (choice == 1)])
    tin_stdev[i] = np.std(time[(group == i) & (choice == 1)])

    outs = time[(group == i) & (choice == 0)]

    tout_[i] = np.mean(outs)
    tout_stdev[i] = np.std(outs)
    
    if outs.size < 20:
        tout_[i] = None
        tout_stdev[i] = None

tout_ = tout_[tout_ != None]
tout_stdev = tout_stdev[tout_stdev != None]

plt.figure(4)
plt.clf()
plt.subplot(1, 2, 1)
plt.semilogx(cohs, p * 100, 'b.')
plt.xlabel('Coherence Level')
plt.ylabel('Correct #')
plt.axis('equal')
plt.show(block=False)

#

ax = plt.subplot(1, 2, 2)
plt.errorbar(cohs, tin_, yerr=tin_stdev, fmt='b.')
plt.errorbar(cohs[:tout_.shape[0]], tout_, yerr=tout_stdev, fmt='r.')
ax.set_xscale("log", nonposx='clip')
plt.axis('equal')
plt.xlabel('Coherence Level')
plt.ylabel('Response Time')
plt.show(block=False)

#pause
##
coh = 0.064
mus = range(0, 101, 10)
mu = np.tile(mus, (n, 1))
group = np.tile(range(len(mus)), (n, 1))
mu = mu.flatten()
group = group.flatten()
I1, I2 = stimulus(coh, mu, 1)

t, history = euler(step, np.zeros((len(mu), 4)), dt, 0, 4, 10)

choice, time, iter = decisiontimes(t, history, thresh)

#Assort the choices and decision times by coherence level

p = np.array([0. for i in range(len(mus))])
tin_ = np.array([0. for i in range(len(mus))])
tin_stdev = np.array([0. for i in range(len(mus))])
tout_ = np.array([0. for i in range(len(mus))])
tout_stdev = np.array([0. for i in range(len(mus))])

for i in range(len(mus)):
    p[i] = np.mean(choice[group == i])
    tin_[i] = np.mean(time[(group == i) & (choice == 1)])
    tin_stdev[i] = np.std(time[(group == i) & (choice == 1)])

    outs = time[(group == i) & (choice == 0)]

    tout_[i] = np.mean(outs)
    tout_stdev[i] = np.std(outs)

    if outs.size < 20:
        tout_[i] = None
        tout_stdev[i] = None

tout_ = tout_[tout_ != None]
tout_stdev = tout_stdev[tout_stdev != None]

plt.figure(5)
plt.clf()
plt.subplot(1, 2, 1)
plt.plot(mus, p * 100, 'b.')
plt.xlabel('\mu_0')
plt.ylabel('Correct #')
plt.axis('equal')
plt.show(block=False)

#
plt.subplot(1, 2, 2)
plt.errorbar(mus, tin_, tin_stdev, tin_stdev, 'b.')
plt.errorbar(mus[:tout_.shape[0]], tout_, tout_stdev, tout_stdev, 'r.')
plt.xlabel('External Stimulus Strength: \mu_0(Hz)')
plt.ylabel('Reaction Time')
plt.show(block=False)

'''
#pause

##
# These graphs show the model's psychometric and chronometric defs. 
# Note that the distribution (as opposed to mean) of the response time is
# not examined in Wong & Wang and may provide a critical comparison with
# experimental data.
#-------------------------------------------------------------------------

# In the last part, I will attempt to compare the dynamical model with
# a diffusion-based model to see if they explain the same decisions and
# stopping times.
#
# The simulation allows me to make a more direct comparison of
# the dynamic model to a diffusion-based model, because I can use the
# same noise in both.
#
# Here's what I mean. I will simply integrate the noise records saved
# from the previous simulation for as long as as the decision lasted
# this allows us to compare the position at which a diffusion would be
# with the same history of evidence. The comparison will be the stopping
# time of the dynamical model versus the integrated positon of a
# diffusion model at that time.

#integrate, then pick out values at the right time
noise_right = np.cumsum(np.squeeze(history[:,2,:]), 1)
noise_left = np.cumsum(np.squeeze(history[:,3,:]), 1)
noise_right = noise_right[np.ravel_multi_index((np.array(range(len(cohs))).transpose(), iter),
                                            noise_right.shape)]
noise_left = noise_left[np.ravel_multi_index((np.array(range(len(cohs))).transpose(), iter),
                                            noise_left.shape)]

# Now which diffusion model do I compare it with? The best one:
# I will determine scaling coefficients for the noise and drift rate
# that best fit the simulated data from Wong & Wang. This amounts to
# performing this regression against the coherence*decision time and the 
# accumulated noise to explain the choices:
design = [coh * time, noise_right - noise_left]
[B BINT] = regress((choice - 0.5) * 2, design)

# Now we scatter-plot the fit against the response time, with colors
# indicating the coherence level and choice.

# The variation with coherence may tell us something interesting about how
# the models compare: I will contrive the colors to vary from black to
# green according to coherence, for choices in, and red to black for
# choices out.
scale = 1 - (log(coh) / min(log(coh)))
colors = (scale * [0 1 0])  *  repmat(choice, 1, 3) + ((1-scale) * [1 0 0])  *  repmat(~choice, 1, 3)

#form a scatter plot with response time on the horizontal axis and diffusion
#parameter on the vertical. Plot correct choices as circles adn incorrect
#as triangles.
figure(5) clf hold on
scatter(time(choice), design(choice, :)*B, 25, colors(choice, :), 'filled')
scatter(time(~choice), design(~choice, :)*B, 49, colors(~choice, :), '^', 'filled')

#pause

# There is an interesting phenomenon here: with a best fit of the drift rate,
# the diffusion model predicts a faster decision for
# high coherences than the dynamic model. Moreover, as time goes on, the
# discrepancy between diffusion decision time and dynamic decision time
# increases (the upward slope of the clouds of the highest coherences.)
# Thus the decision times are not well correlated between the diffusion and
# dynamic models.

# However, there is a clear association  betwen the accumulated noise and
# the error trials, at least for the early errors.

# Perhaps the Wong & Wang model is effectively
# discarding the early information from the first hundred or so
# milliseconds of the stimulus. This could inform a test against the 

# Finally try incorporating more variables into the regression -- separate
# left and rightward noise with variance scaling with the coherence.
# Coherence is also added as an explanatory variable. to account for losing
#information from early in the stimulus:
design = [coh * time, noise_right, noise_left, coh, coh * noise_right, coh * noise_left]
[B, BINT] = regress((choice - 0.5) * 2, design)

#The regression coefficients in this case are interesting -- do the
#diffusion models we play with scale the variance according to the stimulus
#strength?

figure(6) clf hold on
scatter(time(choice), design(choice, :)*B, 25, colors(choice, :), 'filled')
scatter(time(~choice), design(~choice, :)*B, 49, colors(~choice, :), '^', 'filled')

#pause

#This further highlights the change in behavior per-coherence between
#the diffusion-based and dynamic decision model.
# -------------------------------------------------------------------------
'''

plt.show()
