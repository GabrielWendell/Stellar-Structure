##### Olivier Roth
##### Physique Numérique, 2019
##### Hydrodynamique dans l'approximation quasi-statique : structure interne des etoiles


### Importation des modules
import numpy as np , matplotlib.pyplot as plt , time , warnings
import os , sys , contextlib
from matplotlib.pyplot import cm
from scipy.integrate import odeint


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Constantes (unités du SI)
G=6.67259e-11    # constante de gravitation
a=7.5646*1e-16    # constante de radiation
c=2.9979*1e8    # vitesse de la lumière
mp=1.67e-27    # masse du proton
X=0.70    #  fraction massique d'Hydrogène (Soleil)
Z=0.02    # métallicité (Soleil)
Y=1-X-Z    # fraction massique d'Hélium (Soleil)
k=1.380649e-23  # constante de Boltzmann
sigma=a*c/4    # constante de Stefan-Boltzmann
gamma=5/3    # indice adiabatique
R_gaz=8.3145e3    # constante des gaz parfaits
mu=1/(2*X+3/4*Y+1/2*Z)    # poids moléculaire moyen
kappa_es = 0.02*(1+X)    # opacité de la diffusion Thomson


### Constantes liées au Soleil :
M_S = 1.9891*1e30 # kg
R_S = 6.9598*1e8 # m
P0_S = 2.4e16 # Pa
T0_S = 15e6 # K
L_S = 3.8515*1e26 # W


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Densité
def rho(p,t):
    return (p - (a/3)*(t**4))*mu/(R_gaz*t)


# Opacité
def Kappa_ff(p,t):
        return 1.0e24 * (1+X) * (Z+0.0001)* ((rho(p,t)/1e3)**0.7) * (t**(-3.5))

def Kappa_Hminus(p,t):
        return 2.5e-32 * (Z/0.02) * ((rho(p,t)/1e3)**0.5) * ((t)**9.0)

def Kappa(p,t):
    if opak==True: # définition complète de l'opacité
        if t > 1e4:
            Kappa = ( 1/Kappa_Hminus(p, t) + 1/max(kappa_es, Kappa_ff(p,t)) )**-1
        else:
            Kappa = (1/Kappa_Hminus(p,t) + 1/min(Kappa_ff(p,t), kappa_es))**-1
    if opak==False: # uniquement la contribution de la diffusion Thomson
        Kappa=kappa_es
    return Kappa


# Taux de production d'énergie
def eps_pp(t): # chaîne PP
    T6 = t/1e6
    return 0.241*X**2*np.exp(-33.8*T6**(-1/3))

def eps_cno(t): # cycle CNO
    T6=t/1e6
    return 8.7e20*X*Z/2*np.exp(-152.28*T6**(-1/3))

def eps_trA(p,t): # processus triple alpha
    T6=t/1e6 ; T8=t/1e8
    return 5.1*1e1 * (Y**3) * rho(p,t) * T8**(-3)/T6**(-2/3) * np.exp(-44/T8)

def epsilon(p,t):
    T6=t/1e6
    return ( eps_pp(t)+eps_cno(t)+eps_trA(p,t) ) *rho(p,t)* T6**(-2/3)


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Principales équations

def dr_m(r,p,t): # Continuité de masse
    return 1/(4*np.pi*rho(p,t)*(r**2))

def dP_m(m,r): # Equilibre hydrostatique
    return -G*m/(4*np.pi*(r**4))

def dL_m(p,t): # Equilibre thermique
    return epsilon(p,t)


def dT_m_radiative(r,p,l,t): # transport radiatif
        return -3/(4*a*c)*(Kappa(p,t)/(t**3))*l/((4*np.pi*(r**2))**2)

def dT_m_convective(m,r,p,t): # transport convectif
        return -(1-1/gamma)*(t/p)*(G*m*rho(p,t)/r**2)*dr_m(r,p,t)

def dT_m(m,r,p,l,t): # Transport d'énergie
    if rad_conv==True: # radiatif ou convectif
        return -min( abs(dT_m_radiative(r,p,l,t)) , abs(dT_m_convective(m,r,p,t)))
    if rad_conv==False: # uniquement radiatif
        return dT_m_radiative(r,p,l,t)


def DD(x,m): # toutes les principales équations en une définiton, pour utiliser odeint
    r=x[0] ; p=x[1] ; l=x[2] ; t=x[3]
    return [ dr_m(r,p,t), dP_m(m,r), dL_m(p,t), dT_m(m,r,p,l,t) ]


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Gradient de température
def Delta_rad(m,p,l,t,kappa): # gradient radiatif
    return 3*p*l*kappa/(16*np.pi*a*c*G*m*(t**4))

def Delta_ad(): # gradient adiabatique
    return 1-1/gamma


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Calcul de la sensibilité à la température des deux taux de production d'énergie (chaîne pp et cycle CNO) autour de T0
def T_sensitivity(Q, T0, X, Z):
    t = 1.e-8
    q1 = Q(X, Z, T0) ; q2 = Q(X, Z, T0*(1+t))
    dlogq_dlogT = (T0/q1)*(q2-q1)/(T0*t)
    return dlogq_dlogT


def q_pp(X,Z,t): # chaîne PP taux de production d'énergie
    T6 = t/1e6
    return 0.241*X**2*np.exp(-33.8*T6**(-1/3))*T6**(-2/3)

def q_cno(X,Z,t): # cycle CNO taux de production d'énergie
    T6=t/1e6
    return 8.7e20*X*Z/2*np.exp(-152.28*T6**(-1/3))*T6**(-2/3)


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Utilisé pour le graphique de l'opacité
def visu_kappa(p,t):
    Kappa_i=np.zeros(len(t))
    for i in range(len(t)):
        if t[i] > 1e4:
            Kappa_i[i] = ( 1/Kappa_Hminus(p[i], t[i]) + 1/max(kappa_es, Kappa_ff(p[i],t[i])) )**-1
        else:
            Kappa_i[i] = (1/Kappa_Hminus(p[i],t[i]) + 1/min(Kappa_ff(p[i],t[i]), kappa_es))**-1
    return Kappa_i


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Estimation des conditions initiales et conditions aux bords :
# (On suppose que l'étoile se trouve dans la séquence principale)

def radius(Mr):
    if Mr<1.6*M_S:
        rmax = 0.89*R_S*(Mr/M_S)**(0.8)
    if Mr>=1.6*M_S:
        rmax = 1.01*R_S*(Mr/M_S)**(0.57)
    return rmax

def luminosity(Mr):
    if Mr<0.43*M_S:
        L_rmax=0.23*L_S*(Mr/M_S)**(2.3)
    if Mr>=0.43*M_S and Mr<2*M_S:
        L_rmax=L_S*(Mr/M_S)**4
    if Mr>=2*M_S and Mr<55*M_S:
        L_rmax=1.4*L_S*(Mr/M_S)**(3.5)
    if Mr>=55*M_S:
        L_rmax=32000*L_S*(Mr/M_S)
    return L_rmax

def central_pressure(Mr,rm):
    return 3*G*(Mr**2)/(8*np.pi*rm**4)*160

def central_temp(Mr,rm):
    return 2*G*Mr*mp/(3*k*rm)


### Conditions aux bords
def boundary_cond(Mstar,*args):
    if Mr==M_S: # conditions du Soleil
        Rm=R_S ; Lm=L_S ; Pc=P0_S ; Tc=T0_S
    else : # conditions calculées
        Rm=radius(Mr) ; Lm=luminosity(Mr)
        Pc=central_pressure(Mr,Rm) ; Tc=central_temp(Mr,Rm)
    if args:
        Rm=radius(Mr) ; Lm=luminosity(Mr)
        Pc=central_pressure(Mr,Rm) ; Tc=central_temp(Mr,Rm)
    return Pc, Tc, Rm, Lm


### Conditions initiales
def init_cond(mr,pc,tc,rm,lm):
    m0=mr/1000 ; r0=(m0/(4/3*np.pi*rho(pc,tc)))**(1/3)
    Lr0=epsilon(pc,tc)*m0
    P_rmax=(2/3)*(G*mr/rm**2)*(1/kappa_es)
    T_rmax=(lm/(4*np.pi*sigma*(rm**2)))**(1/4)
    return m0,r0,Lr0,P_rmax,T_rmax


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Suppression du message 'lsoda warning' lors de certaines intégrations

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()
        try:
            os.dup2(fileno(to), stdout_fd)
        except ValueError:
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)
        try:
            yield stdout
        finally:
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Fitting method
def odeint_for_Fit(Mstar, yi, pct, precis_up): # retourne un tableau des différences au point de rencontre, donné par intégration avec odeint

    # yi  : valeurs initiales de P0 , T0 , rmax , L_rmax
    # pct : point de rencontre ; ex 0.5 -> 50% de rmax
    # precis_up : utilisé pour augmenter la précision de l'intégration

    p0=yi[0] ; t0=yi[1] ; r2=yi[2] ; l2=yi[3]
    m0,r0,Lr0,P_rmax,T_rmax = init_cond(Mstar,p0,t0,r2,l2)

    y0=[r0,p0,Lr0,t0]
    y2=[r2, P_rmax, l2, T_rmax]
    m=np.linspace(m0,Mstar*pct,N1)
    if precis_up==True:
        m2=np.linspace(Mr,Mstar*pct,N1*100)
    else:
        m2=np.linspace(Mr,Mstar*pct,N1)

    with stdout_redirected():
        x=odeint(DD,y0,m) # résolution avec scipy.integrate
        x2=odeint(DD,y2,m2)

    R=x[:,0] ; P=x[:,1] ; L=x[:,2] ; T=x[:,3]
    R2=x2[:,0] ; P2=x2[:,1] ; L2=x2[:,2] ; T2=x2[:,3]

    Pdiff=(P2[-1]-P[-1]) # différences au point de rencontre
    Tdiff=(T2[-1]-T[-1]) # .
    Rdiff=(R2[-1]-R[-1]) # .
    Ldiff=(L2[-1]-L[-1]) # .

    tab=[Pdiff,Tdiff,Rdiff,Ldiff]
    return tab , [ P[0], T[0], R2[0], L2[0] ]

def tab_div(Mstar, *args): # retourne un nombre pour réduire la valeur des modifications des conditions initiales (pour éviter la divergence) en fonction de la masse initiale de l'étoile et de certains paramètres physiques

    if opak==True and rad_conv==True:
        if Mstar>=0.4*M_S :
            Tab_div=3
            if Mstar>=2.5*M_S and Mstar<=7*M_S:
                Tab_div=1
        if Mstar<0.4*M_S :
            Tab_div=6

    if opak==False and rad_conv==True:
        Tab_div=10

    if opak==False and rad_conv==False:
        Tab_div=10

    if opak==True and rad_conv==False:
        Tab_div=10
        print("\n\nImpossible de calculer ce modèle")
        print("Le modèle avec définition complète de l'opacité et sans convection n'est pas calculable\n\n")

    if args:
        Tab_div=args
    return Tab_div

def Fit(Mstar, yi, pct, eps, precis_up, *args): # fitting method
    # yi  : valeurs initiales de P0 , T0 , rmax , L_rmax
    # pct : point de rencontre ; ex 0.5 -> 50% de rmax
    # eps : tolérance de la différence au point de rencontre entre l'intégration intérieure et extérieure
    # precis_up : utilisé pour augmenter la précision de l'intégration

    StopIteration=False
    i=0
    Y0=[]
    Tab_div = tab_div(Mstar,*args)

    if animated==True: # un tracé à chaque itération pour voir l'évolution
        plt.ion()

    while StopIteration==False:
        print(i,end='\r')

        tab=odeint_for_Fit(Mstar,yi,pct,precis_up)[0] #Ddiff
        lim_norm=odeint_for_Fit(Mstar,yi,pct,precis_up)[1]

        Y0.append([yi[0], yi[1], yi[2], yi[3]])
        delta_diff=[0,0,0,0] ; d_pardiff_d_yi=[0,0,0,0]

        for j in range(4): # création de la Jacobienne
            modif_yi=yi[j]*1e-4
            yi[j]+=modif_yi # modification locale dans les conditions aux bords
            new_tab=odeint_for_Fit(Mstar,yi,pct,precis_up)[0] # newDdiff
            delta_diff[j]=np.array(new_tab)-np.array(tab) # newDdiff - Ddiff
            d_pardiff_d_yi[j]=delta_diff[j]/modif_yi # création de la matrice
            yi[j]-=modif_yi

        d_pardiff_d_yi=np.transpose(d_pardiff_d_yi) # transpose la matrice
        inv_M=np.linalg.inv(d_pardiff_d_yi) # inverse la matrice

        Delta=np.dot(inv_M , np.array(tab)/Tab_div ) # résoud le système

        for j in range(4):
            yi[j]-=Delta[j] # Pc -> Pc - DeltaPc ...

        if (abs(np.array(tab)/np.array(lim_norm))<eps).all(): # arrête la boucle si la convergence est atteinte
            print('Converge en {} itérations.\n'.format(i))
            StopIteration=True

        if args:
            if i==150: # arrête la boucle en cas de divergence ou de convergence trop longue
                print('break à i=',i)
                StopIteration=True
        else:
            if i==10 and rad_conv==True: # pour accélérer le processus à la fin si cela prend trop de temps
                    Tab_div=2

            if i==65: # arrête la boucle en cas de divergence ou de convergence trop longue
                print('break à i=',i)
                StopIteration=True

        if animated==True: # un tracé à chaque itération pour voir l'évolution
            p0=yi[0] ; t0=yi[1] ; r2=yi[2] ; l2=yi[3]
            m0,r0,Lr0,P_rmax,T_rmax = init_cond(Mstar,p0,t0,r2,l2)

            y0=[r0,p0,Lr0,t0] ; y2=[r2, P_rmax, l2, T_rmax]
            m=np.linspace(m0,Mstar*pct,N1) ; m2=np.linspace(Mr,Mstar*pct,N1)

            with stdout_redirected():
                x=odeint(DD,y0,m) ; x2=odeint(DD,y2,m2)

            plt.clf()
            yi_ordr2=[yi[2],yi[0],yi[3],yi[1]]
            cols=['b','k','g','r'] ; leg=['r(m)','P(m)','L(m)','T(m)']
            for j in range(4):
                plt.plot(m/Mstar,x[:,j]/yi_ordr2[j],c=cols[j],label=leg[j])
                plt.plot(m2/Mstar,x2[:,j]/yi_ordr2[j],c=cols[j])
            plt.xlabel("M/$M_{\odot}$")
            plt.ylabel("R/$R_{\odot}$, P/$P0_{\odot}$, L/$L_{\odot}$, T/$T0_{\odot}$")
            plt.legend(loc=7, framealpha=0.9)
            plt.draw() ; plt.pause(1e-1)

        i+=1
    nb_i=np.arange(i)

    if animated==True:
        plt.ioff() ; plt.xlabel("M/$M_{\odot}$") ; plt.show()

    Y0_norm=[]
    for i in range(4):
        Y0_norm.append(np.array(Y0)[:,i])
        Y0_norm[-1]/=lim_norm[i]
    Y0_norm=np.transpose(Y0_norm)

    return yi, Y0_norm, nb_i

def res_Fit(Mstar, eps, pct, precis_up, *args): # résultats de la fitting method
    Pc, Tc, Rm, Lm = boundary_cond(Mr,*args)
    Yi=[Pc,Tc,Rm,Lm]

    ###
    yi , Y0, nb_i= Fit(Mstar,Yi,pct,eps,precis_up,*args)
    ###

    p0=yi[0] ; t0=yi[1] ; r2=yi[2] ; l2=yi[3]
    m0,r0,Lr0,P_rmax,T_rmax = init_cond(Mstar,p0,t0,r2,l2)

    y0=[r0,p0,Lr0,t0]
    y2=[r2, P_rmax, l2, T_rmax]
    m=np.linspace(m0,Mstar*pct,N1)
    if precis_up==True:
        m2=np.linspace(Mr,Mstar*pct,N1*100)
    else:
        m2=np.linspace(Mr,Mstar*pct,N1)

    with stdout_redirected():
        x=odeint(DD,y0,m)
        x2=odeint(DD,y2,m2)

    R=x[:,0] ; P=x[:,1] ; L=x[:,2] ; T=x[:,3]
    R2=x2[:,0] ; P2=x2[:,1] ; L2=x2[:,2] ; T2=x2[:,3]
    return [m,m2], [R,R2,P,P2,L,L2,T,T2], yi , y0, y2, Y0, nb_i


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Contrôle des graphiques et paramètres physiques

# Paramètres physiques
opak=True # définition complète de l'opacité
rad_conv=True # convection possible dans l'étoile


# Fitting method et toutes les définitions qui en dépendent
Fit_method=True # fitting method intégration
Pct=0.5 # point de rencontre (pourcentage de la masse de l'étoile)
if Fit_method==True:
    evo_yi_i=True # évolution des conditions initiales
    mutliP_sep_var=True # évolution des variables jusqu'à la convergence
    visu_opacity=True # opacité dans l'étoile
    grad_T=True # gradients de température


Multi_fit=False # permet la création du diagramme H-R et autres graphs utilisant plusieurs masses
if Multi_fit==True:
    H_R=True # création de diagramme de Hertzsprung Russell
    if H_R==True:
        some_stars=False # H-R seulement avec 7 étoiles connues
    Rho_c_M=True # densité centrale en fonction de Mr
    visu_reg=True # délimitation des zones radiative/convective et autres informations
    ecart_yi=False # modifications dans les conditions intiales en fonction de Mr


multi_pct=False # fitting method avec différents points de rencontre

change_compos=False # changement dans la composition de l'étoile (choisir de changer X ou Z)

PP_CNO=False # taux de production d'énergie

simple_odeint=False # simple intégration avec les conditions initiales déjà pré-définies pour une étoile de la masse du Solei


info=True # informations physiques et numériques sur l'intégration effectuée


# Masss de l'étoile
Mmulti=1
if Fit_method==True: # choisir une masse solaire
    Mmulti=float(input("Masse solaire (voir 'readme.txt' pour les possibilités) : "))
Mr = M_S * Mmulti


# l'étoile est divisée en N1 parties pour l'intégration
N1=int(1e4)
precis_up=False # pour obtenir une meilleure précision sur le résultat, le temps de traitement est également augmenté
# (utile pour visu_reg pour voir l'enveloppe convective du Soleil)


# Evolution de la structure au cours de la fitting method
animated=True # pour voir l'évolution, avec plt.ion()


Time=True # pour voir le temps de d'exécution du programme
tmp1=time.time()


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if Fit_method==True: # fitting method
    Pc, Tc, Rm, Lm = boundary_cond(Mr)

    eps=2e-3 # précision de l'integration, 0.2% de différence au point de rencontre
    [m,m2], [r,r2,P,P2,Lr,Lr2,T,T2], yi , y0, y2, Y0, nb_i = res_Fit(Mr,eps,Pct,precis_up)

    P0=yi[0] ; T0=yi[1] ; rmax=yi[2] ; L_rmax=yi[3]
    r0=y0[0] ; Lr0=y0[2] ; P_rmax=y2[1] ; T_rmax=y2[3]


if simple_odeint==True: # simple intégration
    Mr = M_S ; Rm=R_S ; Lm=L_S ; Pc=P0_S ; Tc=T0_S

    if opak==False and rad_conv==False:
        po_fact=1.11340817858 ; to_fact=1.43893114066
        rm_fact=0.589861009529 ; lm_fact=8.06299195186

    if opak==False and rad_conv==True :
        po_fact=1.6759807663 ; to_fact=1.27624845522
        rm_fact=0.564103089086 ; lm_fact=8.80033557387

    if opak==True and rad_conv==True:
        po_fact=0.890829929422 ; to_fact=0.962626132095
        rm_fact=1.09756978912 ; lm_fact=1.19856908279

    if opak==True and rad_conv==False:
        print("Impossible de calculer ce modèle")

    P0=Pc*po_fact ; T0=Tc*to_fact ; rmax=Rm*rm_fact ; L_rmax=Lm*lm_fact
    m0,r0,Lr0,P_rmax,T_rmax = init_cond(Mr,P0,T0,rmax,L_rmax)

    y0=[r0,P0,Lr0,T0] ; y2=[rmax, P_rmax, L_rmax, T_rmax]
    m=np.linspace(m0,Mr*Pct,N1) ; m2=np.linspace(Mr,Mr*Pct,N1)

    with stdout_redirected():
        x=odeint(DD,y0,m) ; x2=odeint(DD,y2,m2)

    r=x[:,0] ; P=x[:,1] ; Lr=x[:,2] ; T=x[:,3]
    r2=x2[:,0] ; P2=x2[:,1] ; Lr2=x2[:,2] ; T2=x2[:,3]
    yi=[P[0],T[0],r2[0],Lr2[0]]


if info==True: # informations sur l'étoile calculée
    if Fit_method==True:
        print('Calculs avec {} masse solaire.'.format(Mmulti))
        print("\nFraction massique de l'Hydrogène :",X)
        print("Fraction massique de l'Helium :",Y)
        print('Métallicité :',Z)
        print("\nCoeur de l'étoile :")
        print("M init : {} % de Mstar".format(round(100*m[0]/Mr,2)))
        print("R init : {} % de Rstar".format(round(100*r[0]/rmax,2)))

    if opak==True:
        print("\nCalculs avec la définition complète de l'opacité.")
    if opak==False:
        print("\nCalculs avec l'opacité constante.")

    if rad_conv==True:
        print("Transport d'énergie par radiations ou par convection.")
    if rad_conv==False:
        print("Transport d'énergie uniquement radiatif.")

    if Fit_method==True:
        print("\n\nIntégration avec la fitting method :")
        print("Point de rencontre à {}% de Mr".format(Pct))
        print("Précision de l'intégration (ratio maximum au point de rencontre): {}".format(eps))

        print('\nModifications dans les conditions initiales pour obtenir le modèle :')
        print('x Pc :',round(P0/Pc,3))
        print('x Tc :',round(T0/Tc,3))
        print('x Rm :',round(rmax/Rm,3))
        print('x Lm :',round(L_rmax/Lm,3))

        print("\nConditions initiales calculées par rapport aux conditions aux limites connues du Soleil :")
        print('xP0_S :',round(P0/P0_S,3))
        if Mr==M_S:
            print('xT0_S :',round(T0/T0_S,3), ' ; T_surf :',int(T2[0]), ', T_surf Sun :',5778)
        else :
            print('xT0_S :',round(T0/T0_S,3))
        print('xR_S :',round(rmax/R_S,3))
        print('xL_S :',round(L_rmax/L_S,3))

        if opak==True and rad_conv==True:
            mm=np.concatenate((m,m2[::-1]))
            rr=np.concatenate((r,r2[::-1])) ; pp=np.concatenate((P,P2[::-1]))
            ll=np.concatenate((Lr,Lr2[::-1])) ; tt=np.concatenate((T,T2[::-1]))

            Kappa_i=visu_kappa(pp,tt)
            d_rad=Delta_rad(mm,pp,ll,tt,Kappa_i) ; d_ad=Delta_ad()

            idx = np.argwhere(np.diff(np.sign(d_rad - d_ad)) != 0).reshape(-1) + 0 # index of difference between d_rad and d_ad

            if len(idx)==0:
                print("\nÉtoile avec transport d'énergie uniquement par radiations.")
            else :
                if d_rad[0]>d_ad:
                    if d_rad[-2]<d_ad:
                        print("\nDu transport d'énergie convectif au transport d'énergie radiatif à {} % de la masse et à {} % du rayon de l'étoile.\n".format(round(100*mm[idx[0]]/Mr,2), round(100*rr[idx[0]]/rmax,2)))
                    else:
                        print("\nÉtoile avec transport d'énergie uniquement par convection.")
                if d_rad[0]<d_ad:
                    print("\nDu transport d'énergie radiatif au transport d'énergie convectif à {} % de la masse et à {} % du rayon de l'étoile.\n".format(round(100*mm[idx[0]]/Mr,2), round(100*rr[idx[0]]/rmax,2)))

    if simple_odeint==True:
        print("\n\nSimple intégration avec les conditions intiales prédéfinies :")
        print("Point de rencontre à {}% de Mr".format(Pct))
        print('x P0_S :',po_fact)
        print('x T0_S :',to_fact)
        print('x R_S :',rm_fact)
        print('x L_S :',lm_fact)


if Multi_fit==True: # création du diagramme H-R et d'autres graphs
    if H_R==True:
        print("\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#")
        print('\n Réalisation du diagramme Hertzsprung Russell :\n')
        Mri=np.around(np.geomspace(0.36,30,70),2)

        if some_stars==True:
            Mri=np.array([0.77, 0.85, 1, 1.3, 1.75, 4.15, 12]) # Soleil et 6 autres étoiles réelles

    if Rho_c_M==True:
        print("\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#")
        print('\n Réalisation du graphique de densité centrale :\n')
        Mri=np.arange(0.2,3.5,0.1)

    if visu_reg==True:
        print("\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#")
        print('\n Réalisation du graphique des zones radiatives et convectives :\n')
        Mri=np.around(np.geomspace(0.16,30,50),2)

    if ecart_yi==True:
        print("\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#")
        print('\n Réalisation du graphique des modifications dans les conditions intiales :\n')
        M1=np.arange(0.2,3.1,0.175) ; M2=np.linspace(3.5,21,1); M3=np.arange(22,31,2)
        Mri=np.concatenate([M1,M2,M3])

    nb_calc=0 ; calc=[H_R,Rho_c_M,visu_reg,ecart_yi]
    for j in calc:
        if j==True:
            nb_calc+=1
    if nb_calc>1: # au cas où plusieurs options auraient été activées
        Mri=np.around(np.geomspace(0.16,30,50),2)

    print("\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#")
    print('\n Masses solaires :',Mri,'\n')

    eps=5e-3
    Teff=[] ; Lmax=[]
    Rho_c=[]
    Ch_reg1=[] ; Ch_reg2=[] ; Ch_reg3=[] ; Ch_reg4=[]
    M_R50=[] ; M_L90=[]
    Ecart_yi=[] ; Nb_i=[] ; max_ecart=[]

    for i in range(len(Mri)): # changement de masse
        Mr=M_S*Mri[i]
        print('\nCalculs avec {} x M_S'.format(round(Mri[i],2)))

        if any([H_R, Rho_c_M, visu_reg]):
            [m,m2], [r,r2,P,P2,Lr,Lr2,T,T2], yi , y0, y2, Y0, nb_i = res_Fit(Mr,eps,Pct,precis_up)

            if H_R==True:
                Teff.append(T2[0]) # température de surface
                Lmax.append(Lr2[0]/L_S) # luminosité de l'étoile

            if Rho_c_M==True:
                Rho_c.append(rho(P[0],T[0])) # densité centrale

            if visu_reg==True:
                mm=np.concatenate((m,m2[::-1]))
                rr=np.concatenate((r,r2[::-1])) ; pp=np.concatenate((P,P2[::-1]))
                ll=np.concatenate((Lr,Lr2[::-1])) ; tt=np.concatenate((T,T2[::-1]))

                if opak==True:
                    Kappa_i=visu_kappa(pp,tt)
                if opak==False:
                    Kappa_i=kappa_es

                d_rad=Delta_rad(mm,pp,ll,tt,Kappa_i) ; d_ad=Delta_ad()

                idx = np.argwhere(np.diff(np.sign(d_rad - d_ad)) != 0).reshape(-1) + 0 # indice de différence nulle entre d_rad et d_ad

                if len(idx)==0:
                    Ch_reg1.append(0) ; Ch_reg2.append(1)
                    Ch_reg3.append(0) ; Ch_reg4.append(1)
                else :
                    if d_rad[0]>d_ad:
                        if d_rad[-2]<d_ad:
                            Ch_reg1.append(mm[idx[0]]/Mr) ; Ch_reg2.append(1)
                            Ch_reg3.append(rr[idx[0]]/rr[-1]) ; Ch_reg4.append(1)
                        else:
                            if d_rad[-2]>d_ad and d_rad[int(len(d_rad)/2)]<d_ad:
                                Ch_reg1.append(mm[idx[0]]/Mr) ; Ch_reg2.append(mm[idx[1]]/Mr)
                                Ch_reg3.append(rr[idx[0]]/rr[-1]) ; Ch_reg4.append(rr[idx[1]]/rr[-1])
                            else:
                                Ch_reg1.append(0) ; Ch_reg2.append(0)
                                Ch_reg3.append(0) ; Ch_reg4.append(0)

                    if d_rad[0]<d_ad:
                        Ch_reg1.append(0) ; Ch_reg2.append(mm[idx[0]]/Mr)
                        Ch_reg3.append(0) ; Ch_reg4.append(rr[idx[0]]/rr[-1])

                idx_R50=(np.abs(rr-rr[-1]*0.5)).argmin()
                M_R50.append(mm[idx_R50]/Mr) # la masse dans la moitié du rayon de l'étoile

                idx_L90=(np.abs(ll-ll[-1]*0.90)).argmin()
                M_L90.append(mm[idx_L90]/Mr) # la masse qui est responsable de 90% de la production de la puissance totale

        if ecart_yi==True:
            [m,m2], [r,r2,P,P2,Lr,Lr2,T,T2], yi , y0, y2, Y0, nb_i = res_Fit(Mr,eps,Pct,False,6) # args=6 pour que toutes les intégrations se fassent à la même échelle de pas ( pour voir Nb_i(Mr) )
            Pc, Tc, Rm, Lm = boundary_cond(Mr,True)
            P0=yi[0] ; T0=yi[1] ; rmax=yi[2] ; L_rmax=yi[3]

            Emax=np.zeros(4)
            for j in range(4):
                Emax[j]=(max(abs(Y0[:,j]-1))+1)

            max_ecart.append(Emax)
            Ecart_yi.append([P0/Pc, T0/Tc, rmax/Rm, L_rmax/Lm])
            Nb_i.append(nb_i[-1])


if Time==True: # temps pour exécuter le programme
    if animated==False and (Fit_method==True or Multi_fit==True):
        print('\n\n {} secondes pour exécuter le programme.\n'.format(time.time()-tmp1))


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

###### Graphiques

def Plots4(m,r,p,l,t,mm,yi, Legend, *pct_view): # structure interne
    ri=yi[2] ; pi=yi[0] ; li=yi[3] ; ti=yi[1]
    plt.plot(m/mm,r/ri,'b-', label='r(m)')
    plt.plot(m/mm,p/pi,'k-', label='P(m)')
    plt.plot(m/mm,l/li,'g-', label='L(m)')
    plt.plot(m/mm,t/ti,'r-', label='T(m)')

    plt.xlabel("M/$M_{\odot}$")
    plt.ylabel("R/$R_{\odot}$, P/$P0_{\odot}$, L/$L_{\odot}$, T/$T0_{\odot}$")

    if Legend==True:
        #plt.legend(bbox_to_anchor=(1, 0.7))
        plt.legend(loc=7, framealpha=0.9)

    if pct_view: # pour voir le point de rencontre
        plt.plot(m[int(len(m)*Pct)]/mm,r[int(len(m)*Pct)]/ri,'xk')
        plt.plot(m[int(len(m)*Pct)]/mm,p[int(len(m)*Pct)]/pi,'xk')
        plt.plot(m[int(len(m)*Pct)]/mm,l[int(len(m)*Pct)]/li,'xk')
        plt.plot(m[int(len(m)*Pct)]/mm,t[int(len(m)*Pct)]/ti,'xk')

    plt.title('Structure stellaire pour une étoile de {} masse solaire'.format(Mmulti))
    plt.show()

def multiplot_sep_var(yi): # structure interne au cours de la fitting method
    warnings.filterwarnings("ignore") # supprime le message 'subplot deprecation warning'
    color=iter(cm.rainbow(np.linspace(0,1,len(yi))))
    for i in range(len(yi)):
        p0=yi[i][0]*P0 ; t0=yi[i][1]*T0
        r2=yi[i][2]*rmax ; l2=yi[i][3]*L_rmax
        m0,r0,Lr0,P_rmax,T_rmax = init_cond(Mr,p0,t0,r2,l2)

        y0=[r0,p0,Lr0,t0] ; y2=[r2, P_rmax, l2, T_rmax]
        m=np.linspace(m0,Mr*Pct,N1) ; m2=np.linspace(Mr,Mr*Pct,N1)
        with stdout_redirected():
            x=odeint(DD,y0,m) ; x2=odeint(DD,y2,m2)

        R=x[:,0] ; P=x[:,1] ; Lr=x[:,2] ; T=x[:,3]
        R2=x2[:,0] ; P2=x2[:,1] ; Lr2=x2[:,2] ; T2=x2[:,3]

        col=next(color)
        plt.suptitle('Evolution des variables avec la fitting method')

        ax1=plt.subplot(221)
        ax1.set_title('Luminosité (W)')
        ax1.plot(m/Mr,Lr, c=col) ; ax1.plot(m2/Mr,Lr2, c=col)

        ax2=plt.subplot(222)
        ax2.set_title('Température (K)')
        ax2.plot(m/Mr,T, c=col) ; ax2.plot(m2/Mr,T2, c=col)

        ax3=plt.subplot(223)
        ax3.set_title('Pression (Pa)')
        ax3.plot(m/Mr,P, c=col) ; ax3.plot(m2/Mr,P2, c=col)
        ax3.set_xlabel("M/$M_{\odot}$")

        ax4=plt.subplot(224)
        ax4.set_title('Rayon (m)')
        ax4.plot(m/Mr,R, c=col) ; ax4.plot(m2/Mr,R2, c=col)
        ax4.set_xlabel("M/$M_{\odot}$")
    plt.show()

def visu_H_R(Teff,Lmax,Mri): # diagramme H-R
    M_abs = -2.5*np.log10(np.array(Lmax)*L_S)+71.2 # Magnitude absolue
    B_V = (4600/np.array(Teff)-0.19)/0.92 # indice de couleur (B-V)

    dat=np.loadtxt('starsdata.txt') # données du diagramme
    m=dat[:,1] ; Bv_dat=dat[:,2] ; d=dat[:,3] ; dd=dat[:,4]
    M_dat = m + 5*(np.log10(d) + 1)

    fig, ax1 = plt.subplots()
    ax2=ax1.twinx() ; ax3=ax1.twiny()
    ax1.plot(Bv_dat,M_dat,'xk',markersize=1) # données

    if some_stars==True: # H-R seulement avec 7 étoiles connues
        bv_stars=[1.05, 0.88, 0.66, 0.44, 0.17, -0.11, -0.22]
        ma_stars=[6.89, 6.19, 4.48, 2.93, 2.42, -0.57, -5.42]
        nm_stars=['Epsilon Indi','Epsilon Eridani','Soleil','Eta Arietis','Beta Pictoris','Alpha Leonis','Beta Centauri']
        color=iter(cm.rainbow(np.linspace(0,1,len(Mri))))
        for i in range(len(Mri)):
            cols=next(color)
            ax1.plot(bv_stars[i],ma_stars[i],'o',c=cols, label=nm_stars[i])
            ax1.plot(B_V[i],M_abs[i],'x',c=cols, markersize=15) # calculées
        ax1.plot(B_V[-1],M_abs[-1],'xr', label='étoiles calculées')
    else:
        ax1.plot(B_V,M_abs,'sc') # calculées

    ax1.set_ylim(20,-10) ; ax1.set_xlim(-0.5,2.5)
    ax1.set_xlabel('Indice de couleur (B-V)');ax1.set_ylabel('Magnitude absolue')

    ax2.plot(B_V,Lmax,'xk',markersize=0.01) # axe luminosité
    ax2.set_ylabel('Luminosité (L/$L_{\odot}$)')
    ax2.set_ylim(1e-6,6e5) ; ax2.set_yscale('log')

    ax3.plot(Teff,M_abs,'xk',markersize=0.01) # axe température
    ax3.set_xlabel('Teff (K)') ; ax3.set_xlim(30000,500) ; ax3.set_xscale('log')

    ax1.legend(loc=1)
    plt.show()

def visu_grad_T(m,r,p,l,t,m2,r2,p2,l2,t2): # gradient de température
    mm=np.concatenate((m,m2[::-1]))
    rr=np.concatenate((r,r2[::-1])) ; pp=np.concatenate((p,p2[::-1]))
    ll=np.concatenate((l,l2[::-1])) ; tt=np.concatenate((t,t2[::-1]))

    Kappa_i=visu_kappa(pp,tt)
    d_rad=Delta_rad(mm,pp,ll,tt,Kappa_i) ; d_ad=Delta_ad()

    plt.plot(np.linspace(0,mm[-1]/M_S,2),np.ones(2)*d_ad,label=r"$\nabla_{ad}$, $\gamma=\frac{5}{3}$")
    plt.plot(mm[0:-1]/M_S,d_rad[0:-1],'r',label=r"$\nabla_{rad}$")

    idx = np.argwhere(np.diff(np.sign(d_rad - d_ad)) != 0).reshape(-1) + 0 # indice de différence nulle entre d_rad et d_ad

    if len(idx)==0:
        None
    else :
        if d_rad[0]>d_ad:
            if d_rad[-2]<d_ad:
                plt.axvspan(0,mm[idx[0]]/M_S,alpha=0.2,color='grey')
                plt.axvline(mm[idx[0]]/M_S,c='black',linestyle='--')
            else:
                plt.yscale('log')
        if d_rad[0]<d_ad:
            plt.axvspan(mm[idx[0]]/M_S,Mr/M_S,alpha=0.2,color='grey')
            plt.axvline(mm[idx[0]]/M_S,c='black',linestyle='--')
            plt.yscale('log')

    plt.title(r'stable si $\nabla_{rad} < \nabla_{ad}$, convection sinon')
    plt.xlabel("m/$M_{\odot}$") ; plt.xlim(0,Mmulti)
    plt.ylabel('Gradient de température')
    plt.legend();plt.show()

def pp_cno(): # plot chaîne PP / cycle CNO
    T = np.linspace(7.5e6, 4.e7, 100)
    plt.plot(T, q_pp(X, Z, T), label='Chaîne PP')
    plt.plot(T, q_cno(X, Z, T), label='Cycle CNO')
    plt.axvline(T0_S,c='black',linestyle='--',label='Soleil')
    plt.title("Taux de production d'énergie des réactions nucléaires")
    plt.xlabel("Température centrale (K)")
    plt.ylabel("Taux de production d'énergie (W/kg)") ; plt.yscale('log')
    plt.legend() ; plt.show()

##~~~~~~

if simple_odeint==True or Fit_method==True: # structure plot
    if animated==False:
        mm=np.concatenate((m,m2[::-1]))
        rr=np.concatenate((r,r2[::-1]))
        pp=np.concatenate((P,P2[::-1]))
        ll=np.concatenate((Lr,Lr2[::-1]))
        tt=np.concatenate((T,T2[::-1]))
        Plots4(mm,rr,pp,ll,tt, Mr,yi, False)

##~~~~~~

if Fit_method==True:
    if evo_yi_i==True: # évolution des conditions initiales avec la fitting method
        Y0=np.array(Y0)
        leg=['Pc','Tc','R','L']
        for k in range(4): #P,T,R,L
            plt.plot(nb_i,Y0[:,k],'-', label=leg[k])
        plt.xlabel('itérations') ; plt.ylabel('Conditions initiales normalisées');
        plt.title('évolution des paramètres init')
        plt.legend() ; plt.show()

    if mutliP_sep_var==True: # évolution des variables jusqu'à la convergence
        multiplot_sep_var(Y0)

    if visu_opacity==True: # opacité
        plt.plot(m/Mr,visu_kappa(P,T),'k-')
        plt.plot(m2/Mr,visu_kappa(P2,T2),'k-')
        plt.xlabel("M/$M_{\odot}$");plt.ylabel('Kappa') ; plt.yscale('log')
        plt.title('Opacité')
        plt.show()

    if grad_T==True: # gradient de température
        visu_grad_T(m,r,P,Lr,T,m2,r2,P2,Lr2,T2)

##~~~~~~

if Multi_fit==True:
    if H_R==True: # diagramme H-R
        visu_H_R(Teff,Lmax,Mri)

    if Rho_c_M==True:  # densité centrale (Mr)
        maxRhoc=max(Rho_c)
        plt.plot(Mri,Rho_c/maxRhoc,'-r',label='densité centrale')
        plt.axvline(Mri[Rho_c.index(maxRhoc)],c='black',linestyle='--')
        if visu_reg==False:
            plt.title('Densité centrale')
            plt.xlabel("M/$M_{\odot}$") ; plt.ylabel(r"$\rho_c\,/\,max(\rho_c)$")
            plt.show()

    if visu_reg==True: # zones radiatives / convectives
        plt.plot(Mri,Ch_reg1,'-k',linewidth=0.5) # regions fct de M
        plt.plot(Mri,Ch_reg2,'-k',linewidth=0.5)
        plt.fill_between(Mri,0,Ch_reg1,alpha=0.3,color='grey',label='zone convective')
        plt.fill_between(Mri,1,Ch_reg2,alpha=0.3,color='grey')

        plt.plot(Mri,M_R50,'b',label='m(50%R)') # M(50%R)
        plt.plot(Mri,M_L90,'g',label='m(90%L)') # M(90%L)

        if Rho_c_M==False:
            plt.ylabel('m/M')
        else:
            plt.ylabel(r"m/M (noir, bleu, orange) et $\rho\,/\,max(\rho_c)$ (vert)")
        plt.xlabel('M / $M_{\odot}$ (log)')
        plt.xlim(Mri[0],Mri[-1]) ; plt.ylim(0,1)
        plt.xscale('log') ; plt.legend(loc=7)
        plt.show()

        plt.plot(Mri,Ch_reg3,'-k',linewidth=0.5) # regions fct de R
        plt.plot(Mri,Ch_reg4,'-k',linewidth=0.5)
        plt.fill_between(Mri,0,Ch_reg3,alpha=0.3,color='grey',label='zone convective')
        plt.fill_between(Mri,1,Ch_reg4,alpha=0.3,color='grey')
        plt.ylabel('r/Rm')
        plt.xlabel('M / $M_{\odot}$ (log)')
        plt.xlim(Mri[0],Mri[-1]) ; plt.ylim(0,1)
        plt.xscale('log') ; plt.legend(loc=7)
        plt.show()

    if ecart_yi==True: # modifications dans les conditions initiales en fonction de Mr
        ax1=plt.subplot(311)
        ax1.plot(Mri,np.array(Nb_i)/6,label="nombre d'itération pour converger")# Nb_i * 6
        ax1.set_xscale('log') ; ax1.set_xlim(Mri[0],Mri[-1]) ; ax1.legend()

        ax2=plt.subplot(312)
        leg=['P0/Pc', 'T0/Tc', 'rmax/Rm', 'L_rmax/Lm']
        for i in range(4):
            ax2.plot(Mri,np.array(Ecart_yi)[:,i], label=leg[i]) # Ecart_yi
        ax2.plot(Mri,np.ones(len(Mri)),'--k')
        ax2.set_yscale('log')
        ax2.set_xscale('log') ; ax2.set_xlim(Mri[0],Mri[-1]) ; ax2.legend()

        ax3=plt.subplot(313)
        for i in range(4):
            ax3.plot(Mri,np.array(max_ecart)[:,i],label=leg[i]) # max ecart(nb_i)_yi
        ax3.set_yscale('log')
        ax3.set_xscale('log') ; ax3.set_xlim(Mri[0],Mri[-1])
        ax3.set_xlabel("M/$M_{\odot} (log)$"); ax3.legend()
        plt.show()

##~~~~~~

if multi_pct==True: # avec différents points de rencontre
    print("\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#")
    print("\n Changement du point de rencontre pour une étoile de 1 masse solaire.")
    Mr=M_S
    eps=2e-3
    pct_i=np.linspace(0.25,0.75,3)
    print("\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#")
    print('Points de rencontre utilisés (% de Mr):',pct_i)
    for i in pct_i:
        pct=i
        print('\nCalculs avec le point de rencontre à ',pct)

        [m,m2], [r,r2,P,P2,Lr,Lr2,T,T2], yi , y0, y2, Y0, nb_i = res_Fit(Mr,eps,pct,False,6)

        Pc, Tc, Rm, Lm = boundary_cond(Mr,6)
        mm=np.concatenate((m,m2[::-1]))
        rr=np.concatenate((r,r2[::-1])) ; pp=np.concatenate((P,P2[::-1]))
        ll=np.concatenate((Lr,Lr2[::-1])) ; tt=np.concatenate((T,T2[::-1]))

        print('Modifications dans les conditions initiales pour avoir le modèle:')
        print('x Pc :',P[0]/Pc, '\nx Tc :',T[0]/Tc)
        print('x Rm :',r2[0]/Rm, '\nx Lm :',Lr2[0]/Lm,'\n')

        Plots4(mm,rr,pp,ll,tt, Mr,yi, False, True)

##~~~~~~

if change_compos==True: # changements dans la composition de l'étoile
    print("\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#")
    print("\nChangements dans la composition d'une étoile de 1 masse solaire.")
    Mr=M_S
    Xi=[0.5,0.55,0.7]
    Zi=[0.001,0.03,0.29]
    print("\n#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#")
    choice=str(input("Changements majeurs dans (X ou Z): "))
    if choice=='X':
        n=len(Xi)
    elif choice=='Z':
        n=len(Zi)
    else:
        print("Choisir entre X et Z pour obtenir un résultat.")
        n=0

    color=iter(cm.rainbow(np.linspace(0,1,n)))
    eps=2e-3

    for i in range(n):
        if choice=='X':
            X=Xi[i]
            Y=1-X-Z
            mu=1/(2*X+0.75*Y+0.5*Z) ; kappa_es = 0.02*(1+X)
            labl='X ='+str(X)
            print('Calculs avec X={}, Y={}, Z={}'.format(round(X,2),round(Y,2),Z))

        if choice=='Z':
            Z=Zi[i]
            Y=1-X-Z
            mu=1/(2*X+0.75*Y+0.5*Z) ; kappa_es = 0.02*(1+X)
            labl='Z ='+str(Z)
            print('Calculs avec X={}, Y={}, Z={}'.format(round(X,2),round(Y,3),Z))

        [m,m2], [r,r2,P,P2,Lr,Lr2,T,T2], yi , y0, y2, Y0, nb_i = res_Fit(Mr,eps,Pct,False)

        mm=np.concatenate((m,m2[::-1]))/m2[0]
        rr=np.concatenate((r,r2[::-1]))/r2[0] ; pp=np.concatenate((P,P2[::-1]))/P[0]
        ll=np.concatenate((Lr,Lr2[::-1]))/Lr2[0] ; tt=np.concatenate((T,T2[::-1]))/T[0]

        col=next(color)
        plt.plot(mm,rr,c=col) ; plt.plot(mm,pp,c=col)
        plt.plot(mm,ll,c=col) ; plt.plot(mm,tt,c=col,label=labl)

        idx_L90=(np.abs(ll-ll[-1]*0.90)).argmin()
        plt.plot(mm[idx_L90],ll[idx_L90],'xk')
    plt.xlabel("M/$M_{\odot}$")
    plt.ylabel("R/$R_{\odot}$, P/$P0_{\odot}$, L/$L_{\odot}$, T/$T0_{\odot}$")
    plt.legend(loc=7) ; plt.show()

##~~~~~~

if PP_CNO==True: # production d'énergie
    pp_cno() # plot
    print("\nSensibilité à la température autour de T = T0_S :")
    print('PP-Chain sensibilité: ', T_sensitivity(q_pp, 1.5e7, X,Z))
    print('CNO-Cycle sensibilité: ', T_sensitivity(q_cno, 1.5e7, X,Z))
