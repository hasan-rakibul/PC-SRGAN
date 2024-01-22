# This source file is primarily taken from Pouria Behnoudfar and later modified by Rakibul Hasan

from fenics import *
from dolfin import *
import numpy as np
import time
import pandas as pd



x1 = -0.5
x2 = 0.5
y1 = -0.5
y2 = 0.5
Theta_c = 1.2

opt = 1
M_par = 1
Eps =  (1/64)/(2*(2**0.5)*np.arctanh(0.9))

def Fpar(phi,Theta_c,Theta_,opt):
    if opt == 1:
        F  = 0.5*Theta_*((Constant(1)+phi)*ln(Constant(1)+phi)+(Constant(1)-phi)*ln(Constant(1)-phi))-0.5*Theta_c*phi**2
    return F

def dFpar(phi,Theta_c,Theta_,opt):
    if opt == 1:
        # dF = 0.5*Theta_*(Constant(2)+ln(Constant(1)+phi)-ln(Constant(1)-phi))-Theta_c*phi
        dF = diff(Fpar(phi,Theta_c,Theta_,opt),phi)
    return dF

def update_initial_v(u_0, v_n, s, eps, K, Theta_, f_3):
    F_0 = ( v_n*s*dx
    + eps*dot(grad(u_0), grad(s))*dx
    + Constant(K*M_par/Eps**2)*inner(dFpar(u_0,Theta_c,Theta_,opt),s)*dx    # the term in the third line is the reaction nonlinear function
     )              # Sourcing term

    solve(F_0 == 0, v_n)

    return

def generate_data(eps, K, r, Theta_, elem_per_dim, save_as,u0_ref):


    ###### Problem's Parameters
    #
    T =.001 # final time
    # Initial condition



        #######-------------------------------------------------------
    n_ele = elem_per_dim   # no. of elements
    num_steps = 100    # number of time steps
    dt = T / num_steps # time step size
    # Define expressions used in variational forms
    dt = Constant(dt)
    K = Constant(K)
    eps = Constant(eps)


    # ---- G-alpha parameters

    ro = 0

    am = 1 / 2 * (3-ro) / (1+ro)
    af = 1 / (1+ro)
    gamma = 0.5 - af + am
    ####



    mesh  = UnitSquareMesh(n_ele,n_ele)#RectangleMesh(Point(x1, y1), Point(x2, y2), n_ele,n_ele)


    # Define function space
    P_order = 2     # Polynomial order
    Fun_sp = FunctionSpace(mesh, "CG", P_order)#, constrained_domain=pbc)

    # Define test functions
    s = TestFunction(Fun_sp)

    # Define functions for velocity and concentration

    u = Function(Fun_sp)
    u_n = Function(Fun_sp)
    u_nn = Function(Fun_sp)
    v_n = Function(Fun_sp)
    v_nn = Function(Fun_sp)

    f_3 = Constant(0)

    # Boundary data

    tol = 1e-14



    u_n = interpolate(u0_ref, Fun_sp)
    update_initial_v(u_0=u_n, v_n=v_n, s=s, eps=eps, K=K, Theta_=Theta_, f_3=f_3) # updates v_n

    # Time-stepping
    t = 0

    save_dir = 'data/reaction_diffusion_advection/mesh_' + str(elem_per_dim) + '/' + save_as + '/'

    # Create VTK files for visualisation
    vtkfile_u_n = File(save_dir + 'u_n.pvd')

    for i_step in range(num_steps):

        t += dt

        F = (am/(dt*gamma*af)*u*s*dx
        + eps*dot(grad(u), grad(s))*dx
        + Constant(K*M_par/Eps**2)*inner(dFpar(u,Theta_c,Theta_,opt),s)*dx
        - am/(dt*gamma*af)*u_n*s*dx - (am/gamma-1) *v_n*s*dx)

        # Solve variational problem for time step
        solve(F == 0, u)


                                 # here we solve for u^{n+\alpha_f}
        vtkfile_u_n << (u_n, t)

        # Update previous solution
        u_nn.vector()[:]=(u.vector()[:]-u_n.vector()[:])/af+u_n.vector()[:]
        v_nn.vector()[:]= (u_nn.vector()[:]-u_n.vector()[:]-dt*v_n.vector()[:])/(dt*gamma)+v_n.vector()[:]
        u_n = u_nn
        v_n = v_nn





def main():
    start_time = time.time()

    # only to save index_value pairs
    index_val = pd.DataFrame(columns=['eps', 'K', 'r', 'theta'])

    ###### Problem's Parameters

    eps_range = np.linspace(1e-0, 10, 4) # diffusion coefficient
    K_range = np.linspace(1, 4, 4) # reaction rate


    r_range = np.linspace(0, 0, 1)
    theta_range = np.linspace(1, 1.1, 2)

## Random initial condition
    mesh_ref  = UnitSquareMesh(100,100)#RectangleMesh(Point(x1, y1), Point(x2, y2), 100,100)

    # Define function space
    P_order = 2     # Polynomial order
    Fun_sp_ref = FunctionSpace(mesh_ref, "CG", P_order)

    u_0_ref = Function(Fun_sp_ref)
    Dofs = Fun_sp_ref.dofmap().dofs()
    Ndofs= np.size(Dofs)
    u_0_ref.vector()[:] = 0.01*((0.5-np.random.rand( Ndofs)))



## -----------------------------

    count = 0
    total = len(eps_range) * len(K_range) * len(r_range) * len(theta_range)
    for i_eps, eps in enumerate(eps_range):
        for i_K, K in enumerate(K_range):
            for i_r, r in enumerate(r_range):
                for i_theta, theta in enumerate(theta_range):
                    save_as = 'eps' + str(i_eps) + '_K' + str(i_K) + '_r' + str(i_r) + '_theta' + str(i_theta)

                    # storing index_value pairs
                    index_val.loc[save_as] = [eps, K, r, theta]
                    Theta_ = theta
                    generate_data(eps, K, r, Theta_, 7, save_as,u_0_ref) # generating data for 7x7 mesh
                    generate_data(eps, K, r, Theta_, 63, save_as,u_0_ref) # generating data for 63x63 mesh

                    # status update to a log file
                    count += 1
                    with open('FEM_log.txt', 'a') as f:
                        f.write(f"eps: {eps}, K: {K}, r: {r}, theta: {theta}\n")
                        f.write(f"Progress: {count}/{total}\n")
                        f.write(f"Time elapsed: {time.time() - start_time}\n")
                        f.write("--------------------------------------------------\n")

    # save index_value pairs
    index_val.to_csv('data/reaction_diffusion_advection/index-val-mapping.csv')

if __name__ == "__main__":
    main()
