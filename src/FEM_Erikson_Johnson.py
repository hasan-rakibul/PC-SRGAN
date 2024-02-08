# This source file is mostly developed by Pouria Behnoudfar

from fenics import *
from dolfin import *
import numpy as np
import time
import pandas as pd


Degree = 3
x1 = -1
x2 = 0.
y1 = -0.5
y2 = 0.5
def update_initial_v(u_0, v_n, s, eps, K, beta_vec, f_3):
    F_0 = ( v_n*s*dx + dot(beta_vec,grad(u_0))*s*dx
    + eps*dot(grad(u_0), grad(s))*dx
    + K*u_0*(u_0-1)*s*dx    # the term in the third line is the reaction nonlinear function
    - f_3*s*dx )              # Sourcing term

    solve(F_0 == 0, v_n)

    return


def boundary(x, on_boundary):
    tol = 1E-14
    return on_boundary


def generate_data(eps, K, r, theta, elem_per_dim, save_as):

    b1 = r * np.cos(theta) # Velocity component in x direction
    b2 = r * np.sin(theta) # Velocity component in y direction

    ###### Problem's Parameters
    #
    T =.75 # final time

    lam1=(-1+sqrt(1-4*2*eps))/(-2*eps)
    lam2=(-1-sqrt(1-4*2*eps))/(-2*eps)
    #print( "lambda1 %s  "%(lam1))
    #print( "lambda2  %s  "%lam2)

    #r1 and s1
    s_one=(1-sqrt(1+4*pi*pi*eps*eps))/(2*eps)
    r_one=(1+sqrt(1+4*pi*pi*eps*eps))/(2*eps)
    #######-------------------------------------------------------


        #######-------------------------------------------------------
    n_ele = elem_per_dim   # no. of elements
    num_steps = 150    # number of time steps
    dt = T / num_steps # time step size
    # Define expressions used in variational forms
    dt = Constant(dt)
    K = Constant(K)
    eps = Constant(eps)
    beta_vec = as_vector([Constant(b1),Constant(b2)])

    # ---- G-alpha parameters

    ro = 0

    am = 1 / 2 * (3-ro) / (1+ro)
    af = 1 / (1+ro)
    gamma = 0.5 - af + am
    ####



    mesh  = RectangleMesh(Point(x1, y1), Point(x2, y2), n_ele,n_ele)


    # Define function space
    P_order = 2     # Polynomial order
    Fun_sp = FunctionSpace(mesh, "CG", P_order)

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


    u0_ref =  Expression('exp(-2*0)*(exp(lam1*x[0])-exp(lam2*x[0])) +cos(pi*x[1])*((exp(s_one*x[0])-exp(r_one*x[0]))/(exp(-s_one)-exp(-r_one)))',time=0,lam1=lam1, lam2 = lam2 ,s_one = s_one, r_one=r_one,degree = Degree-1,domain=mesh)

    u_n = interpolate(u0_ref, Fun_sp)
    update_initial_v(u_0=u_n, v_n=v_n, s=s, eps=eps, K=K, beta_vec=beta_vec, f_3=f_3) # updates v_n

    # Time-stepping
    t = 0

    save_dir = 'data/reaction_diffusion_advection/mesh_' + str(elem_per_dim) + '/' + save_as + '/'

    # Create VTK files for visualisation
    vtkfile_u_n = File(save_dir + 'u_n.pvd')

    for i_step in range(num_steps):

        t += dt

        F = (am/(dt*gamma*af)*u*s*dx + dot(beta_vec,grad(u))*s*dx
        + eps*dot(grad(u), grad(s))*dx
        + K*u*(u-1)*s*dx - f_3*s*dx
        - am/(dt*gamma*af)*u_n*s*dx - (am/gamma-1) *v_n*s*dx)

        time =i_step*dt
        u_D =Expression('exp(-2*time)*(exp(lam1*x[0])-exp(lam2*x[0])) +cos(pi*x[1])*((exp(s_one*x[0])-exp(r_one*x[0]))/(exp(-s_one)-exp(-r_one)))',time=time,lam1=lam1, lam2 = lam2 ,s_one = s_one, r_one=r_one,degree = Degree-1,domain = mesh)

        bc_D = DirichletBC(Fun_sp, u_D, boundary)
        # Solve variational problem for time step
        solve(F == 0, u, bc_D)    # here we solve for u^{n+\alpha_f}
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

    eps_range = np.linspace(1e-2, 0.11, 10) # diffusion coefficient
    K_range = np.linspace(0,2 , 5) # reaction rate

    r_range = np.linspace(1e-1, 1, 5)
    theta_range = np.linspace(-np.pi/6, np.pi/6, 5)





## Random initial condition
    mesh_ref  = RectangleMesh(Point(x1, y1), Point(x2, y2), 100,100)

    # Define function space
    P_order = 2     # Polynomial order
    Fun_sp_ref = FunctionSpace(mesh_ref, "CG", P_order)

    u_0_ref = Function(Fun_sp_ref)
    Dofs = Fun_sp_ref.dofmap().dofs()
    Ndofs= np.size(Dofs)


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

                    generate_data(eps, K, r, theta, 7, save_as) # generating data for 7x7 mesh
                    generate_data(eps, K, r, theta, 63, save_as) # generating data for 63x63 mesh

                    # status update to a log file
                    count += 1
                    with open('FEM_log.txt', 'a') as f:
                        f.write(f"eps: {eps}, K: {K}, r: {r}, theta: {theta}\n")
                        f.write(f"Progress: {count}/{total}\n")
                        f.write(f"Time elapsed: {time.time() - start_time}\n")
                        f.write("--------------------------------------------------\n")

    # save index_value pairs
    index_val.to_csv('data/Erikson_Johnson/index-val-mapping.csv')

if __name__ == "__main__":
    main()
