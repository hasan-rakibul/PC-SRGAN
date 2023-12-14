# This source file is primarily taken from Pouria Behnoudfar and later modified by Rakibul Hasan

from fenics import *
from dolfin import *
import numpy as np
import time

def update_initial_v(u_0, v_n, bcs_0, s, eps, K, beta_vec, f_3):
    F_0 = ( v_n*s*dx + dot(beta_vec,grad(u_0))*s*dx
    + eps*dot(grad(u_0), grad(s))*dx
    - K*u_0*u_0*(1-u_0)*s*dx    # the term in the third line is the reaction nonlinear function
    - f_3*s*dx )              # Sourcing term
    
    solve(F_0 == 0, v_n, bcs_0)
   
    return 

def generate_data(eps, K, r, theta, elem_per_dim, save_as):

    b1 = r * np.cos(theta) # Velocity component in x direction
    b2 = r * np.sin(theta) # Velocity component in y direction

    ###### Problem's Parameters
    # 
    T = 0.1 # final time

    # BCs
    u_L = 1
    u_R = 0
    u_U = 0
    u_B = 0

    # Initial condition

    # u_0 = Expression('sin(x[0])*sin(x[1])', degree=2)
    u_0 = Constant(0) # Expression('exp(x[0])*sin(x[1])', degree=2)

    #######-------------------------------------------------------
    n_ele = elem_per_dim   # no. of elements
    num_steps = 100    # number of time steps
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

    x1 = 0
    x2 = 1
    y1 = 0
    y2 = 1
    # rect = Rectangle(Point(x1, y1), Point(x2, y2))
    # mesh  = generate_mesh(rect1, n_ele)

    mesh = UnitSquareMesh(n_ele, n_ele)

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

    def boundary_right(x, on_boundary):
        return on_boundary and near(x[0], x2, tol)

    def boundary_left(x, on_boundary):
        return on_boundary and near(x[0], x1, tol)

    def boundary_bottom(x, on_boundary):
        return on_boundary and near(x[1], y1, tol)

    def boundary_up(x, on_boundary):
        return on_boundary and near(x[1], y2, tol)

    bc_L = DirichletBC(Fun_sp, u_L, boundary_left)
    bc_R = DirichletBC(Fun_sp, u_R, boundary_right)
    bc_B = DirichletBC(Fun_sp, u_B, boundary_bottom)
    bc_U = DirichletBC(Fun_sp, u_U, boundary_up)

    bcs = [bc_L, bc_R, bc_U, bc_B]

    bc_L_0 = DirichletBC(Fun_sp, Constant(0), boundary_left)
    bc_R_0 = DirichletBC(Fun_sp, Constant(0), boundary_right)
    bc_B_0 = DirichletBC(Fun_sp, Constant(0), boundary_bottom)
    bc_U_0 = DirichletBC(Fun_sp, Constant(0), boundary_up)

    # Initial solution ------
    bcs_0 = [bc_L_0, bc_R_0, bc_U_0, bc_B_0]
    u_n = interpolate(u_0, Fun_sp)
    bc_L.apply(u_n.vector())
    update_initial_v(u_0=u_n, v_n=v_n, bcs_0=bcs_0, s=s, eps=eps, K=K, beta_vec=beta_vec, f_3=f_3) # updates v_n

    # Time-stepping
    t = 0

    save_dir = 'data/reaction_diffusion_advection/mesh_' + str(elem_per_dim) + '/' + save_as + '/'

    # Create VTK files for visualisation
    vtkfile_u_n = File(save_dir + 'u_n.pvd')

    for i_step in range(num_steps):

        t += dt

        F = (am/(dt*gamma*af)*u*s*dx + dot(beta_vec,grad(u))*s*dx
        + eps*dot(grad(u), grad(s))*dx
        - K*u*u*(1-u)*s*dx - f_3*s*dx
        - am/(dt*gamma*af)*u_n*s*dx - (am/gamma-1) *v_n*s*dx)

        # Solve variational problem for time step
        solve(F == 0, u, bcs)    # here we solve for u^{n+\alpha_f}
        vtkfile_u_n << (u_n, t)

        # Update previous solution
        u_nn.vector()[:]=(u.vector()[:]-u_n.vector()[:])/af+u_n.vector()[:]
        v_nn.vector()[:]= (u_nn.vector()[:]-u_n.vector()[:]-dt*v_n.vector()[:])/(dt*gamma)+v_n.vector()[:]
        u_n = u_nn
        v_n = v_nn

        # v_n.assign(solv_n)
            # print('u max: ', u.vector().array().max())
            
        # print('u max: ', np.max(np.array(u.vector()[:])))
        # print('u_n max: ', np.max(np.array(u_n.vector()[:])))
        # u_n.assign(u)

        # coords = Fun_sp.tabulate_dof_coordinates()
        # vec = u_n.vector().get_local()

        # with open(save_dir + "output_{0}.txt".format(i_step), "w") as outfile:
        #     for coord, val in zip(coords, vec):
        #         print(coord[0], coord[1], val, file=outfile)

def main():
    start_time = time.time()

    ###### Problem's Parameters

    eps_range = np.linspace(1e-1, 1, 10) # diffusion coefficient
    K_range = np.linspace(0, 1, 5) # reaction rate
    
    r_range = np.linspace(0, 1, 5)
    theta_range = np.linspace(-np.pi/4, np.pi/4, 5)

    count = 0
    total = len(eps_range) * len(K_range) * len(r_range) * len(theta_range)
    for i_eps, eps in enumerate(eps_range):
        for i_K, K in enumerate(K_range):
            for i_r, r in enumerate(r_range):
                for i_theta, theta in enumerate(theta_range):
                    save_as = 'eps' + str(i_eps) + '_K' + str(i_K) + '_r' + str(i_r) + '_theta' + str(i_theta)
                    generate_data(eps, K, r, theta, 7, save_as) # generating data for 7x7 mesh
                    generate_data(eps, K, r, theta, 63, save_as) # generating data for 63x63 mesh

                    # status update to a log file
                    count += 1
                    with open('FEM_log.txt', 'a') as f:
                        f.write(f"eps: {eps}, K: {K}, r: {r}, theta: {theta}\n")
                        f.write(f"Progress: {count}/{total}\n")
                        f.write(f"Time elapsed: {time.time() - start_time}\n")
                        f.write("--------------------------------------------------\n")


if __name__ == "__main__":
    main()