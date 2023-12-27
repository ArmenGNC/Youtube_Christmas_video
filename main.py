# Source link for mutiple widgets : https://matplotlib.org/stable/gallery/widgets/slider_snap_demo.html

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import Ellipse_shape_with_6_elements as ell
from matplotlib.widgets import Button, Slider
plt.style.use('dark_background')


def Build_text(X, Y, Z):
    # Creating a document
    f = open("variable_ell.txt", 'w')
    f.writelines("X = [")
    for i in range(len(X)):
        if i == (i - 1):
            f.writelines(str(X[i]) + "] \n")
        else:
            f.writelines(str(X[i]) + ", ")
    f.writelines("Y = [")
    for j in range(len(Y)):
        if j == (j - 1):
            f.writelines(str(Y[j]) + "] \n")
        else:
            f.writelines(str(Y[j]) + ", ")
    f.writelines("Z = [")
    for k in range(len(Z)):
        if k == (k - 1):
            f.writelines(str(Z[k]) + "] \n")
        else:
            f.writelines(str(Z[k]) + ", ")
    f.close()

    return None

def Nodes_position(X,Y,Z):

    ''' The goal of this function is to determine where are the ascending and descending node positions. '''

    # We determine the first Z we have in the list
    z_initial_sign = np.sign(Z[0])


    # We note the sign
    sign_char = ""
    if z_initial_sign < 0:
        sign_char = "negative"
    elif z_initial_sign > 0:
        sign_char = "positive"
    elif z_initial_sign == 0:
        sign_char = "positive"
    else:
        pass


    # We determine the ascending and descending nodes
    new_sign_char = ""
    list_ascending_descending_node = []

    for k in range(len(Z)-1):

        # New sign
        if np.sign(Z[k+1])<0:
            new_sign_char = "negative"
        elif np.sign(Z[k+1]) > 0:
            new_sign_char = "positive"
        elif np.sign(Z[k+1]) == 0:
            new_sign_char = "positive"

        # We compare the two signs
        if sign_char == new_sign_char:
            pass
        elif sign_char != new_sign_char:
            # We note the ascending/descending node position
            list_ascending_descending_node.append(np.array([X[k],Y[k],Z[k]]))
        else:
            pass

        # Update
        sign_char = new_sign_char

    return list_ascending_descending_node

def Perigee_Apogee_positions(X,Y,Z):

    Norm_list = []

    for i in range(len(X)):

        r = np.linalg.norm(np.array([X[i],Y[i],Z[i]]))

        Norm_list.append(r)

    # Minimum and max determination
    min_norm = min(Norm_list)
    max_norm = max(Norm_list)
    index_min_norm = 0
    index_max_norm = 0

    # Min norm index
    for i in range(len(Norm_list)):

        if Norm_list[i] == min_norm:
            index_min_norm = i
            break
        else:
            pass

    # Max norm index
    for i in range(len(Norm_list)):

        if Norm_list[i] == max_norm:
            index_max_norm = i
            break
        else:
            pass


    perigee_coord = np.array([X[index_min_norm], Y[index_min_norm], Z[index_min_norm]])
    apogee_coord = np.array([X[index_max_norm], Y[index_max_norm], Z[index_max_norm]])


    return perigee_coord, apogee_coord

def Ellipse_draw(a,e,i,L,omega,L_omega):

    # Instanciation
    inst = ell.Ellipse_drawing(a,e,i,L,omega,L_omega)

    # First we draw the ellipse
    (X,Y,Z) = inst.ellipse_XYZ()

    # We draw the ascending/descending node
    list_nodes = Nodes_position(X,Y,Z)

    # We draw the O-ωp line
    perigee_r, apogee_r = Perigee_Apogee_positions(X,Y,Z)

    # We draw the True anomaly
    vec_r_among_L , nu = inst.position_XYZ()

    print (" True anomaly is : ", nu)

    return (X,Y,Z),list_nodes, perigee_r, apogee_r, vec_r_among_L, nu

def Planes_meshgrid(angle):

    dim = 5

    # Define inclined plane.
    # angle = 0.5  # <-- This is the variable
    X2, Y2 = np.meshgrid([-dim, dim], [-dim, dim])
    Z2 = Y2 * (angle*np.pi/180)

    return X2,Y2,Z2




# Initial values
a_initial_value = 3
e_initial_value = 0.5
i_initial_value = 1
L_initial_value = 5
L_peri_initial_value = 3
L_omega_initial_value = 2

scale_value = 3
planet_size = 300

planes_visible = False
orbital_plane_visible = True

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
# Ellipse function over here
(X_ell,Y_ell,Z_ell),list_nodes_orbit, rp, ra, pos_xyz, true_anomaly = Ellipse_draw(a_initial_value,e_initial_value,i_initial_value,L_initial_value,L_peri_initial_value,L_omega_initial_value)
descending_node = list_nodes_orbit[0]
ascending_node = list_nodes_orbit[1]


ax.plot3D(X_ell,Y_ell,Z_ell,'r')
# Plot the center
ax.scatter(0, 0, 0, s=planet_size,c='b')    # center
# AN and DN
ax.plot3D([ascending_node[0]],[ascending_node[1]],[ascending_node[2]],'go')
ax.plot3D([descending_node[0]],[descending_node[1]],[descending_node[2]],'go')
# rp and ra
ax.plot3D([rp[0]],[rp[1]],[rp[2]],'ro')
ax.plot3D([ra[0]],[ra[1]],[ra[2]],'ro')
ax.plot3D([0,rp[0]],[0,rp[1]],[0,rp[2]],'r')

# True anomaly
ax.plot3D([pos_xyz[0]], [pos_xyz[1]], [pos_xyz[2]], 'wo')
ax.plot3D([0,pos_xyz[0]], [0,pos_xyz[1]], [0,pos_xyz[2]], 'w')

# Gamma line
ax.plot3D([0,10],[0,0],[0,0],'m')


# Earth and Ecliptic plane
if planes_visible == True:
    # Earth plane
    X_plane,Y_plane,Z_plane = Planes_meshgrid(23)
    ax.plot_surface(X_plane,Y_plane,Z_plane, color='blue', alpha=.4, linewidth=0, zorder=1)

    # Ecliptic plane
    X_eclip,Y_eclip,Z_eclip = Planes_meshgrid(0)
    ax.plot_surface(X_eclip,Y_eclip,Z_eclip, color='y', alpha=.4, linewidth=0, zorder=1)

    ax.text(8, 8, 0, " Ecliptic plane ")
    ax.text(8, 8, 2, " Earth plane ")

else:
    pass

# Orbital plane
X_orbital, Y_orbital, Z_orbital = Planes_meshgrid(0)
ax.plot_surface(X_orbital, Y_orbital, Z_orbital, color='b', alpha=.4, linewidth=0, zorder=1)

# Arrows for frame J2000
# X arrow
ax.plot3D([0,0.5],[0,0],[0,0],'black')
ax.plot3D([0.5,0.4],[0,0.05],[0,0],'black')
ax.plot3D([0.5,0.4],[0,-0.05],[0,0],'black')

# Y arrow
ax.plot3D([0,0],[0,0.5],[0,0],'black')
ax.plot3D([0,0.05],[0.5,0.4],[0,0],'black')
ax.plot3D([0,-0.05],[0.5,0.4],[0,0],'black')

# Z arrow
ax.plot3D([0,0],[0,0],[0,0.5],'black')
ax.plot3D([0,0],[0,0.05],[0.5,0.4],'black')
ax.plot3D([0,-0.0],[0,-0.05],[0.5,0.4],'black')



## Text annotation for ascending and descending node
ax.text(ascending_node[0],ascending_node[1],ascending_node[2]," Ascending node ")
ax.text(descending_node[0],descending_node[1],descending_node[2]," Descending node ")
ax.text(rp[0],rp[1],rp[2]," Perigee ")
ax.text(ra[0],ra[1],ra[2]," Apogee ")
ax.text(7,0,0," γ ", fontweight="bold")
ax.text(pos_xyz[0], pos_xyz[1], pos_xyz[2], " satellite ")
ax.text2D(1.07, 1.05, " Parameters : " , transform=ax.transAxes, fontsize=14,color='g',fontweight="bold")
ax.text2D(1.07,1.0," a = " + str(a_initial_value),transform=ax.transAxes, fontsize=9,color='g')
ax.text2D(1.07,0.95," e = " + str(e_initial_value) ,transform=ax.transAxes, fontsize=9,color='g')
ax.text2D(1.07,0.9," i = " + str(i_initial_value) + " °",transform=ax.transAxes, fontsize=9,color='g')
ax.text2D(1.3,1.0," ν = " + str(true_anomaly) + " °",transform=ax.transAxes, fontsize=9,color='g')
ax.text2D(1.3,0.95," Lωp = " + str(L_peri_initial_value)+ " °",transform=ax.transAxes, fontsize=9,color='g')
ax.text2D(1.3,0.9," LΩ = " + str(L_omega_initial_value)+ " °" ,transform=ax.transAxes, fontsize=9,color='g')

ax.text2D(-0.57, 1.05, " Orbital elements : ", transform=ax.transAxes, fontsize=14, color='g', fontweight="bold")
ax.text2D(-0.57, 1.0, " a : semi-major axis", transform=ax.transAxes, fontsize=9, color='g')
ax.text2D(-0.57, 0.95, " e : eccentricity ", transform=ax.transAxes, fontsize=9, color='g')
ax.text2D(-0.57, 0.9, " i : inclination ", transform=ax.transAxes, fontsize=9, color='g')
ax.text2D(-0.34, 1.0, " ν : True anomaly ", transform=ax.transAxes, fontsize=9, color='g')
ax.text2D(-0.34, 0.95, " Lωp : Longitude of the perigee  ", transform=ax.transAxes, fontsize=9, color='g')
ax.text2D(-0.34, 0.9, " LΩ : Longitude of the ascending node ", transform=ax.transAxes, fontsize=9, color='g')


# Arrows annotation
ax.text(0.5, 0, 0, " X [J2000] ",color='black')
ax.text(0, 0.5, 0, " Y [J2000] ",color='black')
ax.text(0, 0.0, 0.5, " Z [J2000] ",color='black')


ax.set_title('Keplerian elements (a,e,i,L,ωp,LΩ)',fontsize=20,fontweight="bold")

# Set axes label
ax.set_xlabel('X', labelpad=20)
ax.set_ylabel('Y', labelpad=20)
ax.set_zlabel('Z', labelpad=20)

# Vertical (a,e,i)
ax_a = fig.add_axes([0.08, 0.05,0.02, 0.65])
ax_e = fig.add_axes([0.13, 0.05,0.02, 0.65])
ax_i = fig.add_axes([0.18, 0.05,0.02, 0.65])

# Vertical (L, Lω, LΩ)
ax_L = fig.add_axes([0.75, 0.05,0.02, 0.65])
ax_Lom = fig.add_axes([0.80, 0.05,0.02, 0.65])
ax_Lgom = fig.add_axes([0.85, 0.05,0.02, 0.65])

# Axes for RadioButtons
# ax_ecl = fig.add_axes([0.3, 0.01, 0.1, 0.09])

# Set limits
ax.set_xlim(-scale_value,scale_value)
ax.set_ylim(-scale_value,scale_value)
ax.set_zlim(-scale_value,scale_value)


# define the values to use for snapping
allowed_amplitudes = np.concatenate([np.linspace(.1, 5, 100), [6, 7, 8, 9]])

# Other parameter sliders and RadioButtons
s_a = Slider(
    ax_a, "a", 0, 10,
    valinit=a_initial_value, valstep=1,
    initcolor='r',orientation='vertical',color="g"  # Remove the line marking the valinit position.
)

s_e = Slider(
    ax_e, "e", 0.01, 2,
    valinit=e_initial_value, valstep=0.01,
    initcolor='r',orientation='vertical',color="g"  # Remove the line marking the valinit position.
)

s_i = Slider(
    ax_i, "i [deg]", -90, 90,
    valinit=i_initial_value, valstep=2,
    initcolor='r',orientation='vertical',color="g"  # Remove the line marking the valinit position.
)


# Longitudes sliders
s_L = Slider(
    ax_L, "L [deg]", 0, 360,
    valinit=L_initial_value, valstep=2,
    initcolor='r',orientation='vertical',color="g"  # Remove the line marking the valinit position.
)

s_Lom = Slider(
    ax_Lom, "Lω [deg]", 0, 360,
    valinit=L_peri_initial_value, valstep=2,
    initcolor='r',orientation='vertical',color="g"  # Remove the line marking the valinit position.
)

s_Lgom = Slider(
    ax_Lgom, "LΩ [deg]", 0, 360,
    valinit=L_omega_initial_value, valstep=2,
    initcolor='r',orientation='vertical',color="g"  # Remove the line marking the valinit position.
)

# b_eclip_plane = RadioButtons(ax_ecl,(" Ecliptic plane ", " Earth plane ", " Two planes " ),active=(True,False,False),activecolor="white")


def update(val):
    # Values retrieval
    a = s_a.val
    e = s_e.val
    i = s_i.val
    L = s_L.val
    L_om = s_Lom.val
    L_gom = s_Lgom.val

    # Clear
    ax.clear()

    # Calculation of the new ellipse
    (X_upd, Y_upd, Z_upd), list_nodes_orbit_upd, rp_upd, ra_upd, pos_xyz_upd, true_anomaly_upd = Ellipse_draw(a, e, i,L, L_om,L_gom)
    descending_node_upd = list_nodes_orbit_upd[0]
    ascending_node_upd = list_nodes_orbit_upd[1]


    # Plot the center
    l = ax.plot3D(X_upd,Y_upd,Z_upd,'r')
    ax.scatter(0, 0, 0, s=planet_size,c='b')  # center
    # AN and DN
    ax.plot3D([ascending_node_upd[0]], [ascending_node_upd[1]], [ascending_node_upd[2]], 'go')
    ax.plot3D([descending_node_upd[0]], [descending_node_upd[1]], [descending_node_upd[2]], 'go')
    # rp and ra
    ax.plot3D([rp_upd[0]], [rp_upd[1]], [rp_upd[2]], 'ro')
    ax.plot3D([ra_upd[0]], [ra_upd[1]], [ra_upd[2]], 'ro')
    ax.plot3D([0, rp_upd[0]], [0, rp_upd[1]], [0, rp_upd[2]], 'r')

    # Gamma line
    ax.plot3D([0, 10], [0, 0], [0, 0], 'm')

    # True anomaly
    ax.plot3D([pos_xyz_upd[0]], [pos_xyz_upd[1]], [pos_xyz_upd[2]], 'wo')
    ax.plot3D([0, pos_xyz_upd[0]], [0, pos_xyz_upd[1]], [0, pos_xyz_upd[2]], 'w')

    if planes_visible == True:
        # Earth plane
        X_plane, Y_plane, Z_plane = Planes_meshgrid(45)
        ax.plot_surface(X_plane, Y_plane, Z_plane, color='gray', alpha=.4, linewidth=0, zorder=1)

        # Ecliptic plane
        X_eclip, Y_eclip, Z_eclip = Planes_meshgrid(0)
        ax.plot_surface(X_eclip, Y_eclip, Z_eclip, color='y', alpha=.4, linewidth=0, zorder=1)

        ax.text(8, 8, 0, " Ecliptic plane ")
        ax.text(8, 8, 2, " Earth plane ")
    else:
        pass

    # Orbital plane
    X_orbital_upd, Y_orbital_upd, Z_orbital_upd = Planes_meshgrid(0)
    ax.plot_surface(X_orbital_upd, Y_orbital_upd, Z_orbital_upd, color='b', alpha=.4, linewidth=0, zorder=1)

    ## Arrows
    # X arrow
    ax.plot3D([0, 0.5], [0, 0], [0, 0], 'black')
    ax.plot3D([0.5, 0.4], [0, 0.05], [0, 0], 'black')
    ax.plot3D([0.5, 0.4], [0, -0.05], [0, 0], 'black')

    # Y arrow
    ax.plot3D([0, 0], [0, 0.5], [0, 0], 'black')
    ax.plot3D([0, 0.05], [0.5, 0.4], [0, 0], 'black')
    ax.plot3D([0, -0.05], [0.5, 0.4], [0, 0], 'black')

    # Z arrow
    ax.plot3D([0, 0], [0, 0], [0, 0.5], 'black')
    ax.plot3D([0, 0], [0, 0.05], [0.5, 0.4], 'black')
    ax.plot3D([0, -0.0], [0, -0.05], [0.5, 0.4], 'black')


    ## Text annotation for ascending and descending node
    ax.text(ascending_node_upd[0], ascending_node_upd[1], ascending_node_upd[2], " Ascending node ")
    ax.text(descending_node_upd[0], descending_node_upd[1], descending_node_upd[2], " Descending node ")
    ax.text(rp_upd[0], rp_upd[1], rp_upd[2], " Perigee ")
    ax.text(ra_upd[0], ra_upd[1], ra_upd[2], " Apogee ")
    ax.text(7, 0, 0, " γ ",fontweight="bold")
    ax.text(pos_xyz_upd[0], pos_xyz_upd[1], pos_xyz_upd[2], " satellite ")
    ax.text2D(1.07, 1.05, " Parameters : " , transform=ax.transAxes, fontsize=14,color='g',fontweight="bold")
    ax.text2D(1.07, 1.0, " a = " + str(a), transform=ax.transAxes, fontsize=9, color='g')
    ax.text2D(1.07, 0.95, " e = " + str(e), transform=ax.transAxes, fontsize=9, color='g')
    ax.text2D(1.07, 0.9, " i = " + str(i) + " °", transform=ax.transAxes, fontsize=9, color='g')
    ax.text2D(1.3, 1.0, " ν = " + str(true_anomaly_upd) + " °", transform=ax.transAxes, fontsize=9,color='g')
    ax.text2D(1.3, 0.95, " Lωp = " + str(L_om)+ " °", transform=ax.transAxes, fontsize=9, color='g')
    ax.text2D(1.3, 0.9, " LΩ = " + str(L_gom)+ " °", transform=ax.transAxes, fontsize=9, color='g')

    ax.text2D(-0.57, 1.05, " Orbital elements : ", transform=ax.transAxes, fontsize=14, color='g', fontweight="bold")
    ax.text2D(-0.57, 1.0, " a : semi-major axis", transform=ax.transAxes, fontsize=9, color='g')
    ax.text2D(-0.57, 0.95, " e : eccentricity ", transform=ax.transAxes, fontsize=9, color='g')
    ax.text2D(-0.57, 0.9, " i : inclination ", transform=ax.transAxes, fontsize=9, color='g')
    ax.text2D(-0.34, 1.0, " ν : True anomaly ", transform=ax.transAxes, fontsize=9, color='g')
    ax.text2D(-0.34, 0.95, " Lωp : Longitude of the perigee  ", transform=ax.transAxes, fontsize=9, color='g')
    ax.text2D(-0.34, 0.9, " LΩ : Longitude of the ascending node ", transform=ax.transAxes, fontsize=9, color='g')

    # Arrows annotation
    ax.text(0.5, 0, 0, " X [J2000] ",color='black')
    ax.text(0, 0.5, 0, " Y [J2000] ",color='black')
    ax.text(0, 0.0, 0.5, " Z [J2000] ",color='black')


    ax.set_xlim(-scale_value,scale_value)
    ax.set_ylim(-scale_value,scale_value)
    ax.set_zlim(-scale_value,scale_value)

    ax.set_title('Keplerian elements (a,e,i,L,ωp,LΩ)', fontsize=20, fontweight="bold")

    # For the 3D update
    fig.canvas.draw_idle()


s_L.on_changed(update)
s_Lom.on_changed(update)
s_Lgom.on_changed(update)
s_a.on_changed(update)
s_e.on_changed(update)
s_i.on_changed(update)

ax.grid()

ax_reset = fig.add_axes([0.45, 0.025, 0.1, 0.04])
button = Button(ax_reset, 'Reset', hovercolor='0.33',color="blue")
#
#
def reset(event):

    s_L.reset()
    s_Lom.reset()
    s_Lgom.reset()
    s_a.reset()
    s_e.reset()
    s_i.reset()


button.on_clicked(reset)

plt.show()





# Horizontal
# ax_amp = fig.add_axes([0.25, 0.15, 0.65, 0.03])
# ax_freq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# ax_i = fig.add_axes([0.25, 0.05, 0.65, 0.03])

# create the sliders
# samp = Slider(
#     ax_amp, "Amp", 0.1, 9.0,
#     valinit=a0, valstep=allowed_amplitudes,
#     color="green"
# )
#
# sfreq = Slider(
#     ax_freq, "Freq", 0, 10*np.pi,
#     valinit=2*np.pi, valstep=np.pi,
#     initcolor='none'  # Remove the line marking the valinit position.
# )
#
# si = Slider(
#     ax_i, "Inclination", 0, 10*np.pi,
#     valinit=2*np.pi, valstep=np.pi,
#     initcolor='none'  # Remove the line marking the valinit position.
# )


# t = np.arange(0.0, 1.0, 0.001)
# a0 = 5
# f0 = 3
# s = a0 * np.sin(2 * np.pi * f0 * t)
#
# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.25)                 # bottom=0.25
# l, = ax.plot(t, s, lw=2)



# ## Source : https://stackoverflow.com/questions/43490330/a-slider-doesnt-change-a-value-in-a-3d-plot


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.widgets import Slider, Button, RadioButtons
# import variable_test as v
#
#
# def Calculation_new_Z(z,h):
#
#     new_z = []
#
#     for i in range(len(z)):
#
#         new_z.append(h + z[i])
#
#     return new_z
#
#
# fig = plt.figure()
# plt.subplots_adjust(bottom=0.25)
# ax = fig.add_subplot(121, projection='3d')
#
# ax.plot3D(v.X,v.Y,v.Z)
# plt.axis()
#
#
#
# ax2 = fig.add_subplot(122, projection='3d')
#
#
# l=ax2.plot3D(v.X,v.Y,v.Z)  # ,rstride=2, cstride=2
#
# axhauteur = plt.axes([0.2, 0.1, 0.65, 0.03])
# shauteur = Slider(axhauteur, 'Hauteur', 0.5, 10.0, valinit=0)
# slargeur = Slider(axhauteur, 'Largeur', 0.5, 10.0, valinit=0).orientation
#
#
# def update(val):
#     h = shauteur.val
#     ax2.clear()
#     new_Z = Calculation_new_Z(v.Z,h)
#     l=ax2.plot3D(v.X,v.Y,new_Z) # ,rstride=2, cstride=2
#     ax2.set_zlim(0,10)
#     fig.canvas.draw_idle()
# shauteur.on_changed(update)
# ax2.set_zlim(0,10)
#
# plt.show()

