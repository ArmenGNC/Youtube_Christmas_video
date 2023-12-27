import numpy as np



class Ellipse_drawing():

    def __init__(self,a,e,i,L,L_peri,L_node):
        self.a = a
        self.e = e
        self.i = i
        self.L = L
        self.Lperi = L_peri
        self.Lnode = L_node
        self.epsilon = 1e-6
        self.deg2rad = np.pi/180

    def equationKepler(self, e, M, epsilon):
        E = M
        delta = 1e6
        while delta > epsilon:
            new_E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
            delta = abs(new_E - E)
            E = new_E
        return E

    def position_xy(self):
        L = self.L
        Lperi = self.Lperi
        M = L - Lperi
        e = self.e
        E = self.equationKepler(e, M * self.deg2rad, self.epsilon)
        a = self.a
        x = a * (np.cos(E) - e)
        y = a * np.sqrt(1 - e * e) * np.sin(E)
        return (x, y), M

    def ellipse_xy(self,N=1000):
        E = np.linspace(0, 2 * np.pi, N)
        a = self.a
        e = self.e
        x = a * (np.cos(E) - e)
        y = a * np.sqrt(1 - e * e) * np.sin(E)
        return (x, y)

    def xy_XYZ(self, x, y):
        I = self.i
        Lperi = self.Lperi
        Omega = self.Lnode
        I *= self.deg2rad
        Omega *= self.deg2rad
        Lperi *= self.deg2rad
        omega = Lperi - Omega
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        R1 = np.array([[cos_omega, -sin_omega, 0], [sin_omega, cos_omega, 0], [0, 0, 1]])
        cos_I = np.cos(I)
        sin_I = np.sin(I)
        R2 = np.array([[1, 0, 0], [0, cos_I, -sin_I], [0, sin_I, cos_I]])
        cos_Omega = np.cos(Omega)
        sin_Omega = np.sin(Omega)
        R3 = np.array([[cos_Omega, -sin_Omega, 0], [sin_Omega, cos_Omega, 0], [0, 0, 1]])
        R = np.dot(R3, np.dot(R2, R1))
        xyz = np.array([x, y, 0])
        XYZ = np.dot(R, xyz)
        return XYZ

    def position_XYZ(self):
        (x, y), M = self.position_xy()
        return self.xy_XYZ(x, y) , M

    def ellipse_XYZ(self, N=1500):
        (x, y) = self.ellipse_xy(N=1500)
        X = np.zeros(N)
        Y = np.zeros(N)
        Z = np.zeros(N)
        for k in range(N):
            XYZ = self.xy_XYZ(x[k], y[k])
            X[k] = XYZ[0]
            Y[k] = XYZ[1]
            Z[k] = XYZ[2]
        return (X, Y, Z)