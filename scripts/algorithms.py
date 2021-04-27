import numpy as np
from mat import mat
from utils import in_half_plane, s_norm, Rz, angle, i2p


class Algorithms:
    def __init__(self):
        self.i = 0
        self.state = 0

    def pathFollower(self, flag, r, q, p, chi, chi_inf, k_path, c, rho, lamb, k_orbit):
        """
        Input:
            flag = 1 for straight line, 2 for orbit
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)
            p = current position of uav in NED (m)
            chi = course angle of UAV (rad)
            chi_inf = straight line path following parameter
            k_path = straight line path following parameter
            c = center of orbit in NED (m)
            rho = radius of orbit (m)
            lamb = direction of orbit, 1 clockwise, -1 counter-clockwise
            k_orbit = orbit path following parameter

        Outputs:
            e_crosstrack = crosstrack error (m)
            chi_c = commanded course angle (rad)
            h_c = commanded altitude (m)

        Example Usage
            e_crosstrack, chi_c, h_c = pathFollower(path)

        Reference: Beard, Small Unmanned Aircraft, Chapter 10, Algorithms 3 and 4
        Copyright 2018 Utah State University
        """

        if flag == 1:  # straight line
            chi_q = np.arctan2(q[1,0],q[0,0])
            rd = r[2,0]
            ki = mat([0,0,1]).T
            epi = p-r
            R_ip = mat([[np.cos(chi_q),np.sin(chi_q),0],[-np.sin(chi_q),np.cos(chi_q),0],[0,0,1]])
            
            error = R_ip*(epi)
            q_ki = mat(np.cross(q.T,ki.T)).T
            n = q_ki/np.linalg.norm(q_ki)
            s = epi-((epi.T*n)[0,0])*n
            
            while ((chi_q-chi)<(-np.pi)):
                chi_q = (chi_q + (2*np.pi))
            
            while ((chi_q - chi) > np.pi):
                chi_q = (chi_q - 2*np.pi)
            
            e_crosstrack = error[1,0]
            chi_d  = -chi_inf*(2/(np.pi))*np.arctan(k_path*e_crosstrack)
            
            chi_c = chi_q+chi_d
            h_c = (-rd-((np.sqrt((pow(s[0,0],2)+pow(s[1,0],2))))*(q[2,0]/(np.sqrt((pow(q[0,0],2)+pow(q[1,0],2)))))))
        elif flag == 2:  # orbit following
            d = np.sqrt((pow((p[0,0]-c[0,0]),2)+pow((p[1,0]-c[1,0]),2)))
            phi = np.arctan2((p[1,0]-c[1,0]),(p[0,0]-c[0,0]))
            
            while ((phi - chi)<(-np.pi)):
                phi = (phi + (2*np.pi))
            
            while ((phi - chi) > np.pi):
                phi = (phi - (2*np.pi))
            
            e_crosstrack = d-rho
	    print(e_crosstrack)
            chi_c = phi + lamb*(((np.pi/2)+np.arctan2((k_orbit*(d-rho)),rho))) #+0.6
	    print(chi)
	    print(chi_c)
            h_c = -c[2,0]
        else:
            raise Exception("Invalid path type")

        return e_crosstrack, chi_c, h_c

    # followWpp algorithm left here for reference
    # It is not used in the final implementation
    def followWpp(self, w, p, newpath):
        """
        followWpp implements waypoint following via connected straight-line
        paths.

        Inputs:
            w = 3xn matrix of waypoints in NED (m)
            p = position of MAV in NED (m)
            newpath = flag to initialize the algorithm or define new waypoints

        Outputs
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)

        Example Usage;
            r, q = followWpp(w, p, newpath)

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 5
        Copyright 2018 Utah State University
        """

        if self.i is None:
            self.i = 0

        if newpath:
            # initialize index
            self.i = 1

        # check sizes
        m, N = w.shape
        assert N >= 3
        assert m == 3

        # calculate the q vector
        r = w[:, self.i - 1]
        qi1 = s_norm(w[:, self.i], -w[:, self.i - 1])

        # Calculate the origin of the current path
        qi = s_norm(w[:, self.i + 1], -w[:, self.i])

        # Calculate the unit normal to define the half plane
        ni = s_norm(qi1, qi)

        # Check if the MAV has crossed the half-plane
        if in_half_plane(p, w[:, self.i], ni):
            if self.i < (N - 2):
                self.i += 1
        q = qi1

        return r, q

    # followWppFillet algorithm left here for reference.
    # It is not used in the final implementation
    def followWppFillet(self, w, p, R, newpath):
        """
        followWppFillet implements waypoint following via straightline paths
        connected by fillets

        Inputs:
            W = 3xn matrix of waypoints in NED (m)
            p = position of MAV in NED (m)
            R = fillet radius (m)
            newpath = flag to initialize the algorithm or define new waypoints

        Outputs
            flag = flag for straight line path (1) or orbit (2)
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)
            c = center of orbit in NED (m)
            rho = radius of orbit (m)
            lamb = direction or orbit, 1 clockwise, -1 counter clockwise

        Example Usage
            [flag, r, q, c, rho, lamb] = followWppFillet( w, p, R, newpath )

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 6
        Copyright 2018 Utah State University
        """

        if self.i is None:
            self.i = 0
            self.state = 0
        if newpath:
            # Initialize the waypoint index
            self.i = 2
            self.state = 1

            # Check size of waypoints matrix
            m, N = w.shape  # Where 'N' is the number of waypoints and 'm' dimensions
            assert N >= 3
            assert m == 3
        else:
            [m, N] = w.shape
            assert N >= 3
            assert m == 3
        # Calculate the q vector and fillet angle
        qi1 = mat(s_norm(w[:, self.i], -w[:, self.i - 1]))
        qi = mat(s_norm(w[:, self.i + 1], -w[:, self.i]))
        e = acos(-qi1.T * qi)

        # Determine if the MAV is on a straight or orbit path
        if self.state == 1:
            c = mat([0, 0, 0]).T
            rho = 0
            lamb = 0

            flag = 1
            r = w[:, self.i - 1]
            q = q1
            z = w[:, self.i] - (R / (np.tan(e / 2))) * qi1
            if in_half_plane(p, z, qi1):
                self.state = 2

        elif self.state == 2:
            r = [0, 0, 0]
            q = [0, 0, 0]

            flag = 2
            c = w[:, self.i] - (R / (np.sin(e / 2))) * s_norm(qi1, -qi)
            rho = R
            lamb = np.sign(qi1(1) * qi(2) - qi1(2) * qi(1))
            z = w[:, self.i] + (R / (np.tan(e / 2))) * qi

            if in_half_plane(p, z, qi):
                if self.i < (N - 1):
                    self.i = self.i + 1
                self.state = 1

        else:
            # Fly north as default
            flag = -1
            r = p
            q = mat([1, 0, 0]).T
            c = np.nan(3, 1)
            rho = np.nan
            lamb = np.nan

        return flag, r, q, c, rho, lamb

    def findDubinsParameters(self, p_s, chi_s, p_e, chi_e, R):
        """
        findDubinsParameters determines the dubins path parameters

        Inputs:
        p_s = start position (m)
        chi_s = start course angle (rad)
        p_e = end position (m)
        chi_e = end course angle (rad)
        R = turn radius (m)

        Outputs
        dp.L = path length (m)
        dp.c_s = start circle origin (m)
        dp.lamb_s = start circle direction (unitless)
        dp.c_e = end circle origin (m)
        dp.lamb_e = end circle direction (unitless)
        dp.z_1 = Half-plane H_1 location (m)
        dp.q_12 = Half-planes H_1 and H_2 unit normals (unitless)
        dp.z_2 = Half-plane H_2 location (m)
        dp.z_3 = Half-plane H_3 location  (m)
        dp.q_3 = Half-plane H_3 unit normal (m)
        dp.case = case (unitless)

        Example Usage
        dp = findDubinsParameters( p_s, chi_s, p_e, chi_e, R )

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 7
        Copyright 2018 Utah State University
        """

        # TODO Algorithm 7 goes here
        pi = np.pi
        chi_e = float(chi_e)
        chi_s = float(chi_s)

        assert np.linalg.norm(p_s[0:2] - p_e[0:2]) >= 3*R

        el = mat([1, 0, 0]).T

        cs = np.cos(chi_s)
        ss = np.sin(chi_s)
        ce = np.cos(chi_e)
        se = np.sin(chi_e)

        c_rs = p_s + R * Rz(pi/2) * mat([cs, ss, 0]).T          # [np.cos(chi_s), np.sin(chi_s), 0]
        c_ls = p_s + R * Rz(-pi / 2) * mat([cs, ss, 0]).T       # [np.cos(chi_s), np.sin(chi_s), 0]
        c_re = p_e + R * Rz(pi / 2) * mat([ce, se, 0]).T        # [np.cos(chi_e), np.sin(chi_e), 0]
        c_le = p_e + R * Rz(-pi / 2) * mat([ce, se, 0]).T     # [np.cos(chi_e), np.sin(chi_e), 0]

        #Case 1 R-S-R
        th = angle(c_re - c_rs)
        L1 = np.linalg.norm(c_rs-c_re) + R*(2*pi + (th-pi/2)%(2*pi) - (chi_s-pi/2)%(2*pi))%(2*pi) + R*(2*pi + (chi_e-pi/2)%(2*pi) - (th-pi/2)%(2*pi))%(2*pi)

        #Case 2 R-S-L
        th = angle(c_le-c_rs)
        ell = np.linalg.norm(c_le-c_rs)
        th2 = th - pi/2 + np.arcsin(2*R/ell)

        if not np.isreal(th2):
            L2 = np.nan
        else:
            L2 = np.sqrt(ell**2-4*R**2) + R*(2*pi + (th2)%(2*pi) - (chi_s-pi/2)%(2*pi))%(2*pi) + R*(2*pi + (th2 + pi)%(2*pi) - (chi_e + pi/2)%(2*pi))%(2*pi)

        #Case 3 L-S-R
        th = angle(c_re - c_ls)
        ell = np.linalg.norm(c_re - c_ls)
        th2 = np.arccos(2 * R / ell)

        if not np.isreal(th2):
            L3 = np.nan
        else:
            L3 = np.sqrt(ell**2 - 4*R**2) + R*(2*pi + (chi_s+pi/2)%(2*pi) - (th+th2)%(2*pi))%(2*pi) + R*(2*pi + (chi_e-pi/2)%(2*pi) - (th+th2-pi)%(2*pi))%(2*pi)

        #Case 4 L-S-L
        th = angle(c_le-c_ls)
        L4 = np.linalg.norm(c_ls-c_le) + R*(2*pi + (chi_s+pi/2)%(2*pi) - (th+pi/2)%(2*pi))%(2*pi) + R*(2*pi + (th+pi/2)%(2*pi) - (chi_e+pi/2)%(2*pi))%(2*pi)

        # Define the parameters for the minimum length path
        lengths = np.array([L1, L2, L3, L4])
        L, i_min = np.min(lengths), np.argmin(lengths)

        # package output into DubinsParameters class
        dp = DubinsParameters()

        if i_min == 0:
            dp.c_s = c_rs
            dp.lambda_s = 1
            dp.c_e = c_re
            dp.lambda_e = 1
            dp.q_1 = (dp.c_e - dp.c_s) / np.linalg.norm(dp.c_e - dp.c_s)
            dp.z_1 = dp.c_s + R * Rz(-pi / 2) * dp.q_1
            dp.z_2 = dp.c_e + R * Rz(-pi / 2) * dp.q_1

        elif i_min == 1:
            dp.c_s = c_rs
            dp.lambda_s = 1
            dp.c_e = c_le
            dp.lambda_e = -1
            dp.ell = np.linalg.norm(dp.c_e - dp.c_s)
            dp.th = float(angle(dp.c_e - dp.c_s))
            th2 = dp.th - pi / 2 + np.arcsin(2 * R / dp.ell)
            dp.q_1 = Rz(th2 + pi / 2) * el
            dp.z_1 = dp.c_s + R * Rz(th2) * el
            dp.z_2 = dp.c_e + R * Rz(th2 + pi) * el

        elif i_min == 2:
            dp.c_s = c_ls
            dp.c_e = c_re
            dp.lambda_s = -1
            dp.lambda_e = 1
            dp.ell = np.linalg.norm(dp.c_e - dp.c_s)
            dp.th = float(angle(dp.c_e - dp.c_s))
            th2 = np.arccos(2 * R / dp.ell)
            dp.q_1 = Rz(dp.th + th2 - pi / 2) * el
            dp.z_1 = dp.c_s + R * Rz(dp.th + th2) * el
            dp.z_2 = dp.c_e + R * Rz(dp.th + th2 - pi) * el

        elif i_min == 3:
            dp.c_s = c_ls
            dp.c_e = c_le
            dp.lambda_s = -1
            dp.lambda_e = -1
            dp.q_1 = (dp.c_e - dp.c_s) / np.linalg.norm(dp.c_e - dp.c_s)
            dp.z_1 = dp.c_s + R * Rz(pi / 2) * dp.q_1
            dp.z_2 = dp.c_e + R * Rz(pi / 2) * dp.q_1

        dp.z_3 = p_e
        dp.q_3 = Rz(chi_e)*el
        dp.case = i_min
        dp.lengths = lengths
        dp.c_rs = c_rs
        dp.c_ls = c_ls
        dp.c_re = c_re
        dp.c_le = c_le

        # # TODO populate dp members here
        # dp.L = L
        # dp.c_s = c_s
        # dp.lambda_s = lambda_s
        # dp.c_e = c_e
        # dp.lambda_e = lambda_e
        # dp.z_1 = z_1
        # dp.q_1 = q_1
        # dp.z_2 = z_2
        # dp.z_3 = z_3
        # dp.q_3 = q_3
        # dp.case = i_min
        # dp.lengths = [L1, L2, L3, L4]
        # dp.theta = th
        # dp.ell = ell
        # dp.c_rs = c_rs
        # dp.c_ls = c_ls
        # dp.c_re = c_re
        # dp.c_le = c_le

        return dp

    def followWppDubins(self, W, Chi, p, R, newpath):
        """
        followWppDubins implements waypoint following via Dubins paths

        Inputs:
            W = list of waypoints in NED (m)
            Chi = list of course angles at waypoints in NED (rad)
            p = mav position in NED (m)
            R = fillet radius (m)
            newpath = flag to initialize the algorithm or define new waypoints

        Outputs
            flag = flag for straight line path (1) or orbit (2)
            r = origin of straight-line path in NED (m)
            q = direction of straight-line path in NED (m)
            c = center of orbit in NED (m)
            rho = radius of orbit (m)
            lamb = direction or orbit, 1 clockwise, -1 counter clockwise
            self.i = waypoint number
            dp = dubins path parameters

        Example Usage
            flag, r, q, c, rho, lamb = followWppDubins(W, Chi, p, R, newpath)

        Reference: Beard, Small Unmanned Aircraft, Chapter 11, Algorithm 8
        Copyright 2018 Utah State University
        """

        # TODO Algorithm 8 goes here


        if self.i is None:
            self.i = 0
            self.state = 0
        if newpath:
            self.i = 1
            self.state = 1
            [m,N] = W.shape
            assert (N >= 3)
            assert (m == 3)
        else:
            [m, N] = W.shape
            assert (N >= 3)
            assert (m == 3)

        wi = W[:, self.i]
        wiMinus = W[:, self.i-1]

        chi_i = Chi[self.i]
        chi_minus = Chi[self.i - 1]
        
        dp = self.findDubinsParameters(wiMinus, chi_minus, wi, chi_i, R)

        flag = 1
        r = mat([[0, 0, 0]]).T
        q = mat([[0, 0, 0]]).T
        c = mat([[0, 0, 0]]).T
        rho = np.Inf
        lamb = 1

        if self.state == 1:
            flag = 2
            c = dp.c_s
            rho = R
            lamb = dp.lambda_s
            if in_half_plane(p, dp.z_1, -dp.q_1):
                self.state = 2
        elif self.state == 2:
            flag = 2
            c = dp.c_s
            rho = R
            lamb = dp.lambda_s
            if in_half_plane(p, dp.z_1, dp.q_1):
                self.state = 3
        elif self.state == 3:
            flag = 1
            r = dp.z_1
            q = dp.q_1
            if in_half_plane(p, dp.z_2, dp.q_1):
                self.state = 4
        elif self.state == 4:
            flag = 2
            c = dp.c_e
            rho = R
            lamb = dp.lambda_e
            if in_half_plane(p, dp.z_3, dp.q_3):
                self.state = 5
        else:
            flag = 2
            c = dp.c_e
            rho = R
            lamb = dp.lambda_e
            if in_half_plane(p, dp.z_3, dp.q_3):
                self.state = 1
                if self.i < N:
                    self.i += 1

        return flag, r, q, c, rho, lamb, self.i, dp

    def Rz(th):
        c = float(np.cos(theta))
        s = float(np.sin(theta))

        return mat([[c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]])

    def angle(v):
        out = np.arctan2(v[1],v[0])
        return out

class DubinsParameters:
    def __init__(self):
        """
        Member Variables:
            L = path length (m)
            c_s = start circle origin (m)
            lamb_s = start circle direction (unitless)
            c_e = end circle origin (m)
            lamb_e = end circle direction (unitless)
            z_1 = Half-plane H_1 location (m)
            q_1 = Half-planes H_1 and H_2 unit normals (unitless)
            z_2 = Half-plane H_2 location (m)
            z_3 = Half-plane H_3 location  (m)
            q_3 = Half-plane H_3 unit normal (m)
            case = case (unitless)
        """

        self.L = 0
        self.c_s = mat([0, 0, 0]).T
        self.lamb_s = -1
        self.c_e = mat([0, 0, 0]).T
        self.lamb_e = 1
        self.z_1 = mat([0, 0, 0]).T
        self.q_1 = mat([0, 0, 0]).T
        self.z_2 = mat([0, 0, 0]).T
        self.z_3 = mat([0, 0, 0]).T
        self.q_3 = mat([0, 0, 0]).T
        self.case = 0
        self.lengths = np.array([[0, 0, 0, 0]])
        self.theta = 0
        self.ell = 0
        self.c_rs = mat([0, 0, 0]).T
        self.c_ls = mat([0, 0, 0]).T
        self.c_re = mat([0, 0, 0]).T
        self.c_le = mat([0, 0, 0]).T
