from numpy import sqrt,power,pi,radians,degrees,sin,cos

class SingleCrackStress:
    """
    Exact solution for flat crack
    """
    def __init__(self,theta,theta1,theta2,r,r1,r2,stress,crack_length) -> None:
        self.theta = theta
        self.theta1 = theta1
        self.theta2 = theta2
        self.r = r
        self.r1 = r1
        self.r2 = r2
        self.stress = stress
        self.a = crack_length
    
    def stress_x(self):
        left = self.stress*self.r / sqrt(self.r1*self.r2)
        left = left * cos(radians(self.theta - ((self.theta1+self.theta2)/2)))
        right = (self.stress * power(self.a,2)) / power(self.r1*self.r2,1.5)
        right = right * self.r1 * sin(radians(self.theta1)) * sin(radians(1.5 * (self.theta1+self.theta2)))
        sx = left - right
        self.x = sx
        return sx
    
    def stress_y(self):
        left = self.stress*self.r / sqrt(self.r1*self.r2)
        left = left * cos(radians(self.theta - ((self.theta1+self.theta2)/2)))
        right = (self.stress * power(self.a,2)) / power(self.r1*self.r2,1.5)
        right = right * self.r1 * sin(radians(self.theta1)) * sin(radians(1.5 * (self.theta1+self.theta2)))
        sy = left + right
        self.y = sy
        return sy
    
    def tau_xy(self):
        p1 = (self.stress * power(self.a,2)) / power(self.r1*self.r2,1.5)
        p2 = self.r1 * sin(radians(self.theta1))
        p3 = sin(radians(1.5 * (self.theta1+self.theta2)))
        txy = p1*p2*p3
        self.xy = txy
        return txy
    
    def calculate_stresses(self):
        self.stress_x()
        self.stress_y()
        self.tau_xy()
    

def near_field_stress_y(stress,crack_len,r,theta):
    p1 = stress * sqrt(pi * crack_len)
    p2 = (1 / sqrt(2*pi*r))
    p3 = cos(radians(theta/2)) * (1 + (sin(radians(theta/2)) * sin(radians(theta*1.5))))
    result = p1 * (p2 * p3)
    return result


def k_one(stress,crack_len):
    k1 = stress * sqrt(pi*crack_len)
    return k1

def sig_theta1(k1,r,theta):
    mode1 = (k1 / sqrt(2*pi*r)) * power(cos(radians(theta/2)), 3)
    return mode1

def sig_theta2(k2,r,theta):
    p1 = -k2 / sqrt(2*pi*r)
    p2 = 3 * sin(radians(theta/2))
    p3 = power(cos(radians(theta/2)), 2)
    mode2 = p1*p2*p3
    return mode2
