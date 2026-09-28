# quat_arm.py -- quaternion kinematics for a revolute-joint robot arm (MicroPython)
#
# Features
#   * Quat class (Hamilton convention, w,x,y,z): multiply, conjugate, rotate, slerp
#   * Generic forward kinematics from a joint table (axis + link offset per joint)
#   * Damped least-squares inverse kinematics (works for any joint layout)
#   * Smooth motion by slerping each joint quaternion (smoothstep easing)
#   * Hobby-servo PWM output (50 Hz)
#
# Pure MicroPython: only uses math, time, machine. No numpy.

import math
import time

try:
    from machine import Pin, PWM
except ImportError:  # lets you run/test on desktop Python
    class Pin:
        def __init__(self, *a, **k): pass
    class PWM:
        def __init__(self, *a, **k): pass
        def duty_u16(self, v): pass
    if not hasattr(time, "sleep_ms"):
        time.sleep_ms = lambda ms: time.sleep(ms / 1000)


# ---------------------------------------------------------------- vec3 helpers
def v_add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def v_sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def v_dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def v_norm(a): return math.sqrt(v_dot(a, a))
def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _solve3(A, b):
    """Solve 3x3 system A x = b via adjugate/determinant."""
    a, bb, c = A[0]
    d, e, f = A[1]
    g, h, i = A[2]
    det = a * (e * i - f * h) - bb * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        det = 1e-12
    inv = (
        ((e * i - f * h), (c * h - bb * i), (bb * f - c * e)),
        ((f * g - d * i), (a * i - c * g), (c * d - a * f)),
        ((d * h - e * g), (bb * g - a * h), (a * e - bb * d)),
    )
    return tuple(sum(inv[r][k] * b[k] for k in range(3)) / det for r in range(3))


# ------------------------------------------------------------------ quaternion
class Quat:
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w, self.x, self.y, self.z = w, x, y, z

    @staticmethod
    def from_axis_angle(axis, angle):
        n = v_norm(axis)
        s = math.sin(angle / 2) / n
        return Quat(math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s)

    def __mul__(self, o):  # Hamilton product: self * o
        return Quat(
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        )

    def conj(self):
        return Quat(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self):
        n = self.norm()
        return Quat(self.w / n, self.x / n, self.y / n, self.z / n)

    def dot(self, o):
        return self.w * o.w + self.x * o.x + self.y * o.y + self.z * o.z

    def rotate(self, v):
        """Rotate vector v by this (unit) quaternion: v' = v + 2w(u x v) + 2u x (u x v)."""
        u = (self.x, self.y, self.z)
        t = v_cross(u, v)
        t = (2 * t[0], 2 * t[1], 2 * t[2])
        c = v_cross(u, t)
        return (v[0] + self.w * t[0] + c[0],
                v[1] + self.w * t[1] + c[1],
                v[2] + self.w * t[2] + c[2])

    def slerp(self, o, t):
        d = self.dot(o)
        if d < 0:  # take the short way around
            o = Quat(-o.w, -o.x, -o.y, -o.z)
            d = -d
        if d > 0.9995:  # nearly identical -> lerp
            return Quat(self.w + t * (o.w - self.w),
                        self.x + t * (o.x - self.x),
                        self.y + t * (o.y - self.y),
                        self.z + t * (o.z - self.z)).normalized()
        th = math.acos(d)
        s = math.sin(th)
        a = math.sin((1 - t) * th) / s
        b = math.sin(t * th) / s
        return Quat(a * self.w + b * o.w, a * self.x + b * o.x,
                    a * self.y + b * o.y, a * self.z + b * o.z)

    def angle_about(self, axis):
        """Signed rotation angle (rad) of this quat about a unit axis, in (-pi, pi]."""
        n = v_norm(axis)
        s = (self.x * axis[0] + self.y * axis[1] + self.z * axis[2]) / n
        ang = 2 * math.atan2(s, self.w)
        if ang > math.pi:
            ang -= 2 * math.pi
        elif ang <= -math.pi:
            ang += 2 * math.pi
        return ang


# ----------------------------------------------------------------------- servo
class Servo:
    """Hobby servo on a PWM pin. write(rad) takes the JOINT angle in radians,
    where 0 rad = servo centre (offset_deg)."""

    def __init__(self, pin, min_us=500, max_us=2500, range_deg=180, offset_deg=90, invert=False):
        self.pwm = PWM(Pin(pin), freq=50)
        self.min_us, self.max_us = min_us, max_us
        self.range_deg, self.offset_deg = range_deg, offset_deg
        self.sign = -1 if invert else 1

    def write(self, rad):
        deg = self.offset_deg + self.sign * math.degrees(rad)
        deg = max(0, min(self.range_deg, deg))
        us = self.min_us + (deg / self.range_deg) * (self.max_us - self.min_us)
        self.pwm.duty_u16(int(us / 20000 * 65535))


# ------------------------------------------------------------------------- arm
class Arm:
    """
    joints : list of dicts, one per revolute joint, in order from the base:
        'axis'   : unit rotation axis in the parent frame  e.g. (0,0,1)
        'offset' : vector from the previous joint to this joint at zero pose
        'limits' : (min_rad, max_rad)
    tool   : vector from the last joint to the end effector at zero pose
    servos : optional list of Servo, one per joint
    """

    def __init__(self, joints, tool, servos=None):
        self.joints = joints
        self.tool = tool
        self.servos = servos
        self.n = len(joints)
        self.q = [Quat() for _ in joints]      # current joint quaternions
        self.theta = [0.0] * self.n

    # -- forward kinematics: chain of quaternion products
    def fk(self, thetas):
        qa = Quat()
        p = (0.0, 0.0, 0.0)
        frames = []
        for j, th in zip(self.joints, thetas):
            p = v_add(p, qa.rotate(j["offset"]))
            frames.append((p, qa.rotate(j["axis"])))   # joint pos + axis in world
            qa = qa * Quat.from_axis_angle(j["axis"], th)
        p = v_add(p, qa.rotate(self.tool))
        return p, qa.normalized(), frames

    # -- inverse kinematics: damped least squares using the geometric Jacobian
    def ik(self, target, start=None, iters=80, tol=1e-3, damping=0.05, max_step=0.3):
        th = list(start if start is not None else self.theta)
        for _ in range(iters):
            p, _, frames = self.fk(th)
            e = v_sub(target, p)
            if v_norm(e) < tol:
                return th, True
            cols = [v_cross(ax, v_sub(p, jp)) for jp, ax in frames]   # revolute joint Jacobian
            A = [[sum(cols[k][r] * cols[k][c] for k in range(self.n)) + (damping ** 2 if r == c else 0.0)
                  for c in range(3)] for r in range(3)]
            y = _solve3(A, e)
            for k in range(self.n):
                d = v_dot(cols[k], y)
                d = max(-max_step, min(max_step, d))
                lo, hi = self.joints[k]["limits"]
                th[k] = max(lo, min(hi, th[k] + d))
        p, _, _ = self.fk(th)
        return th, v_norm(v_sub(target, p)) < tol * 5

    # -- push joint quaternions out to the servos
    def _apply(self, quats):
        self.q = quats
        self.theta = [q.angle_about(j["axis"]) for q, j in zip(quats, self.joints)]
        if self.servos:
            for s, th in zip(self.servos, self.theta):
                s.write(th)

    def set_angles(self, thetas):
        self._apply([Quat.from_axis_angle(j["axis"], t) for j, t in zip(self.joints, thetas)])

    # -- smooth Cartesian move: slerp each joint quaternion start -> goal
    def move_to(self, target, duration_ms=1000, steps=40):
        th, ok = self.ik(target)
        if not ok:
            return False  # unreachable / outside limits
        goal = [Quat.from_axis_angle(j["axis"], t) for j, t in zip(self.joints, th)]
        start = list(self.q)
        dt = max(1, duration_ms // steps)
        for s in range(1, steps + 1):
            t = s / steps
            t = t * t * (3 - 2 * t)    # smoothstep ease-in/out
            self._apply([a.slerp(b, t) for a, b in zip(start, goal)])
            time.sleep_ms(dt)
        return True

    def position(self):
        return self.fk(self.theta)[0]


# --------------------------------------------------------------------- example
def make_default_arm(L0=0.05, L1=0.10, L2=0.10, pins=(15, 16, 17)):
    """Base yaw (Z), shoulder pitch (Y), elbow pitch (Y); arm points straight up at zero."""
    lim = (-math.pi / 2, math.pi / 2)
    joints = [
        {"axis": (0, 0, 1), "offset": (0, 0, 0),  "limits": lim},   # base yaw
        {"axis": (0, 1, 0), "offset": (0, 0, L0), "limits": lim},   # shoulder
        {"axis": (0, 1, 0), "offset": (0, 0, L1), "limits": lim},   # elbow
    ]
    servos = [Servo(p) for p in pins]
    return Arm(joints, tool=(0, 0, L2), servos=servos)


if __name__ == "__main__":
    arm = make_default_arm()
    arm.set_angles([0, 0, 0])
    for tgt in [(0.15, 0.00, 0.10), (0.10, 0.10, 0.15), (0.00, 0.15, 0.10)]:
        ok = arm.move_to(tgt, duration_ms=1200)
        print("target", tgt, "reached" if ok else "UNREACHABLE", "-> pos", arm.position())
        time.sleep_ms(300)