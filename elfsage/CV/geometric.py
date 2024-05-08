import math
import cv2
import numpy as np
import scipy.cluster.hierarchy as sch


class Figure:
    def __init__(self):
        pass


class Line(Figure):
    def __init__(self, points=None, param=None, polar=None):
        super().__init__()

        self._p1 = None
        self._p2 = None
        self._a = None
        self._b = None
        self._rho = None
        self._theta = None

        if points is not None:
            if not isinstance(points, (tuple, list)):
                raise TypeError('Points must be either tuple or list')

            self._p1 = points[0]
            self._p2 = points[1]

            self._theta = math.atan(- (self._p1[0] - self._p2[0]) / (self._p1[1] - self._p2[1]))\
                if self._p1[1] - self._p2[1] != 0 else math.pi / 2
            self._rho = math.sin(self._theta) * self._p1[1] + math.cos(self._theta) * self._p1[0]

        elif param is not None:
            if not isinstance(param, (tuple, list)):
                raise TypeError('Line parameters must be either tuple or list')

            self._a = param[0]
            self._b = param[1]

            self._theta = math.atan(- 1 / self._a) if self._a != 0 else math.pi/2
            self._rho = self._b * math.sin(self._theta)

        elif polar is not None:
            if not isinstance(polar, (tuple, list)):
                raise TypeError('Radial parameters must be either tuple or list')

            self._rho = polar[0]
            self._theta = polar[1]

        if self._rho is not None and self._theta is not None:
            self._theta %= 2 * math.pi

            if self._rho < 0:
                self._rho = abs(self._rho)
                self._theta += math.pi

            if self._theta > math.pi:
                self._theta -= 2*math.pi

    def __str__(self):
        return f'[rho={self.rho}, theta={self.theta}, theta_deg={self.theta_deg}]'

    def __repr__(self):
        return f'[rho={self.rho}, theta={self.theta}, theta_deg={self.theta_deg}]'

    @staticmethod
    def by_points(p1, p2):
        return Line(points=(p1, p2))

    @staticmethod
    def by_params(a, b):
        return Line(param=(a, b))

    @staticmethod
    def by_polar(rho, theta):
        return Line(polar=(rho, theta))

    @staticmethod
    def by_vector(x, y):
        if x == 0 and y == 0:
            raise ValueError('Can not define line by zero-length vector')

        theta = math.atan2(y, x)
        rho = x / math.cos(theta)

        return Line(polar=(rho, theta))

    @property
    def rho(self):
        return self._rho

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        angle = value % (2 * math.pi)

        if angle > math.pi:
            angle -= 2 * math.pi

        self._theta = angle

    @property
    def theta_deg(self):
        return self.theta / math.pi * 180

    @property
    def is_vertical(self):
        return self.theta_deg % 180 == 0

    @property
    def is_horizontal(self):
        return self.theta_deg % 90 == 0

    @property
    def points(self):
        if self._p1 is not None and self._p2 is not None:
            return np.array([self._p1, self._p2])

        return np.array([])

    @property
    def polar(self):
        return np.array([self.rho, self.theta])

    @property
    def vector(self):
        return np.array([self.rho*math.cos(self.theta), self.rho*math.sin(self.theta)])

    def y(self, x):
        if self.is_vertical:
            return float('inf')

        return - math.cos(self.theta) / math.sin(self.theta) * x + self.rho / math.sin(self.theta)

    def draw(self, image, color, thickness, continuous=False, origin=(0, 0)):
        origin = np.array(origin)

        if not continuous and self._p1 is not None and self._p2 is not None:
            p1 = self._p1
            p2 = self._p2
        else:
            if self.is_vertical:
                p1 = (self.rho+origin[0], 0)
                p2 = (self.rho+origin[0], image.shape[0])
            else:
                p1 = (0, self.y(-origin[0])+origin[1])
                p2 = (image.shape[1]-1, self.y(image.shape[1]-1-origin[0])+origin[1])

        p1 = np.array(p1).astype(np.int32)
        p2 = np.array(p2).astype(np.int32)

        cv2.line(image, p1, p2, color, thickness)

        return image

    def rotate(self, angle):
        self.theta += angle

        return self

    def intersection(self, line):
        if (self.theta % math.pi) == (line.theta % math.pi):
            return np.array([np.inf, np.inf])

        x = (self.rho*math.sin(line.theta) - line.rho*math.sin(self.theta)) / (math.cos(self.theta)*math.sin(line.theta) - math.cos(line.theta)*math.sin(self.theta))
        y = (self.rho*math.cos(line.theta) - line.rho*math.cos(self.theta)) / (math.sin(self.theta)*math.cos(line.theta) - math.sin(line.theta)*math.cos(self.theta))

        return np.array([x, y])


class FigureCollection(list):
    def __init__(self, figures=()):
        if len(figures) == 0:
            raise ValueError('Can not create empty collection')

        for item in figures:
            if not isinstance(item, Figure):
                raise ValueError('Collection may contain only geometric objects')

        super().__init__(figures)

    def append(self, __object):
        if not isinstance(__object, Figure):
            raise ValueError('May append only geometric objects')

        return super().append(__object)


class LineCollection(FigureCollection):
    def __init__(self, lines):
        for item in lines:
            if not isinstance(item, Line):
                raise ValueError('Collection may contain only lines')

        super().__init__(lines)

    def append(self, __object):
        if not isinstance(__object, Line):
            raise ValueError('May append only lines')

        return super().append(__object)

    def clusterize(self, cluster_number):
        lines_polar = np.array([line.polar for line in self])
        features = np.hstack([
            lines_polar[:, 0:1] / max(lines_polar[:, 0]),
            (np.sin(2 * lines_polar[:, 1:2]) + 1) / 2,
            (np.cos(2 * lines_polar[:, 1:2]) + 1) / 2,
        ])

        link_matrix = sch.linkage(features, 'centroid', 'euclidean')
        clusters = sch.cut_tree(link_matrix, cluster_number).ravel()
        clusters_argsort = clusters.argsort()
        groups = np.split(
            np.array(self)[clusters_argsort],
            np.unique(clusters[clusters_argsort], return_index=True)[1][1:]
        )

        return [LineCollection(group) for group in groups]

    def rotate(self, angle):
        for line in self:
            line.rotate(angle)

        return self

    def mean(self):
        rotate = False

        for line in self:
            # if line.is_vertical:
            if 0 <= abs(line.theta_deg) <= 45 or 135 <= abs(line.theta_deg) <= 180:
                # return line
                rotate = True
                self.rotate(math.pi/2)
                break

        x1 = 0
        x2 = 1e6

        points = np.array(
            [[x1, line.y(x1), x2, line.y(x2)]
             for line in self]
        ).reshape(-1, 2)
        params = np.linalg.lstsq(np.hstack([points[:, :1], np.ones((points.shape[0], 1),)]), points[:, 1:2], rcond=-1)

        mean_line = Line.by_params(*params[0].ravel())

        if rotate:
            mean_line.rotate(-math.pi/2)
            self.rotate(-math.pi/2)

        return mean_line

    def draw(self, image, color, thickness, continuous=False, origin=(0, 0)):
        for line in self:
            line.draw(image, color, thickness, continuous, origin)

        return image

    def intersections(self):
        intersections = np.empty((len(self), len(self), 2), np.float32)

        for l1, line1 in enumerate(self):
            for l2, line2 in enumerate(self):
                intersections[l1, l2] = line1.intersection(line2)

        return intersections
