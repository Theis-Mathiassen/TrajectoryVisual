# This code is based on:
# https://github.com/yumengs-exp/MLSimp/blob/main/Utils/cluster.py
# https://github.com/yumengs-exp/MLSimp/blob/main/Utils/partition.py


import math
from typing import Tuple
import numpy as np
from collections import deque, defaultdict

eps = 1e-12   # defined the segment length theta, if length < eps then l_h=0

min_traj_cluster = 2

def neighborhood(seg, segs, epsilon=2.0):
    segment_set = []
    for segment_tmp in segs:
        seg_long, seg_short = compare(seg, segment_tmp)  # get long segment by compare segment
        if seg_long.get_all_distance(seg_short) <= epsilon:
            segment_set.append(segment_tmp)
    return segment_set


def expand_cluster(segs, queue: deque, cluster_id: int, epsilon: float, min_lines: int):
    while len(queue) != 0:
        curr_seg = queue.popleft()
        curr_num_neighborhood = neighborhood(curr_seg, segs, epsilon=epsilon)
        if len(curr_num_neighborhood) >= min_lines:
            for m in curr_num_neighborhood:
                if m.cluster_id == -1:
                    queue.append(m)
                    m.cluster_id = cluster_id
        else:
            pass

def line_segment_clustering(traj_segments, epsilon: float = 2.0, min_lines: int = 5):
    cluster_id = 0
    cluster_dict = defaultdict(list)
    for seg in traj_segments:
        _queue = deque(list(), maxlen=50)
        if seg.cluster_id == -1:
            seg_num_neighbor_set = neighborhood(seg, traj_segments, epsilon=epsilon)
            if len(seg_num_neighbor_set) >= min_lines:
                seg.cluster_id = cluster_id
                for sub_seg in seg_num_neighbor_set:
                    sub_seg.cluster_id = cluster_id  # assign clusterId to segment in neighborhood(seg)
                    _queue.append(sub_seg)  # insert sub segment into queue
                expand_cluster(traj_segments, _queue, cluster_id, epsilon, min_lines)
                cluster_id += 1
            else:
                seg.cluster_id = -1
        # print(seg.cluster_id, seg.traj_id)
        if seg.cluster_id != -1:
            cluster_dict[seg.cluster_id].append(seg)

    remove_cluster = dict()
    cluster_number = len(cluster_dict)
    for i in range(0, cluster_number):
        traj_num = len(set(map(lambda s: s.traj_id, cluster_dict[i])))
        #print("the %d cluster lines:" % i, traj_num)
        if traj_num < min_traj_cluster:
            remove_cluster[i] = cluster_dict.pop(i)
    return cluster_dict, remove_cluster


def representative_trajectory_generation(cluster_segment: dict, min_lines: int = 3, min_dist: float = 2.0):
    representive_point = defaultdict(list)
    for i in cluster_segment.keys():
        cluster_size = len(cluster_segment.get(i))
        sort_point = []  # [Point, ...], size = cluster_size*2
        rep_point, zero_point = Point(0, 0, -1), Point(1, 0, -1)

        for j in range(cluster_size):
            rep_point = rep_point + (cluster_segment[i][j].end - cluster_segment[i][j].start)
        rep_point = rep_point / float(cluster_size)

        cos_theta = rep_point.dot(zero_point) / rep_point.distance(Point(0, 0, -1))  # cos(theta)
        sin_theta = math.sqrt(1 - math.pow(cos_theta, 2))  # sin(theta)

        for j in range(cluster_size):
            s, e = cluster_segment[i][j].start, cluster_segment[i][j].end
            cluster_segment[i][j] = Segment(Point(s.x * cos_theta + s.y * sin_theta, s.y * cos_theta - s.x * sin_theta, -1),
                                            Point(e.x * cos_theta + e.y * sin_theta, e.y * cos_theta - e.x * sin_theta, -1),
                                            traj_id=cluster_segment[i][j].traj_id,
                                            cluster_id=cluster_segment[i][j].cluster_id)
            sort_point.extend([cluster_segment[i][j].start, cluster_segment[i][j].end])

        sort_point = sorted(sort_point, key=lambda _p: _p.x)
        for p in range(len(sort_point)):
            intersect_cnt = 0.0
            start_y = Point(0, 0, -1)
            for q in range(cluster_size):
                s, e = cluster_segment[i][q].start, cluster_segment[i][q].end
                if (sort_point[p].x <= e.x) and (sort_point[p].x >= s.x):
                    if s.x == e.x:
                        continue
                    elif s.y == e.y:
                        intersect_cnt += 1
                        start_y = start_y + Point(sort_point[p].x, s.y, -1)
                    else:
                        intersect_cnt += 1
                        start_y = start_y + Point(sort_point[p].x, (e.y-s.y)/(e.x-s.x)*(sort_point[p].x-s.x)+s.y, -1)
            if intersect_cnt >= min_lines:
                tmp_point: Point = start_y / intersect_cnt
                tmp = Point(tmp_point.x*cos_theta-sin_theta*tmp_point.y,
                            sin_theta*tmp_point.x+cos_theta*tmp_point.y, -1)
                _size = len(representive_point[i]) - 1
                if _size < 0 or (_size >= 0 and tmp.distance(representive_point[i][_size]) > min_dist):
                    representive_point[i].append(tmp)
    return representive_point

class Point(object):
    def __init__(self, x, y, traj_id=None):
        self.trajectory_id = traj_id
        self.x = x
        self.y = y

    def __repr__(self):
        return "{0:.8f},{1:.8f}".format(self.x, self.y)

    def get_point(self):
        return self.x, self.y

    def __add__(self, other: 'Point'):
        if not isinstance(other, Point):
            raise TypeError("The other type is not 'Point' type.")
        _add_x = self.x + other.x
        _add_y = self.y + other.y
        return Point(_add_x, _add_y, traj_id=self.trajectory_id)

    def __sub__(self, other: 'Point'):
        if not isinstance(other, Point):
            raise TypeError("The other type is not 'Point' type.")
        _sub_x = self.x - other.x
        _sub_y = self.y - other.y
        return Point(_sub_x, _sub_y, traj_id=self.trajectory_id)

    def __mul__(self, x: float):
        if isinstance(x, float):
            return Point(self.x*x, self.y*x, traj_id=self.trajectory_id)
        else:
            raise TypeError("The other object must 'float' type.")

    def __truediv__(self, x: float):
        if isinstance(x, float):
            return Point(self.x / x, self.y / x, traj_id=self.trajectory_id)
        else:
            raise TypeError("The other object must 'float' type.")

    def distance(self, other: 'Point'):
        return math.sqrt(math.pow(self.x-other.x, 2) + math.pow(self.y-other.y, 2))

    def dot(self, other: 'Point'):
        return self.x * other.x + self.y * other.y

    def as_array(self):
        return np.array((self.x, self.y))


def _point2line_distance(point, start, end):
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)
    return np.divide(np.abs(np.linalg.norm(np.cross(end - start, start - point))),
                     np.linalg.norm(end - start))


class Segment(object):
    eps = 1e-12

    def __init__(self, start_point: Point, end_point: Point, traj_id: int = None, cluster_id: int = -1):
        self.start = start_point
        self.end = end_point
        self.traj_id = traj_id
        self.cluster_id = cluster_id

    def set_cluster(self, cluster_id: int):
        self.cluster_id = cluster_id

    def pair(self) -> Tuple[Point, Point]:
        return self.start, self.end

    @property
    def length(self):
        return self.end.distance(self.start)

    def perpendicular_distance(self, other: 'Segment'):
        l1 = other.start.distance(self._projection_point(other, typed="start"))
        l2 = other.end.distance(self._projection_point(other, typed="end"))
        if l1 < self.eps and l2 < self.eps:
            return 0
        else:
            return (math.pow(l1, 2) + math.pow(l2, 2)) / (l1 + l2)

    def parallel_distance(self, other: 'Segment'):
        l1 = self.start.distance(self._projection_point(other, typed='start'))
        l2 = self.end.distance(self._projection_point(other, typed='end'))
        return min(l1, l2)

    def angle_distance(self, other: 'Segment'):
        self_vector = self.end - self.start
        self_dist, other_dist = self.end.distance(self.start), other.end.distance(other.start)

        if self_dist < self.eps:
            return _point2line_distance(self.start.as_array(), other.start.as_array(), other.end.as_array())
        elif other_dist < self.eps:
            return _point2line_distance(other.start.as_array(), self.start.as_array(), self.end.as_array())

        cos_theta = self_vector.dot(other.end - other.start) / (
                    self.end.distance(self.start) * other.end.distance(other.start))
        if cos_theta > self.eps:
            if cos_theta >= 1:
                cos_theta = 1.0
            return other.length * math.sqrt(1 - math.pow(cos_theta, 2))
        else:
            return other.length

    def _projection_point(self, other: 'Segment', typed="e"):
        if typed == 's' or typed == 'start':
            tmp = other.start - self.start
        else:
            tmp = other.end - self.start
        u = tmp.dot(self.end - self.start) / max(math.pow(self.end.distance(self.start), 2), 0.000001)
        return self.start + (self.end - self.start) * u

    def get_all_distance(self, seg: 'Segment'):
        res = self.angle_distance(seg)
        if str(self.start) != str(self.end):
            res += self.parallel_distance(seg)
        if self.traj_id != seg.traj_id:
            res += self.perpendicular_distance(seg)
        return res


def compare(segment_a: Segment, segment_b: Segment) -> Tuple[Segment, Segment]:
    return (segment_a, segment_b) if segment_a.length > segment_b.length else (segment_b, segment_a)

def segment_mdl_comp(traj, start_index, current_index, typed='par'):
    length_hypothesis = 0
    length_data_hypothesis_perpend = 0
    length_data_hypothesis_angle = 0

    seg = Segment(traj[start_index], traj[current_index])
    if typed == "par" or typed == "PAR":
        if seg.length < eps:
            length_hypothesis = 0
        else:
            length_hypothesis = math.log2(seg.length)

    # compute the segment hypothesis
    for i in range(start_index, current_index, 1):
        sub_seg = Segment(traj[i], traj[i+1])
        if typed == 'par' or typed == 'PAR':
            length_data_hypothesis_perpend += seg.perpendicular_distance(sub_seg)
            length_data_hypothesis_angle += seg.angle_distance(sub_seg)
        elif typed == "nopar" or typed == "NOPAR":
            length_hypothesis += sub_seg.length

    if typed == 'par' or typed == 'PAR':
        if length_data_hypothesis_perpend > eps:
            length_hypothesis += math.log2(length_data_hypothesis_perpend)
        if length_data_hypothesis_angle > eps:
            length_hypothesis += math.log2(length_data_hypothesis_angle)
        return length_hypothesis
    elif typed == "nopar" or typed == "NOPAR":
        if length_hypothesis < eps:
            return 0
        else:
            return math.log2(length_hypothesis)  # when typed == nopar the L(D|H) is zero.
    else:
        raise ValueError("The parameter 'typed' given value has error!")


def approximate_trajectory_partitioning(traj, traj_id=None, theta=5.0):
    size = len(traj)
    start_index: int = 0; length: int = 1

    partition_trajectory = []
    while (start_index + length) < size:
        curr_index = start_index + length
        cost_par = segment_mdl_comp(traj, start_index, curr_index, typed='par')
        cost_nopar = segment_mdl_comp(traj, start_index, curr_index, typed='nopar')
        if cost_par > (cost_nopar+theta):
            seg = Segment(traj[start_index], traj[curr_index-1], traj_id=traj_id)
            partition_trajectory.append(seg)
            start_index = curr_index - 1
            length = 1
        else:
            length += 1
    seg = Segment(traj[start_index], traj[size-1], traj_id=traj_id, cluster_id=-1)
    partition_trajectory.append(seg)
    return partition_trajectory


def rdp_trajectory_partitioning(trajectory, traj_id=None, epsilon=1.0):
    size = len(trajectory)
    d_max = 0.0
    index = 0
    for i in range(1, size-1, 1):
        d = _point2line_distance(trajectory[i].as_array(), trajectory[0].as_array(), trajectory[-1].as_array())
        if d > d_max:
            d_max = d
            index = i

    if d_max > epsilon:
        result = rdp_trajectory_partitioning(trajectory[:index+1], epsilon=epsilon) + \
                 rdp_trajectory_partitioning(trajectory[index:], epsilon=epsilon)
    else:
        result = [Segment(trajectory[0], trajectory[-1], traj_id=traj_id, cluster_id=-1)]
    return result

