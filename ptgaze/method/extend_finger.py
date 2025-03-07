import numpy as np

def line_from_points(p1, p2):
    """ 从两点生成直线方程 Ax + By + C = 0 """
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C

def point_line_distance(line, point):
    """ 计算点到直线的距离 """
    A, B, C = line
    x, y = point
    return abs(A * x + B * y + C) / np.sqrt(A**2 + B**2)

def is_point_in_box(point, box):
    """ 判断点是否在矩形框内 """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def does_line_intersect_box(line, box):
    """
    判断一条直线是否与物体框相交。

    :param line: 直线，格式为((x1, y1), (x2, y2))
    :param box: 物体框，格式为(x1, y1, x2, y2)，即矩形的两个对角线点
    :return: 如果直线与物体框相交，则返回True，否则返回False
    """
    # 物体框的四个角坐标
    box_points = [
        (box[0], box[1]),  # 左上
        (box[2], box[1]),  # 右上
        (box[2], box[3]),  # 右下
        (box[0], box[3])  # 左下
    ]

    # 判断直线是否与物体框的每条边相交
    for i in range(4):
        p1 = box_points[i]
        p2 = box_points[(i + 1) % 4]  # 获取下一点，形成边

        # 判断直线与物体框的边是否相交
        if do_lines_intersect(line[0], line[1], p1, p2):
            return True
    return False


def do_lines_intersect(p1, p2, q1, q2):
    """
    判断两条线段是否相交。

    :param p1, p2: 直线段1的两个端点
    :param q1, q2: 直线段2的两个端点
    :return: 如果两条线段相交，返回True，否则返回False
    """

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    return (cross(p1, q1, q2) * cross(p2, q1, q2) < 0) and (cross(q1, p1, p2) * cross(q2, p1, p2) < 0)

# 获取食指所在的直线并延长
def extend_line(line, img_height):
    """ 延长直线，确保从图像顶部到底部 """
    A, B, C = line
    # 获取直线的两个点，确保它跨越整个图像的高度
    y1 = 0
    x1 = int((-C - B * y1) / (A + 0.01))  # 计算y = 0时的x
    y2 = img_height
    x2 = int((-C - B * y2) / A)  # 计算y = img_height时的x
    return (x1, y1), (x2, y2)
