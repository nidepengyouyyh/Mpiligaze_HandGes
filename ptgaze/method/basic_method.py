import math

# 计算两个点之间的距离：L2距离（欧式距离）
def points_distance(x0, y0, x1, y1):
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

# 计算两条线段之间的夹角，以弧度表示
def compute_angle(x0, y0, x1, y1, x2, y2, x3, y3):
    AB = [x1 - x0, y1 - y0]
    CD = [x3 - x2, y3 - y2]

    dot_product = AB[0] * CD[0] + AB[1] * CD[1]

    AB_distance = points_distance(x0, y0, x1, y1) + 0.001   # 防止分母出现0
    CD_distance = points_distance(x2, y2, x3, y3) + 0.001

    cos_theta = dot_product / (AB_distance * CD_distance)

    theta = math.acos(cos_theta)

    return theta

def detect_all_finger_state(all_points):
    finger_first_angle_bend_threshold = math.pi * 0.25  # 大拇指弯曲阈值
    finger_other_angle_bend_threshold = math.pi * 0.5  # 其他手指弯曲阈值
    finger_other_angle_straighten_threshold = math.pi * 0.2  # 其他手指伸直阈值

    first_is_bend = False
    first_is_straighten = False
    second_is_bend = False
    second_is_straighten = False
    third_is_bend = False
    third_is_straighten = False
    fourth_is_bend = False
    fourth_is_straighten = False
    fifth_is_bend = False
    fifth_is_straighten = False

    finger_first_angle = compute_angle(all_points['point0'][0], all_points['point0'][1], all_points['point1'][0], all_points['point1'][1],
                                        all_points['point2'][0], all_points['point2'][1], all_points['point4'][0], all_points['point4'][1])
    finger_second_angle = compute_angle(all_points['point0'][0], all_points['point0'][1], all_points['point5'][0], all_points['point5'][1],
                                        all_points['point6'][0], all_points['point6'][1], all_points['point8'][0], all_points['point8'][1])
    finger_third_angle = compute_angle(all_points['point0'][0], all_points['point0'][1], all_points['point9'][0], all_points['point9'][1],
                                        all_points['point10'][0], all_points['point10'][1], all_points['point12'][0], all_points['point12'][1])
    finger_fourth_angle = compute_angle(all_points['point0'][0], all_points['point0'][1], all_points['point13'][0], all_points['point13'][1],
                                        all_points['point14'][0], all_points['point14'][1], all_points['point16'][0], all_points['point16'][1])
    finger_fifth_angle = compute_angle(all_points['point0'][0], all_points['point0'][1], all_points['point17'][0], all_points['point17'][1],
                                       all_points['point18'][0], all_points['point18'][1], all_points['point20'][0], all_points['point20'][1])

    if finger_first_angle > finger_first_angle_bend_threshold:              # 判断大拇指是否弯曲
        first_is_bend = True
        first_is_straighten = False
    else:
        first_is_bend = False
        first_is_straighten = True

    if finger_second_angle > finger_other_angle_bend_threshold:            # 判断食指是否弯曲
        second_is_bend = True
    elif finger_second_angle < finger_other_angle_straighten_threshold:
        second_is_straighten = True
    else:
        second_is_bend = False
        second_is_straighten = False

    if finger_third_angle > finger_other_angle_bend_threshold:              # 判断中指是否弯曲
        third_is_bend = True
    elif finger_third_angle < finger_other_angle_straighten_threshold:
        third_is_straighten = True
    else:
        third_is_bend = False
        third_is_straighten = False

    if finger_fourth_angle > finger_other_angle_bend_threshold:             # 判断无名指是否弯曲
        fourth_is_bend = True
    elif finger_fourth_angle < finger_other_angle_straighten_threshold:
        fourth_is_straighten = True
    else:
        fourth_is_bend = False
        fourth_is_straighten = False

    if finger_fifth_angle > finger_other_angle_bend_threshold:              # 判断小拇指是否弯曲
        fifth_is_bend = True
    elif finger_fifth_angle < finger_other_angle_straighten_threshold:
        fifth_is_straighten = True
    else:
        fifth_is_bend = False
        fifth_is_straighten = False

    # 将手指的弯曲或伸直状态存在字典中
    bend_states = {'first': first_is_bend, 'second': second_is_bend, 'third': third_is_bend, 'fourth': fourth_is_bend, 'fifth': fifth_is_bend}
    straighten_states = {'first': first_is_straighten, 'second': second_is_straighten, 'third': third_is_straighten, 'fourth': fourth_is_straighten, 'fifth': fifth_is_straighten}

    return bend_states, straighten_states

def check_for_index_only(straighten_states, bend_states):
    # 判断是否只有食指伸直，其他手指弯曲
    if (straighten_states['second'] == True and
        bend_states['first'] == True and
        bend_states['third'] == True and
        bend_states['fourth'] == True and
        bend_states['fifth'] == True):
        return True
    return False

# 示例：调用函数检查食指手势
# 假设你已经从MediaPipe获得了手部关键点 `all_points` 字典
# 例如: all_points = {'point0': (x0, y0), 'point1': (x1, y1), ...}

# 假设all_points是MediaPipe提供的手部关键点数据字典

# 如果你检测到只有食指伸直：


