import meshcat
import meshcat.geometry as g
import numpy as np


np.set_printoptions(precision=5, suppress=True)

COLORS = {
    'r': 0xff0000,
    'g': 0x00ff00,
    'b': 0x0000ff,
}

def get_visualizer(zmq_url: str ="tcp://127.0.0.1:6000"):
    '''
    default meshcat zmq url provided so you can just leave an instance running
    '''
    return meshcat.Visualizer(zmq_url=zmq_url)

def draw_point_cloud(vis: meshcat.Visualizer, key: str, points: np.array, colors: np.array, size: float = 0.01):
    # points must be 3 x N
    vis[key].set_object(g.Points(g.PointsGeometry(points, color=colors), g.PointsMaterial(size=size)))

def draw_transform(vis: meshcat.Visualizer, key: str, transform: np.array, length: float = 0.5, linewidth: float = 1.0):
    origin = transform[:3, 3]
    x = transform[:3, 0]
    y = transform[:3, 1]
    z = transform[:3, 2]

    assert np.isclose(np.linalg.norm(x), 1.0)
    assert np.isclose(np.linalg.norm(y), 1.0)
    assert np.isclose(np.linalg.norm(z), 1.0)

    x1 = origin + length * x
    y1 = origin + length * y
    z1 = origin + length * z

    x_axis = np.vstack([origin, x1]).T
    y_axis = np.vstack([origin, y1]).T
    z_axis = np.vstack([origin, z1]).T

    vis[key]['x'].set_object(g.Line(g.PointsGeometry(x_axis), g.MeshBasicMaterial(color=COLORS['r'], linewidth=linewidth)))
    vis[key]['y'].set_object(g.Line(g.PointsGeometry(y_axis), g.MeshBasicMaterial(color=COLORS['g'], linewidth=linewidth)))
    vis[key]['z'].set_object(g.Line(g.PointsGeometry(z_axis), g.MeshBasicMaterial(color=COLORS['b'], linewidth=linewidth)))


# TODO clean this up for just open pose
# def draw_body_skeleton(vis: meshcat.Visualizer, joints: np.array, dbg: bool=False):
#     # joints is a 3 x 49 np array
#     skeleton_str = 'skeleton_dbg' if dbg else 'skeleton'
#     vis[skeleton_str]['openpose']['kpts'].set_object(g.Points( g.PointsGeometry(joints[:, :25], color=joints[:, :25]),g.PointsMaterial(size=0.01)))
#     vis[skeleton_str]['spin']['kpts'].set_object(g.Points( g.PointsGeometry(joints[:, 25:], color=joints[:, 25:]),g.PointsMaterial(size=0.01)))

#     bb_max = np.max(joints, axis=1)
#     bb_min = np.min(joints, axis=1)

#     if dbg:
#         print(f"-----body 3d bounding box:")
#         print(f"x: {bb_max[0]}, {bb_min[0]}, delta: {bb_max[0] - bb_min[0]}")
#         print(f"y: {bb_max[1]}, {bb_min[1]}, delta: {bb_max[1] - bb_min[1]}")
#         print(f"z: {bb_max[2]}, {bb_min[2]}, delta: {bb_max[2] - bb_min[2]}")

#     '''
#     copied from frankmocap/renderer/glViewer.py
#     '''
#     #Openpose25 + SPIN global 24
#     link_openpose = [  [8,1], [1,0] , [0,16] , [16,18] , [0,15], [15,17],
#                 [1,2],[2,3],[3,4],      #Right Arm
#                 [1,5], [5,6], [6,7],       #Left Arm
#                 [8,12], [12,13], [13,14], [14,21], [14,19], [14,20],
#                 [8,9], [9,10], [10,11], [11,24], [11,22], [11,23]
#                 ]

#     link_spin24 =[  [14,16], [16,12], [12,17] , [17,18] ,
#                 [12,9],[9,10],[10,11],      #Right Arm
#                 [12,8], [8,7], [7,6],       #Left Arm
#                 [14,3], [3,4], [4,5],
#                 [14,2], [2,1], [1,0]
#                 ]
#     link_spin24 = np.array(link_spin24) + 25
#     '''
#     end copied from frankmocap/renderer/glViewer.py
#     '''

#     # OpenPose skeleton
#     for conn in link_openpose:
#         point_pair = np.zeros((3, 2), dtype=np.float32)
#         point_pair[:, 0] = joints[:, conn[0]]
#         point_pair[:, 1] = joints[:, conn[1]]
#         vis[skeleton_str][f'openpose'][f'{JOINT_NAMES[conn[0]]} to {JOINT_NAMES[conn[1]]}'].set_object(g.LineSegments(g.PointsGeometry(point_pair)))

#     # SPIN skeleton
#     for conn in link_spin24:
#         point_pair = np.zeros((3, 2), dtype=np.float32)
#         point_pair[:, 0] = joints[:, conn[0]]
#         point_pair[:, 1] = joints[:, conn[1]]
#         vis[skeleton_str][f'spin'][f'{JOINT_NAMES[conn[0]]} to {JOINT_NAMES[conn[1]]}'].set_object(g.LineSegments(g.PointsGeometry(point_pair)))

#     # Spine to ground projection
#     point_pair = np.zeros((3, 2), dtype=np.float32)
#     conn = [0,1]
#     point_pair[:, 0] = joints[:, conn[0]]
#     point_pair[:, 1] = joints[:, conn[1]]

#     vis[skeleton_str][f'base'][f'{JOINT_NAMES[conn[0]]} to {JOINT_NAMES[conn[1]]}'].set_object(g.LineSegments(g.PointsGeometry(point_pair)))


# def draw_hand_skeleton(vis: meshcat.Visualizer, joints: np.array, is_left_hand: bool, dbg: bool=False):
#     hand_str = 'hand_dbg' if dbg else 'hand'

#     # joints is a 3 x 21 np array
#     vis[hand_str]['left' if is_left_hand else 'right']['skeleton'].set_object(g.Points( g.PointsGeometry(joints, color=joints),g.PointsMaterial(size=0.01)))

#     '''
#     copied from frankmocap/renderer/glViewer.py
#     '''
#     g_connMat_hand21 = [ [0,1], [1,2], [2,3], [3,4],
#                         [0,5], [5,6], [6,7], [7,8],
#                         [0,9], [9,10], [10,11],[11,12],
#                         [0,13], [13,14], [14, 15], [15, 16],
#                         [0, 17], [17, 18], [18, 19], [19, 20]]
#     g_connMat_hand21 = np.array(g_connMat_hand21, dtype=int)
#     '''
#     end copied from frankmocap/renderer/glViewer.py
#     '''

#     for conn in g_connMat_hand21:
#         point_pair = np.zeros((3, 2), dtype=np.float32)
#         point_pair[:, 0] = joints[:, conn[0]]
#         point_pair[:, 1] = joints[:, conn[1]]
#         vis[hand_str]['left' if is_left_hand else 'right'][f'{conn[0]} to {conn[1]}'].set_object(g.LineSegments(g.PointsGeometry(point_pair)))

#     bb_max = np.max(joints, axis=1)
#     bb_min = np.min(joints, axis=1)

#     if dbg:
#         print(f"-----{'left hand' if is_left_hand else 'right hand'} 3d bounding box:")
#         print(f"x: {bb_max[0]}, {bb_min[0]}, delta: {bb_max[0] - bb_min[0]}")
#         print(f"y: {bb_max[1]}, {bb_min[1]}, delta: {bb_max[1] - bb_min[1]}")
#         print(f"z: {bb_max[2]}, {bb_min[2]}, delta: {bb_max[2] - bb_min[2]}")


def test():
    vis = get_visualizer()

    vis["box1"].set_object(g.Box([0.1, 0.2, 0.3]))

    identity = np.eye(4)
    identity[3, 2] = 1.0
    draw_transform(vis, 'origin', identity, 0.1)


    from meshcat.animation import Animation
    import meshcat.transformations as tf
    anim = Animation()

    with anim.at_frame(vis, 0) as frame:
        # `frame` behaves like a Visualizer, in that we can
        # call `set_transform` and `set_property` on it, but
        # it just stores information inside the animation
        # rather than changing the current visualization
        frame["box1"].set_transform(tf.translation_matrix([0, 0, 0]))
    with anim.at_frame(vis, 30) as frame:
        frame["box1"].set_transform(tf.translation_matrix([0, 1, 0]))

    # `set_animation` actually sends the animation to the
    # viewer. By default, the viewer will play the animation
    # right away. To avoid that, you can also pass `play=false`.
    vis.set_animation(anim)

    input("Done?")


if __name__ == "__main__":
    test()