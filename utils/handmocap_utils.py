from typing import List

from datetime import datetime
import numpy as np
import torch

import cv2
import os
import os.path as osp

from frankmocap.demo.demo_options import DemoOptions
from frankmocap.bodymocap.body_mocap_api import BodyMocap
from frankmocap.handmocap.hand_mocap_api import HandMocap
import frankmocap.mocap_utils.demo_utils as demo_utils
import frankmocap.mocap_utils.general_utils as gnu
from frankmocap.mocap_utils.timer import Timer
from frankmocap.bodymocap.body_bbox_detector import BodyPoseEstimator
from frankmocap.handmocap.hand_bbox_detector import HandBboxDetector
from frankmocap.integration.copy_and_paste import integration_copy_paste

import frankmocap.renderer.image_utils as imu
from frankmocap.renderer.viewer2D import ImShow

from utils.meshcat_viewer import get_visualizer, draw_body_skeleton, draw_point_cloud, draw_hand_skeleton


class HandMocapWrapper(object):
    def __init__(self, render_video_out: bool=False, render_meshcat: bool=False, dbg: bool=False):
        self.frankmocap_args = DemoOptions().get_default(is_third_person=False)
        self.render_video_out = render_video_out

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        assert torch.cuda.is_available(), "Current version only supports GPU"

        print(self.frankmocap_args)
        self.hand_bbox_detector =  HandBboxDetector(self.frankmocap_args.view_type, device)

        #Set Mocap regressor
        self.hand_mocap = HandMocap(self.frankmocap_args.checkpoint_hand, self.frankmocap_args.smpl_dir, device = device)

        # Set Visualizer
        if self.frankmocap_args.renderer_type in ['pytorch3d', 'opendr']:
            from frankmocap.renderer.screen_free_visualizer import Visualizer
        else:
            from frankmocap.renderer.visualizer import Visualizer
        self.visualizer = Visualizer(self.frankmocap_args.renderer_type)

        # meshcat
        self.render_meshcat = render_meshcat
        if self.render_meshcat:
            self.meshcat_viz = get_visualizer()

    def set_new_video_path(self, video_path: str):
        self.frankmocap_args = DemoOptions().get_default(video_path=video_path, is_third_person=False)
        self.frankmocap_args.no_video_out = not self.render_video_out
        if video_path != "webcam":
            # hideous hack
            # vlog
            # relative_clip_dir = video_path.split('/data/pulkitag/data/dxyang/vlog/')[1].split('/clip.mp4')[0]
            # self.frankmocap_args.out_dir = f'/data/pulkitag/data/dxyang/vlog_output/{relative_clip_dir}'

            # ego4d
            # relative_clip_dir = video_path.split('/')[-1].split('.mp4')[0]
            # self.frankmocap_args.out_dir = f'/data/pulkitag/data/dxyang/ego4d_output/{relative_clip_dir}'

            # VIDEO CLIP
            relative_clip_dir = video_path.split('/')[-1].split('.mp4')[0]
            self.frankmocap_args.out_dir = os.path.expanduser(f"~/localdata/cam2hand/{relative_clip_dir}")

            if not os.path.exists(self.frankmocap_args.out_dir):
                os.makedirs(self.frankmocap_args.out_dir)
        else:
            self.frankmocap_args.out_dir = os.path.expanduser("~/localdata/cam2hand/0731022_webcam")

    def run(self):
        self.run_hand_mocap(self.frankmocap_args, self.hand_bbox_detector, self.hand_mocap, self.visualizer)

    def render_video_out(self):
        # save images as a video
        demo_utils.gen_video_out(self.frankmocap_args.out_dir, self.frankmocap_args.seq_name)

    def run_hand_mocap(self, args, bbox_detector, hand_mocap, visualizer):
        #Set up input data (images or webcam)
        input_type, input_data = demo_utils.setup_input(args)

        assert args.out_dir is not None, "Please specify output dir to store the results"
        cur_frame = args.start_frame
        video_frame = 0

        # skeleton_list = []

        while True:
            # load data
            load_bbox = False

            if input_type =='image_dir':
                if cur_frame < len(input_data):
                    image_path = input_data[cur_frame]
                    img_original_bgr  = cv2.imread(image_path)
                else:
                    img_original_bgr = None

            elif input_type == 'bbox_dir':
                if cur_frame < len(input_data):
                    image_path = input_data[cur_frame]['image_path']
                    hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                    body_bbox_list = input_data[cur_frame]['body_bbox_list']
                    img_original_bgr  = cv2.imread(image_path)
                    load_bbox = True
                else:
                    img_original_bgr = None

            elif input_type == 'video':
                _, img_original_bgr = input_data.read()
                if video_frame < cur_frame:
                    video_frame += 1
                    continue
                # save the obtained video frames
                image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
                if img_original_bgr is not None:
                    video_frame += 1
                    if args.save_frame:
                        gnu.make_subdir(image_path)
                        cv2.imwrite(image_path, img_original_bgr)

            elif input_type == 'webcam':
                _, img_original_bgr = input_data.read()

                if video_frame < cur_frame:
                    video_frame += 1
                    continue
                # save the obtained video frames
                image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
                if img_original_bgr is not None:
                    video_frame += 1
                    if True:
                        gnu.make_subdir(image_path)
                        cv2.imwrite(image_path, img_original_bgr)
            elif input_type == "realsense":
                img_original_rgb = input_data.get_latest_rgb()

                img_original_bgr = cv2.cvtColor(img_original_rgb, cv2.COLOR_RGB2BGR)

                if video_frame < cur_frame:
                    video_frame += 1
                    continue

                # save the obtained video frames
                image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
                if img_original_bgr is not None:
                    video_frame += 1
                    if args.save_frame:
                        gnu.make_subdir(image_path)
                        cv2.imwrite(image_path, img_original_bgr)
            else:
                assert False, "Unknown input_type"

            cur_frame +=1
            if img_original_bgr is None or cur_frame > args.end_frame:
                break
            # print("--------------------------------------")

            # bbox detection
            if load_bbox:
                body_pose_list = None
                raw_hand_bboxes = None
            elif args.crop_type == 'hand_crop':
                # hand already cropped, thererore, no need for detection
                img_h, img_w = img_original_bgr.shape[:2]
                body_pose_list = None
                raw_hand_bboxes = None
                hand_bbox_list = [ dict(right_hand = np.array([0, 0, img_w, img_h])) ]
            else:
                # Input images has other body part or hand not cropped.
                # Use hand detection model & body detector for hand detection
                assert args.crop_type == 'no_crop'
                detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
                body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = detect_output

            # save the obtained body & hand bbox to json file
            if args.save_bbox_output:
                # false by default
                demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

            # if len(hand_bbox_list) < 1:
            #     # print(f"No hand deteced: {image_path}")
            #     res_img = np.concatenate((img_original_bgr, img_original_bgr), axis=1)
            #     demo_utils.save_res_img(args.out_dir, image_path, res_img)
            #     # skeleton_list.append(np.zeros((1, 3, 49)))
            #     continue

            # Hand Pose Regression
            pred_output_list = hand_mocap.regress(
                    img_original_bgr, hand_bbox_list, add_margin=True)
            assert len(hand_bbox_list) == len(body_bbox_list)
            assert len(body_bbox_list) == len(pred_output_list)


            if self.render_meshcat:
                if pred_output_list[0]['left_hand'] is not None:
                    # point cloud of vertices
                    lh_verts_smpl = pred_output_list[0]['left_hand']['pred_vertices_smpl'].T # 3 x n
                    lh_verts_img = pred_output_list[0]['left_hand']['pred_vertices_img'].T / 1000.0 # 3 x n

                    # skeleton
                    lh_joints_smpl = pred_output_list[0]['left_hand']['pred_joints_smpl'].T # 3 x 21
                    lh_joints_img = pred_output_list[0]['left_hand']['pred_joints_img'].T / 1000.0 # 3 x 21
                else:
                    # point cloud of vertices
                    lh_verts_smpl = np.zeros((3, 0)) # 3 x n
                    lh_verts_img = np.zeros((3, 0)) # 3 x n

                    # skeleton
                    lh_joints_smpl = np.zeros((3, 21))
                    lh_joints_img = np.zeros((3, 21))

                draw_point_cloud(self.meshcat_viz, 'lh_verts_smpl', lh_verts_smpl, colors=lh_verts_smpl)
                draw_point_cloud(self.meshcat_viz, 'lh_verts_img', lh_verts_img, colors=lh_verts_img)
                draw_hand_skeleton(self.meshcat_viz, lh_joints_smpl, is_left_hand=True)
                draw_hand_skeleton(self.meshcat_viz, lh_joints_img, is_left_hand=True)

                if pred_output_list[0]['right_hand'] is not None:
                    # point cloud of vertices
                    rh_verts_smpl = pred_output_list[0]['right_hand']['pred_vertices_smpl'].T  # 3 x n
                    rh_verts_img = pred_output_list[0]['right_hand']['pred_vertices_img'].T / 1000.0  # 3 x n

                    # skeleton
                    rh_joints_smpl = pred_output_list[0]['right_hand']['pred_joints_smpl'].T # 3 x n
                    rh_joints_img = pred_output_list[0]['right_hand']['pred_joints_img'].T / 1000.0  # 3 x n

                else:
                    # point cloud of vertices
                    rh_verts_smpl = np.zeros((3, 0)) # 3 x n
                    rh_verts_img = np.zeros((3, 0)) # 3 x n

                    # skeleton
                    rh_joints_smpl = np.zeros((3, 21)) # 3 x n
                    rh_joints_img = np.zeros((3, 21)) # 3 x n

                draw_point_cloud(self.meshcat_viz, 'rh_verts_smpl', rh_verts_smpl, colors=rh_verts_smpl)
                draw_point_cloud(self.meshcat_viz, 'rh_verts_img', rh_verts_img, colors=rh_verts_img)
                draw_hand_skeleton(self.meshcat_viz, rh_joints_smpl, is_left_hand=False)
                draw_hand_skeleton(self.meshcat_viz, rh_joints_img, is_left_hand=False)


            # extract mesh for rendering (vertices in image space and faces) from pred_output_list
            pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

            # # visualize
            dbg_res_img = visualizer.visualize(
                img_original_bgr,
                pred_mesh_list = pred_mesh_list,
                hand_bbox_list = hand_bbox_list,
                vis_raw_hand_bbox = False)
            res_img = visualizer.visualize(
                img_original_bgr,
                pred_mesh_list = pred_mesh_list,
                hand_bbox_list = hand_bbox_list,
                vis_raw_hand_bbox = False,
                black_bg=True)

            # show result in the screen
            if not args.no_display:
                res_img = res_img.astype(np.uint8)
                ImShow(res_img)

            # save result image
            if len(pred_mesh_list) != 0:
                if args.out_dir is not None:
                    demo_utils.save_res_img(args.out_dir, image_path, res_img)
                    demo_utils.save_res_img(args.out_dir, image_path, dbg_res_img, append_str="overlay")

            # save predictions to pkl
            if args.save_pred_pkl:
                demo_type = 'hand'
                demo_utils.save_pred_to_pkl(
                    args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

            # print(f"Processed : {image_path}")

        # save the skeleton array for all frames
        # body_skeletons_np = np.concatenate(skeleton_list, axis=0)
        # np.save(f"{args.out_dir}/skeletons.npy", body_skeletons_np)

        # save images as a video
        if not args.no_video_out and input_type in ['video', 'webcam']:
            demo_utils.gen_video_out(args.out_dir, args.seq_name)

        if input_type =='webcam' and input_data is not None:
            input_data.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    vid_str = "webcam"
    render_meshcat = True
    render_video_out = True
    hand_mocap_engine = HandMocapWrapper(render_video_out=render_video_out, render_meshcat=render_meshcat)
    hand_mocap_engine.set_new_video_path(vid_str)
    hand_mocap_engine.run()
